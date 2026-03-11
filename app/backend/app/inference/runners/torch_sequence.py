from __future__ import annotations

import json
import pickle
import time

from app.inference.base import BaseModelRunner, InferencePrediction, ProbabilityScore
from app.inference.preprocessing import preprocess_sequence_text, texts_to_sequences
from app.registry.contracts import RegisteredModel


class TorchBiLSTMAttentionRunner(BaseModelRunner):
    def __init__(self, registered_model: RegisteredModel) -> None:
        import numpy as np
        import torch

        self._np = np
        self._torch = torch
        self._registered_model = registered_model
        self._manifest = registered_model.manifest
        self._device = _resolve_device(torch, self._manifest.runtime.device)
        self._labels = self._load_labels()
        self._vocab = self._load_vocab()
        self._model = self._load_model()

    def predict(self, text: str) -> InferencePrediction:
        started = time.perf_counter()
        processed = preprocess_sequence_text(text)
        max_len = self._manifest.runtime.max_sequence_length
        token_count = len(processed.split())
        sequence = texts_to_sequences([processed], self._vocab, max_len)
        tensor = self._torch.tensor(sequence, dtype=self._torch.long, device=self._device)
        with self._torch.no_grad():
            logits = self._model(tensor).squeeze(0)

        probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
        top_idx = int(self._torch.argmax(logits).item())
        ended = time.perf_counter()

        return InferencePrediction(
            predicted_label=self._labels[top_idx],
            confidence=float(probabilities[top_idx]),
            probabilities=[
                ProbabilityScore(label=self._labels[index], score=float(score))
                for index, score in enumerate(probabilities)
            ]
            if self._manifest.inference.return_probabilities
            else None,
            latency_ms=round((ended - started) * 1000, 2),
            sequence_length_used=min(token_count, max_len),
            was_truncated=token_count > max_len,
        )

    def _load_vocab(self) -> dict[str, int]:
        vocab_path = self._registered_model.artifact_resolution.vocabulary[0]
        with vocab_path.open("rb") as handle:
            vocab = pickle.load(handle)
        return dict(vocab)

    def _load_labels(self) -> dict[int, str]:
        artifact = self._registered_model.artifact_resolution.label_classes_file
        if artifact and artifact.exists():
            labels = json.loads(artifact.read_text(encoding="utf-8"))
            return {index: label for index, label in enumerate(labels)}
        return {
            label.id: label.effective_name
            for label in (self._manifest.labels or [])
        }

    def _load_model(self):
        checkpoint_path = self._registered_model.artifact_resolution.weights[0]
        checkpoint = _torch_load(self._torch, checkpoint_path, self._device)
        state_dict = checkpoint.get("model_state", checkpoint)

        embedding_weights = state_dict["embedding.weight"]
        embedding_shape = tuple(embedding_weights.shape)
        embedding_matrix = self._np.zeros(embedding_shape, dtype=self._np.float32)
        model = BiLSTMAttention(
            torch_module=self._torch,
            embedding_matrix=embedding_matrix,
            num_classes=len(self._labels),
            hidden_dim=int(self._manifest.model.get("hidden_dim", 128)),
            num_layers=int(self._manifest.model.get("num_layers", 2)),
            dropout=float(self._manifest.model.get("dropout", 0.3)),
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self._device)
        model.eval()
        return model


def _torch_load(torch_module, checkpoint_path, device):
    try:
        return torch_module.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint_path, map_location=device)


def _resolve_device(torch_module, runtime_device: str) -> str:
    if runtime_device == "cpu":
        return "cpu"
    if runtime_device == "cuda" and torch_module.cuda.is_available():
        return "cuda"
    if runtime_device == "mps" and getattr(torch_module.backends, "mps", None):
        if torch_module.backends.mps.is_available():
            return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_attention_module(torch_module, hidden_dim: int):
    class AdditiveAttention(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch_module.nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, hidden_states, mask=None):
            scores = self.attn(hidden_states).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)
            weights = torch_module.softmax(scores, dim=1)
            context = (weights.unsqueeze(-1) * hidden_states).sum(dim=1)
            return context, weights

    return AdditiveAttention()


def BiLSTMAttention(
    torch_module,
    embedding_matrix,
    num_classes: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
):
    class _BiLSTMAttention(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch_module.nn.Embedding.from_pretrained(
                torch_module.FloatTensor(embedding_matrix), freeze=False, padding_idx=0
            )
            _, embedding_dim = embedding_matrix.shape
            self.lstm = torch_module.nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attention = _build_attention_module(torch_module, hidden_dim * 2)
            self.dropout = torch_module.nn.Dropout(dropout)
            self.fc = torch_module.nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            mask = x != 0
            embeddings = self.dropout(self.embedding(x))
            hidden_states, _ = self.lstm(embeddings)
            context, _ = self.attention(hidden_states, mask)
            return self.fc(self.dropout(context))

    return _BiLSTMAttention()

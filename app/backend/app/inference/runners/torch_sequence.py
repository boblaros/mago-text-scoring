from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.inference.base import BaseModelRunner, InferencePrediction, ProbabilityScore
from app.inference.preprocessing import apply_text_preprocessing, texts_to_sequences
from app.inference.runtime_support import (
    SERIALIZED_ARTIFACT_EXTENSIONS,
    decode_label_value,
    load_label_records,
    load_optional_label_encoder,
    load_serialized_artifact,
)
from app.registry.contracts import RegisteredModel


ARCHITECTURE_ALIASES = {
    "bilstm": "bilstm-attention",
    "bilstm_attention": "bilstm-attention",
    "bilstm-attention": "bilstm-attention",
    "glove_bilstm": "bilstm-attention",
    "glove-bilstm": "bilstm-attention",
    "cnn": "text-cnn",
    "glove_cnn": "text-cnn",
    "glove-cnn": "text-cnn",
    "textcnn": "text-cnn",
    "text_cnn": "text-cnn",
    "text-cnn": "text-cnn",
    "embedding_mlp": "embedding-mlp",
    "embedding-mlp": "embedding-mlp",
    "glove_mlp": "embedding-mlp",
    "glove-mlp": "embedding-mlp",
    "mlp": "embedding-mlp",
}


class TorchSequenceClassifierRunner(BaseModelRunner):
    def __init__(self, registered_model: RegisteredModel) -> None:
        import numpy as np
        import torch

        self._np = np
        self._torch = torch
        self._registered_model = registered_model
        self._manifest = registered_model.manifest
        self._device = _resolve_device(torch, self._manifest.runtime.device)
        self._artifact_labels = (
            load_label_records(registered_model.artifact_resolution.label_classes_file)
            or load_label_records(registered_model.artifact_resolution.label_map_file)
        )
        self._manifest_labels = {
            label.id: label.effective_name
            for label in (self._manifest.labels or [])
        }
        self._label_encoder = load_optional_label_encoder(
            registered_model.artifact_resolution.label_encoder_file
        )
        self._vocab = self._load_vocab()
        self._architecture = ""
        self._model = self._load_model()

    def predict(self, text: str) -> InferencePrediction:
        started = time.perf_counter()
        processed = apply_text_preprocessing(
            text,
            self._manifest.runtime.preprocessing,
            default_steps=("preprocess_sequence_text",),
        )
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
            predicted_label=self._decode_label(top_idx),
            confidence=float(probabilities[top_idx]),
            probabilities=[
                ProbabilityScore(label=self._decode_label(index), score=float(score))
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
        suffix = vocab_path.suffix.lower()

        if suffix == ".json":
            payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        elif suffix in SERIALIZED_ARTIFACT_EXTENSIONS:
            payload = load_serialized_artifact(vocab_path)
        else:
            payload = _parse_text_vocab(vocab_path)

        if isinstance(payload, dict):
            return {
                str(token): int(index)
                for token, index in payload.items()
            }
        if isinstance(payload, list):
            vocab: dict[str, int] = {"<PAD>": 0}
            for index, token in enumerate(payload, start=1):
                token_str = str(token).strip()
                if token_str:
                    vocab[token_str] = index
            return vocab

        raise RuntimeError(f"Vocabulary artifact '{vocab_path.name}' is not a supported format.")

    def _load_model(self):
        checkpoint_path = self._registered_model.artifact_resolution.weights[0]
        checkpoint = _torch_load(self._torch, checkpoint_path, self._device)
        state_dict = checkpoint.get("model_state", checkpoint)
        self._architecture = _resolve_architecture_slug(
            self._manifest.framework.architecture,
            self._manifest.model,
            state_dict,
        )
        num_classes = max(
            len(self._artifact_labels) or 0,
            len(self._manifest_labels) or 0,
            _infer_num_classes(state_dict),
        )
        if num_classes <= 0:
            raise RuntimeError("Could not infer the number of output classes from the checkpoint.")

        embedding_weights = state_dict.get("embedding.weight")
        if embedding_weights is None:
            raise RuntimeError("PyTorch text models must include embedding.weight in the checkpoint.")

        embedding_shape = tuple(embedding_weights.shape)
        embedding_matrix = self._np.zeros(embedding_shape, dtype=self._np.float32)

        if self._architecture == "embedding-mlp":
            model = EmbeddingMLP(
                torch_module=self._torch,
                embedding_matrix=embedding_matrix,
                num_classes=num_classes,
                hidden_dim=int(self._manifest.model.get("hidden_dim", 256)),
                dropout=float(self._manifest.model.get("dropout", 0.3)),
            )
        elif self._architecture == "text-cnn":
            model = TextCNN(
                torch_module=self._torch,
                embedding_matrix=embedding_matrix,
                num_classes=num_classes,
                num_filters=int(
                    self._manifest.model.get(
                        "num_filters",
                        self._manifest.model.get("cnn_num_filters", 128),
                    )
                ),
                kernel_sizes=_parse_kernel_sizes(
                    self._manifest.model.get(
                        "kernel_sizes",
                        self._manifest.model.get("cnn_kernel_sizes", [2, 3, 4]),
                    )
                ),
                dropout=float(self._manifest.model.get("dropout", 0.3)),
            )
        elif self._architecture == "bilstm-attention":
            model = BiLSTMAttention(
                torch_module=self._torch,
                embedding_matrix=embedding_matrix,
                num_classes=num_classes,
                hidden_dim=int(self._manifest.model.get("hidden_dim", 128)),
                num_layers=int(self._manifest.model.get("num_layers", 2)),
                dropout=float(self._manifest.model.get("dropout", 0.3)),
            )
        else:
            raise RuntimeError(
                "Unsupported PyTorch architecture. Supported sequence-classification "
                "architectures are embedding-mlp, text-cnn, and bilstm-attention."
            )

        model.load_state_dict(state_dict, strict=True)
        model.to(self._device)
        model.eval()
        return model

    def _decode_label(self, value: Any) -> str:
        return decode_label_value(
            value,
            manifest_labels=self._manifest_labels,
            artifact_labels=self._artifact_labels,
            label_encoder=self._label_encoder,
        )


class TorchBiLSTMAttentionRunner(TorchSequenceClassifierRunner):
    """Backward-compatible alias for the historical BiLSTM runner."""


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


def _parse_text_vocab(path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {"<PAD>": 0}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" in stripped:
            token, raw_index = stripped.split("\t", 1)
            vocab[token] = int(raw_index)
            continue
        if " " in stripped:
            token, raw_index = stripped.rsplit(" ", 1)
            if raw_index.isdigit():
                vocab[token] = int(raw_index)
                continue
        vocab[stripped] = line_number
    return vocab


def _normalize_architecture_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-").replace("_", "-")


def _resolve_architecture_slug(
    framework_architecture: str | None,
    model_payload: dict[str, Any],
    state_dict: dict[str, Any],
) -> str:
    candidates = [
        str(framework_architecture or ""),
        str(model_payload.get("architecture") or ""),
        str(model_payload.get("family_slug") or ""),
        str(model_payload.get("model_family") or ""),
    ]
    for candidate in candidates:
        normalized = _normalize_architecture_name(candidate)
        if normalized in ARCHITECTURE_ALIASES:
            return ARCHITECTURE_ALIASES[normalized]

    if any(key.startswith("lstm.") for key in state_dict):
        return "bilstm-attention"
    if any(key.startswith("convs.") for key in state_dict):
        return "text-cnn"
    if any(key.startswith("fc.0.") for key in state_dict):
        return "embedding-mlp"

    return ""


def _infer_num_classes(state_dict: dict[str, Any]) -> int:
    for candidate in ("fc.weight", "fc.3.weight", "classifier.weight"):
        weights = state_dict.get(candidate)
        if weights is not None and hasattr(weights, "shape") and len(weights.shape) >= 1:
            return int(weights.shape[0])
    return 0


def _parse_kernel_sizes(value: Any) -> list[int]:
    if isinstance(value, list | tuple):
        return [int(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return [2, 3, 4]
        if stripped.startswith("["):
            return [int(item) for item in json.loads(stripped)]
        return [int(item.strip()) for item in stripped.split(",") if item.strip()]
    return [2, 3, 4]


def EmbeddingMLP(
    torch_module,
    embedding_matrix,
    num_classes: int,
    hidden_dim: int = 256,
    dropout: float = 0.3,
):
    class _EmbeddingMLP(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            _, embed_dim = embedding_matrix.shape
            self.embedding = torch_module.nn.Embedding.from_pretrained(
                torch_module.FloatTensor(embedding_matrix), freeze=False, padding_idx=0
            )
            self.fc = torch_module.nn.Sequential(
                torch_module.nn.Linear(embed_dim, hidden_dim),
                torch_module.nn.ReLU(),
                torch_module.nn.Dropout(dropout),
                torch_module.nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            mask = (x != 0).float().unsqueeze(-1)
            embeddings = self.embedding(x) * mask
            document = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.fc(document)

    return _EmbeddingMLP()


def TextCNN(
    torch_module,
    embedding_matrix,
    num_classes: int,
    num_filters: int = 128,
    kernel_sizes: list[int] | None = None,
    dropout: float = 0.3,
):
    class _TextCNN(torch_module.nn.Module):
        def __init__(self):
            super().__init__()
            _, embed_dim = embedding_matrix.shape
            effective_kernel_sizes = kernel_sizes or [2, 3, 4]
            self.embedding = torch_module.nn.Embedding.from_pretrained(
                torch_module.FloatTensor(embedding_matrix), freeze=False, padding_idx=0
            )
            self.convs = torch_module.nn.ModuleList(
                [
                    torch_module.nn.Conv1d(
                        embed_dim,
                        num_filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                    for kernel_size in effective_kernel_sizes
                ]
            )
            self.dropout = torch_module.nn.Dropout(dropout)
            self.fc = torch_module.nn.Linear(num_filters * len(effective_kernel_sizes), num_classes)

        def forward(self, x):
            embeddings = self.embedding(x).permute(0, 2, 1)
            pooled = [
                torch_module.relu(conv(embeddings)).max(dim=2).values
                for conv in self.convs
            ]
            return self.fc(self.dropout(torch_module.cat(pooled, dim=1)))

    return _TextCNN()


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

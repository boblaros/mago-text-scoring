from __future__ import annotations

import inspect
import threading
import time

from app.inference.base import BaseModelRunner, InferencePrediction, ProbabilityScore
from app.inference.preprocessing import normalize_text
from app.registry.contracts import RegisteredModel


class TransformersSequenceClassifierRunner(BaseModelRunner):
    def __init__(self, registered_model: RegisteredModel) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._registered_model = registered_model
        self._manifest = registered_model.manifest
        self._model_dir = registered_model.model_dir
        self._device = _resolve_device(torch, self._manifest.runtime.device)

        self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self._model_dir))
        self._model.to(self._device)
        self._model.eval()
        self._labels = self._manifest.labels_by_id()
        self._lock = threading.Lock()
        self._accepted_input_names = {
            name
            for name in inspect.signature(self._model.forward).parameters
            if name != "kwargs"
        }

    def predict(self, text: str) -> InferencePrediction:
        started = time.perf_counter()
        normalized = normalize_text(text)
        with self._lock:
            full_encoding = self._tokenizer(
                normalized,
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )
            encoded = self._tokenizer(
                normalized,
                truncation=self._manifest.runtime.truncation,
                padding="max_length" if self._manifest.runtime.padding else False,
                max_length=self._manifest.runtime.max_sequence_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self._device)
                for key, value in encoded.items()
                if key in self._accepted_input_names
            }
            with self._torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(0)

        probabilities = self._torch.softmax(logits, dim=-1).cpu().tolist()
        top_idx = int(self._torch.argmax(logits).item())
        label = self._labels[top_idx].effective_name if top_idx in self._labels else str(top_idx)
        ended = time.perf_counter()

        return InferencePrediction(
            predicted_label=label,
            confidence=float(probabilities[top_idx]),
            probabilities=[
                ProbabilityScore(
                    label=self._labels.get(index, None).effective_name
                    if self._labels.get(index, None)
                    else str(index),
                    score=float(score),
                )
                for index, score in enumerate(probabilities)
            ]
            if self._manifest.inference.return_probabilities
            else None,
            latency_ms=round((ended - started) * 1000, 2),
            sequence_length_used=int(encoded["input_ids"].shape[-1]),
            was_truncated=len(full_encoding["input_ids"])
            > self._manifest.runtime.max_sequence_length,
        )


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

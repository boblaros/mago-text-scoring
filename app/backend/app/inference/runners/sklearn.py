from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app.inference.base import BaseModelRunner, InferencePrediction, ProbabilityScore
from app.inference.preprocessing import apply_text_preprocessing
from app.inference.runtime_support import (
    SERIALIZED_ARTIFACT_EXTENSIONS,
    decode_label_value,
    load_label_records,
    load_optional_label_encoder,
    load_serialized_artifact,
)
from app.registry.contracts import RegisteredModel


class SklearnSequenceClassifierRunner(BaseModelRunner):
    def __init__(self, registered_model: RegisteredModel) -> None:
        self._registered_model = registered_model
        self._manifest = registered_model.manifest
        self._estimator = load_serialized_artifact(registered_model.artifact_resolution.weights[0])
        _hydrate_known_sklearn_defaults(self._estimator)
        self._embedded_pipeline = bool(
            getattr(self._estimator, "steps", None)
            or getattr(self._estimator, "named_steps", None)
        )
        self._auxiliary_transformers = self._load_auxiliary_transformers()
        self._manifest_labels = {
            label.id: label.effective_name
            for label in (self._manifest.labels or [])
        }
        self._artifact_labels = (
            load_label_records(registered_model.artifact_resolution.label_classes_file)
            or load_label_records(registered_model.artifact_resolution.label_map_file)
        )
        self._label_encoder = load_optional_label_encoder(
            registered_model.artifact_resolution.label_encoder_file
        )

    def predict(self, text: str) -> InferencePrediction:
        started = time.perf_counter()
        processed = apply_text_preprocessing(
            text,
            self._manifest.runtime.preprocessing,
            default_steps=("normalize_text",),
        )
        features: Any = [processed]
        if not self._embedded_pipeline:
            features = self._apply_auxiliary_transformers(features)

        probabilities, class_values = self._predict_proba(features)
        top_idx = int(np.argmax(probabilities))
        ended = time.perf_counter()

        return InferencePrediction(
            predicted_label=self._decode_label(class_values[top_idx]),
            confidence=float(probabilities[top_idx]),
            probabilities=[
                ProbabilityScore(
                    label=self._decode_label(class_value),
                    score=float(score),
                )
                for class_value, score in zip(class_values, probabilities, strict=False)
            ]
            if self._manifest.inference.return_probabilities
            else None,
            latency_ms=round((ended - started) * 1000, 2),
            sequence_length_used=len(processed.split()),
            was_truncated=False,
        )

    def _load_auxiliary_transformers(self) -> list[tuple[Path, Any]]:
        if self._embedded_pipeline:
            return []

        transformers: list[tuple[Path, Any]] = []
        for path in self._registered_model.artifact_resolution.config:
            if path.suffix.lower() not in SERIALIZED_ARTIFACT_EXTENSIONS:
                continue
            transformers.append((path, load_serialized_artifact(path)))
        return transformers

    def _apply_auxiliary_transformers(self, features: Any) -> Any:
        current = features
        for path, transformer in self._auxiliary_transformers:
            if hasattr(transformer, "transform"):
                current = transformer.transform(current)
                continue
            if callable(transformer):
                current = transformer(current)
                continue
            raise RuntimeError(
                f"Config artifact '{path.name}' does not expose a transform() method."
            )
        return current

    def _predict_proba(self, features: Any) -> tuple[np.ndarray, list[Any]]:
        classes = list(getattr(self._estimator, "classes_", []))

        if hasattr(self._estimator, "predict_proba"):
            probabilities = np.asarray(self._estimator.predict_proba(features), dtype=float)
            row = probabilities[0] if probabilities.ndim > 1 else probabilities
            if not classes:
                classes = list(range(len(row)))
            return np.asarray(row, dtype=float), classes

        if hasattr(self._estimator, "decision_function"):
            decision = np.asarray(self._estimator.decision_function(features), dtype=float)
            row = decision[0] if decision.ndim > 1 else decision
            scores = np.asarray(row, dtype=float).reshape(-1)
            if scores.size == 1:
                positive_score = 1.0 / (1.0 + np.exp(-float(scores[0])))
                probabilities = np.asarray([1.0 - positive_score, positive_score], dtype=float)
            else:
                probabilities = _softmax(scores)
            if not classes:
                classes = list(range(len(probabilities)))
            return probabilities, classes

        predicted = self._estimator.predict(features)
        predicted_value = predicted[0] if hasattr(predicted, "__getitem__") else predicted
        if not classes:
            classes = [predicted_value]
        if predicted_value not in classes:
            classes = [*classes, predicted_value]
        probabilities = np.zeros(len(classes), dtype=float)
        probabilities[classes.index(predicted_value)] = 1.0
        return probabilities, classes

    def _decode_label(self, value: Any) -> str:
        return decode_label_value(
            value,
            manifest_labels=self._manifest_labels,
            artifact_labels=self._artifact_labels,
            label_encoder=self._label_encoder,
        )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)
    if not np.isfinite(total) or total <= 0:
        return np.full_like(values, 1.0 / max(len(values), 1), dtype=float)
    return exp_values / total


SKLEARN_COMPATIBILITY_DEFAULTS: dict[str, dict[str, Any]] = {
    "LogisticRegression": {
        "multi_class": "auto",
        "l1_ratio": None,
        "n_jobs": None,
        "random_state": None,
        "class_weight": None,
        "warm_start": False,
        "verbose": 0,
    },
}


def _hydrate_known_sklearn_defaults(estimator: Any) -> None:
    for candidate in _walk_estimators(estimator):
        defaults = SKLEARN_COMPATIBILITY_DEFAULTS.get(candidate.__class__.__name__)
        if not defaults:
            continue
        for attribute, value in defaults.items():
            if hasattr(candidate, attribute):
                continue
            setattr(candidate, attribute, value)


def _walk_estimators(estimator: Any):
    yield estimator

    steps = getattr(estimator, "steps", None)
    if isinstance(steps, list | tuple):
        for _, step in steps:
            yield from _walk_estimators(step)
        return

    named_steps = getattr(estimator, "named_steps", None)
    if isinstance(named_steps, dict):
        for step in named_steps.values():
            yield from _walk_estimators(step)

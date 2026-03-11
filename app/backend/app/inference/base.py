from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class ProbabilityScore:
    label: str
    score: float


@dataclass(slots=True)
class InferencePrediction:
    predicted_label: str
    confidence: float
    probabilities: list[ProbabilityScore] | None
    latency_ms: float
    sequence_length_used: int | None = None
    was_truncated: bool | None = None


class BaseModelRunner(ABC):
    @abstractmethod
    def predict(self, text: str) -> InferencePrediction:
        raise NotImplementedError


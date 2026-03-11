from __future__ import annotations

from app.schemas.analysis import DomainResult
from app.services.aggregation import build_aggregate_result


def test_aggregate_summary_is_transparently_derived() -> None:
    aggregate = build_aggregate_result(
        [
            DomainResult(
                domain="sentiment",
                display_name="Sentiment",
                predicted_label="Joy",
                confidence=0.92,
                model_id="sentiment",
                model_name="Sentiment",
                latency_ms=12.2,
            ),
            DomainResult(
                domain="complexity",
                display_name="Complexity",
                predicted_label="B1-B2",
                confidence=0.81,
                model_id="complexity",
                model_name="Complexity",
                latency_ms=9.1,
            ),
            DomainResult(
                domain="age",
                display_name="Age",
                predicted_label="18-29",
                confidence=0.74,
                model_id="age",
                model_name="Age",
                latency_ms=10.0,
            ),
            DomainResult(
                domain="abuse",
                display_name="Abuse",
                predicted_label="Not Cyberbullying",
                confidence=0.95,
                model_id="abuse",
                model_name="Abuse",
                latency_ms=14.8,
            ),
        ]
    )

    assert aggregate.highest_confidence_domain == "abuse"
    assert aggregate.summary == (
        "The text reads as Joy, lands in B1-B2 complexity, aligns most with the "
        "18-29 age band, and maps to the Not Cyberbullying abuse class."
    )

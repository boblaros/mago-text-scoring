from __future__ import annotations

from uuid import uuid4

from app.registry.model_registry import ModelRegistry
from app.schemas.analysis import (
    AnalysisResponse,
    DomainResult,
    ProbabilityItem,
    RoutingOverview,
    TextProfile,
)
from app.services.aggregation import build_aggregate_result


class AnalysisService:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def analyze(self, text: str, domains: list[str] | None = None) -> AnalysisResponse:
        requested_domains = domains or self._registry.active_domains()
        selected_models = self._registry.get_active_models(requested_domains)
        results: list[DomainResult] = []

        for model in selected_models:
            runner = self._registry.get_runner(model)
            prediction = runner.predict(text)
            results.append(
                DomainResult(
                    domain=model.canonical_domain,
                    display_name=model.manifest.ui.domain_display_name
                    or model.canonical_domain.title(),
                    predicted_label=prediction.predicted_label,
                    confidence=prediction.confidence,
                    probabilities=[
                        ProbabilityItem(label=item.label, score=item.score)
                        for item in (prediction.probabilities or [])
                    ]
                    or None,
                    model_id=model.manifest.model_id,
                    model_name=model.manifest.display_name,
                    model_version=model.manifest.version,
                    latency_ms=prediction.latency_ms,
                    sequence_length_used=prediction.sequence_length_used,
                    was_truncated=prediction.was_truncated,
                )
            )

        return AnalysisResponse(
            request_id=str(uuid4()),
            text_profile=TextProfile(
                char_count=len(text),
                word_count=len(text.split()),
                line_count=max(1, text.count("\n") + 1),
            ),
            routing=RoutingOverview(
                requested_domains=requested_domains,
                resolved_domains=[result.domain for result in results],
            ),
            results=results,
            aggregate=build_aggregate_result(results),
        )

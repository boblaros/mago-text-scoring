from __future__ import annotations

from app.schemas.analysis import AggregateResult, DomainResult


def build_aggregate_result(results: list[DomainResult]) -> AggregateResult:
    if not results:
        return AggregateResult()

    confidences = [result.confidence for result in results]
    top = max(results, key=lambda result: result.confidence)
    summary = _build_summary(results)
    return AggregateResult(
        summary=summary,
        mean_confidence=round(sum(confidences) / len(confidences), 4),
        highest_confidence_domain=top.domain,
    )


def _build_summary(results: list[DomainResult]) -> str:
    lookup = {result.domain: result for result in results}
    fragments = []

    if sentiment := lookup.get("sentiment"):
        fragments.append(f"reads as {sentiment.predicted_label}")
    if complexity := lookup.get("complexity"):
        fragments.append(f"lands in {complexity.predicted_label} complexity")
    if age := lookup.get("age"):
        fragments.append(f"aligns most with the {age.predicted_label} age band")
    if abuse := lookup.get("abuse"):
        fragments.append(f"maps to the {abuse.predicted_label} abuse class")

    if not fragments:
        return ""

    if len(fragments) == 1:
        return f"The text {fragments[0]}."
    return "The text " + ", ".join(fragments[:-1]) + f", and {fragments[-1]}."


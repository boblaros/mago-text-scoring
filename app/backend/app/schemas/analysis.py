from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    domains: list[str] | None = None


class ProbabilityItem(BaseModel):
    label: str
    score: float


class DomainResult(BaseModel):
    domain: str
    display_name: str
    predicted_label: str
    confidence: float
    probabilities: list[ProbabilityItem] | None = None
    model_id: str
    model_name: str
    model_version: str | None = None
    latency_ms: float
    sequence_length_used: int | None = None
    was_truncated: bool | None = None


class AggregateResult(BaseModel):
    summary: str | None = None
    mean_confidence: float | None = None
    highest_confidence_domain: str | None = None


class TextProfile(BaseModel):
    char_count: int
    word_count: int
    line_count: int


class RoutingOverview(BaseModel):
    mode: Literal["broadcast-active-domains"] = "broadcast-active-domains"
    requested_domains: list[str]
    resolved_domains: list[str]


class AnalysisResponse(BaseModel):
    request_id: str
    text_profile: TextProfile
    routing: RoutingOverview
    results: list[DomainResult]
    aggregate: AggregateResult


class DomainCatalogModel(BaseModel):
    model_id: str
    display_name: str
    description: str | None = None
    version: str | None = None
    framework_type: str
    framework_task: str | None = None
    framework_library: str | None = None
    backbone: str | None = None
    architecture: str | None = None
    runtime_device: str | None = None
    runtime_max_sequence_length: int | None = None
    output_type: str | None = None
    is_active: bool
    priority: int
    notes: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)


class DomainCatalogEntry(BaseModel):
    domain: str
    display_name: str
    color_token: str
    group: str | None = None
    active_model_id: str
    active_model_name: str
    active_model_version: str | None = None
    model_count: int
    models: list[DomainCatalogModel]


class DomainCatalogResponse(BaseModel):
    domains: list[DomainCatalogEntry]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app_name: str
    version: str
    discovered_domains: list[str]

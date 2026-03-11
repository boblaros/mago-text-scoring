from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ModelHealthStatus = Literal["ready", "missing_artifacts", "incompatible"]
DashboardAvailability = Literal["missing", "partial", "available"]
DashboardSectionStatus = Literal["available", "missing", "image_only", "not_applicable"]


class DomainCatalogModel(BaseModel):
    model_id: str
    domain: str
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
    status: ModelHealthStatus
    status_reason: str | None = None
    can_activate: bool
    dashboard_status: DashboardAvailability = "missing"
    dashboard_sections_available: int = 0
    dashboard_sections_total: int = 0
    dashboard_generated_at: str | None = None


class DomainCatalogEntry(BaseModel):
    domain: str
    display_name: str
    color_token: str
    group: str | None = None
    active_model_id: str | None = None
    active_model_name: str | None = None
    active_model_version: str | None = None
    model_count: int
    models: list[DomainCatalogModel]


class DomainCatalogResponse(BaseModel):
    domains: list[DomainCatalogEntry]


class CatalogSnapshotResponse(BaseModel):
    active_domains: list[DomainCatalogEntry]
    management_domains: list[DomainCatalogEntry]


class DashboardSectionSummary(BaseModel):
    id: str
    title: str
    status: DashboardSectionStatus
    description: str | None = None
    reason: str | None = None
    files: list[str] = Field(default_factory=list)
    charts: list[str] = Field(default_factory=list)


class DashboardSourceItem(BaseModel):
    category: str
    path: str
    reason: str | None = None


class DashboardManifestSummary(BaseModel):
    schema_version: str
    generated_at: str | None = None
    dashboard_root: str
    model: dict[str, Any] = Field(default_factory=dict)
    entrypoints: dict[str, str] = Field(default_factory=dict)
    sections: list[DashboardSectionSummary] = Field(default_factory=list)
    selected_sources: list[DashboardSourceItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DashboardFigure(BaseModel):
    id: str
    path: str
    title: str | None = None
    section_id: str | None = None
    figure: dict[str, Any] = Field(default_factory=dict)


class DashboardImageAsset(BaseModel):
    title: str
    path: str
    url: str
    section_id: str | None = None


class ModelDashboardResponse(BaseModel):
    model_id: str
    available: bool
    manifest: DashboardManifestSummary | None = None
    overview: dict[str, Any] | None = None
    source_audit: dict[str, Any] | None = None
    documents: dict[str, Any] = Field(default_factory=dict)
    figures: list[DashboardFigure] = Field(default_factory=list)
    images: list[DashboardImageAsset] = Field(default_factory=list)


class ModelPatchRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    is_active: bool | None = None


class ModelReorderRequest(BaseModel):
    ordered_model_ids: list[str] = Field(min_length=1)


class UploadLabelClass(BaseModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=80)
    display_name: str | None = Field(default=None, max_length=80)

    @field_validator("name", "display_name", mode="before")
    @classmethod
    def _strip_names(cls, value: str | None):
        if value is None:
            return value
        stripped = value.strip()
        return stripped or None


class UploadModelMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(min_length=2, max_length=120)
    domain: str = Field(min_length=2, max_length=80)
    display_name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    version: str | None = Field(default=None, max_length=80)
    enable_on_upload: bool = False
    framework_type: Literal["transformers", "pytorch", "sklearn"]
    framework_task: str = "sequence-classification"
    framework_library: str | None = Field(default=None, max_length=80)
    backbone: str | None = Field(default=None, max_length=120)
    architecture: str | None = Field(default=None, max_length=120)
    base_model: str | None = Field(default=None, max_length=120)
    embeddings: str | None = Field(default=None, max_length=120)
    output_type: str | None = Field(default="single-label-classification", max_length=120)
    runtime_device: str = Field(default="auto", max_length=16)
    runtime_max_sequence_length: int = Field(default=256, ge=1, le=4096)
    runtime_batch_size: int = Field(default=1, ge=1, le=512)
    runtime_truncation: bool = True
    runtime_padding: bool | str = True
    ui_display_name: str | None = Field(default=None, max_length=120)
    color_token: str | None = Field(default=None, max_length=80)
    group: str | None = Field(default=None, max_length=80)
    labels: list[UploadLabelClass] = Field(min_length=1)
    model_payload: dict[str, Any] = Field(default_factory=dict, alias="model_config")

    @field_validator(
        "model_id",
        "domain",
        "display_name",
        "description",
        "version",
        "framework_task",
        "framework_library",
        "backbone",
        "architecture",
        "base_model",
        "embeddings",
        "output_type",
        "runtime_device",
        "ui_display_name",
        "color_token",
        "group",
        mode="before",
    )
    @classmethod
    def _strip_strings(cls, value: str | None):
        if value is None:
            return value
        stripped = value.strip()
        return stripped or None

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, value: list[UploadLabelClass]):
        ids = {label.id for label in value}
        names = {label.name for label in value}
        if len(ids) != len(value):
            raise ValueError("Label ids must be unique.")
        if len(names) != len(value):
            raise ValueError("Label names must be unique.")
        return value

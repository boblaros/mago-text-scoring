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
    framework_problem_type: str | None = Field(default=None, max_length=120)
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
    runtime_preprocessing: str | None = Field(default=None, max_length=240)
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
        "framework_problem_type",
        "backbone",
        "architecture",
        "base_model",
        "embeddings",
        "output_type",
        "runtime_device",
        "runtime_preprocessing",
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


class UploadFileDescriptor(BaseModel):
    name: str = Field(min_length=1, max_length=240)
    size_bytes: int | None = Field(default=None, ge=0)
    relative_path: str | None = Field(default=None, max_length=500)


class LocalUploadPreflightRequest(BaseModel):
    registration_mode: Literal["uploaded", "generated"]
    metadata: UploadModelMetadata
    artifact_manifest: dict[str, list[UploadFileDescriptor]] = Field(default_factory=dict)
    dashboard_manifest: list[UploadFileDescriptor] = Field(default_factory=list)


class ArtifactValidationSummary(BaseModel):
    slot: str
    title: str
    required: bool
    valid: bool
    message: str | None = None
    files: list[str] = Field(default_factory=list)


class LocalUploadPreflightResponse(BaseModel):
    ready: bool
    config_source: Literal["uploaded", "generated"]
    normalized_metadata: UploadModelMetadata
    config_preview: str
    artifact_checks: list[ArtifactValidationSummary] = Field(default_factory=list)
    dashboard_attached: bool = False
    warnings: list[str] = Field(default_factory=list)


class HuggingFacePreflightRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=240)
    metadata: UploadModelMetadata


class HuggingFaceArtifactCheck(BaseModel):
    path: str
    category: str
    required: bool
    available: bool
    size_bytes: int | None = Field(default=None, ge=0)
    message: str | None = None


class HuggingFacePreflightResponse(BaseModel):
    normalized_repo_id: str
    repo_url: str
    detected_framework_type: str | None = None
    detected_task: str | None = None
    framework_library: str | None = None
    architecture: str | None = None
    backbone: str | None = None
    base_model: str | None = None
    estimated_download_size_bytes: int | None = Field(default=None, ge=0)
    disk_free_bytes: int = Field(ge=0)
    memory_total_bytes: int | None = Field(default=None, ge=0)
    memory_estimate_bytes: int | None = Field(default=None, ge=0)
    runtime_supported: bool
    compatible: bool
    ready_to_import: bool
    required_files: list[HuggingFaceArtifactCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    normalized_metadata: UploadModelMetadata
    config_preview: str


class ModelRegistrationResult(BaseModel):
    model_id: str
    source: Literal["local", "huggingface"]
    branch: Literal["local-config-upload", "local-generated-config", "huggingface"]
    config_source: Literal["uploaded", "generated"]
    framework_type: str
    display_name: str
    domain: str
    is_active: bool
    status: ModelHealthStatus
    status_reason: str | None = None
    dashboard_status: DashboardAvailability = "missing"
    warnings: list[str] = Field(default_factory=list)


class ModelRegistrationResponse(BaseModel):
    snapshot: CatalogSnapshotResponse
    result: ModelRegistrationResult

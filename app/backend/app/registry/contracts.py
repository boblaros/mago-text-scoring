from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class LabelClass(BaseModel):
    id: int
    name: str
    display_name: str | None = None

    @property
    def effective_name(self) -> str:
        return self.display_name or self.name


class ModelFramework(BaseModel):
    type: str
    task: str | None = None
    library: str | None = None
    backbone: str | None = None
    architecture: str | None = None
    base_model: str | None = None
    problem_type: str | None = None
    embeddings: str | None = None


class ModelArtifacts(BaseModel):
    base_dir: str | None = None
    weights: list[str] = Field(default_factory=list)
    tokenizer: list[str] = Field(default_factory=list)
    config: list[str] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    label_map_file: str | None = None
    label_classes_file: str | None = None
    label_encoder_file: str | None = None


class ModelRuntime(BaseModel):
    max_sequence_length: int = 256
    truncation: bool = True
    padding: bool | str = True
    batch_size: int = 1
    device: str = "auto"
    preprocessing: str | None = None


class ModelInference(BaseModel):
    return_probabilities: bool = True
    confidence_method: str = "softmax_max"


class ModelUI(BaseModel):
    domain_display_name: str | None = None
    color_token: str | None = None
    group: str | None = None


class ModelManifest(BaseModel):
    model_id: str
    domain: str
    display_name: str
    description: str | None = None
    version: str | None = None
    is_active: bool = False
    priority: int = 0
    framework: ModelFramework
    artifacts: ModelArtifacts = Field(default_factory=ModelArtifacts)
    model: dict[str, Any] = Field(default_factory=dict)
    runtime: ModelRuntime = Field(default_factory=ModelRuntime)
    dashboard: dict[str, Any] = Field(default_factory=dict)
    label_type: str | None = None
    labels: list[LabelClass] | None = None
    inference: ModelInference = Field(default_factory=ModelInference)
    ui: ModelUI = Field(default_factory=ModelUI)

    @classmethod
    def from_yaml_dict(cls, raw: dict[str, Any]) -> "ModelManifest":
        label_block = raw.get("labels", {})
        raw = raw.copy()
        raw["label_type"] = label_block.get("type")
        raw["labels"] = label_block.get("classes")
        return cls.model_validate(raw)

    def labels_by_id(self) -> dict[int, LabelClass]:
        return {label.id: label for label in self.labels or []}

    def to_yaml_dict(self) -> dict[str, Any]:
        payload = self.model_dump(exclude_none=True)
        if not payload.get("dashboard"):
            payload.pop("dashboard", None)
        labels = payload.pop("labels", [])
        label_type = payload.pop("label_type", None)
        if label_type or labels:
            payload["labels"] = {
                "type": label_type,
                "classes": labels,
            }
        return payload


class ResolvedArtifacts(BaseModel):
    weights: list[Path] = Field(default_factory=list)
    tokenizer: list[Path] = Field(default_factory=list)
    config: list[Path] = Field(default_factory=list)
    vocabulary: list[Path] = Field(default_factory=list)
    label_map_file: Path | None = None
    label_classes_file: Path | None = None
    label_encoder_file: Path | None = None
    notes: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class RegisteredModel(BaseModel):
    manifest: ModelManifest
    config_path: Path
    model_dir: Path
    canonical_domain: str
    artifact_resolution: ResolvedArtifacts

    @property
    def active(self) -> bool:
        return self.manifest.is_active

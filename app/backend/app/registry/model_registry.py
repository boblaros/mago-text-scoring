from __future__ import annotations

import json
import re
import shutil
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from app.inference.factory import InferencePluginRegistry
from app.registry.artifacts import resolve_artifacts
from app.registry.contracts import ModelManifest, RegisteredModel
from app.registry.dashboard_loader import load_model_dashboard, summarize_dashboard
from app.schemas.models import (
    ArtifactValidationSummary,
    HuggingFaceArtifactCheck,
    HuggingFacePreflightRequest,
    HuggingFacePreflightResponse,
    LocalUploadPreflightRequest,
    LocalUploadPreflightResponse,
    ModelRegistrationResult,
    UploadFileDescriptor,
    UploadLabelClass,
    UploadModelMetadata,
)
from app.services.huggingface_import import (
    HuggingFaceImportService,
    HuggingFaceInspection,
)


@dataclass(slots=True)
class UploadedPayload:
    path: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ArtifactRequirement:
    slot: str
    required: bool
    min_files: int
    max_files: int | None
    allowed_extensions: tuple[str, ...]
    description: str

    @property
    def title(self) -> str:
        return self.slot.replace("_", " ").title()


class RegistryValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        field_errors: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.field_errors = field_errors or {}


@dataclass(frozen=True, slots=True)
class RegistrationOutcome:
    snapshot: dict[str, list[dict]]
    result: ModelRegistrationResult


ARTIFACT_REQUIREMENTS: dict[str, tuple[ArtifactRequirement, ...]] = {
    "transformers": (
        ArtifactRequirement(
            slot="weights",
            required=True,
            min_files=1,
            max_files=2,
            allowed_extensions=(".safetensors", ".bin", ".pt", ".pth"),
            description="Model weights exported for transformers runtime.",
        ),
        ArtifactRequirement(
            slot="tokenizer",
            required=True,
            min_files=1,
            max_files=None,
            allowed_extensions=(".json", ".txt", ".model", ".vocab"),
            description="Tokenizer assets such as tokenizer.json and tokenizer_config.json.",
        ),
        ArtifactRequirement(
            slot="config",
            required=True,
            min_files=1,
            max_files=None,
            allowed_extensions=(".json", ".yaml", ".yml", ".bin"),
            description="Runtime and model config files.",
        ),
        ArtifactRequirement(
            slot="label_map_file",
            required=False,
            min_files=0,
            max_files=1,
            allowed_extensions=(".json", ".pkl"),
            description="Optional label encoder or label map artifact.",
        ),
        ArtifactRequirement(
            slot="label_classes_file",
            required=False,
            min_files=0,
            max_files=1,
            allowed_extensions=(".json", ".pkl"),
            description="Optional exported label classes.",
        ),
    ),
    "pytorch": (
        ArtifactRequirement(
            slot="weights",
            required=True,
            min_files=1,
            max_files=2,
            allowed_extensions=(".pt", ".pth", ".bin"),
            description="Checkpoint or weight file for the PyTorch model.",
        ),
        ArtifactRequirement(
            slot="vocabulary",
            required=True,
            min_files=1,
            max_files=None,
            allowed_extensions=(".pkl", ".json", ".txt"),
            description="Vocabulary or preprocessing lookup assets.",
        ),
        ArtifactRequirement(
            slot="config",
            required=True,
            min_files=1,
            max_files=None,
            allowed_extensions=(".json", ".yaml", ".yml", ".bin"),
            description="Model and runtime config files.",
        ),
        ArtifactRequirement(
            slot="label_classes_file",
            required=False,
            min_files=0,
            max_files=1,
            allowed_extensions=(".json", ".pkl"),
            description="Optional label classes artifact.",
        ),
        ArtifactRequirement(
            slot="label_encoder_file",
            required=False,
            min_files=0,
            max_files=1,
            allowed_extensions=(".json", ".pkl"),
            description="Optional label encoder artifact.",
        ),
    ),
    "sklearn": (
        ArtifactRequirement(
            slot="weights",
            required=True,
            min_files=1,
            max_files=2,
            allowed_extensions=(".pkl", ".joblib", ".bin"),
            description="Serialized classical model artifact.",
        ),
        ArtifactRequirement(
            slot="config",
            required=True,
            min_files=1,
            max_files=None,
            allowed_extensions=(".json", ".yaml", ".yml", ".txt"),
            description="Feature extraction or runtime config.",
        ),
        ArtifactRequirement(
            slot="label_classes_file",
            required=False,
            min_files=0,
            max_files=1,
            allowed_extensions=(".json", ".pkl"),
            description="Optional label classes artifact.",
        ),
        ArtifactRequirement(
            slot="label_encoder_file",
            required=False,
            min_files=0,
            max_files=1,
            allowed_extensions=(".json", ".pkl"),
            description="Optional label encoder artifact.",
        ),
    ),
}


class ModelRegistry:
    def __init__(
        self,
        settings,
        plugin_registry: InferencePluginRegistry,
        *,
        hf_import_service: HuggingFaceImportService | None = None,
    ) -> None:
        self._settings = settings
        self._plugin_registry = plugin_registry
        self._hf_import_service = hf_import_service or HuggingFaceImportService(settings)
        self._models_by_domain: dict[str, list[RegisteredModel]] = defaultdict(list)
        self._runner_cache: dict[str, object] = {}
        self._lock = threading.Lock()

    def discover(self) -> None:
        self._models_by_domain.clear()
        self._runner_cache.clear()
        for config_path in sorted(self._settings.model_discovery_root.rglob("model-config.yaml")):
            manifest = _load_manifest(config_path)
            model_dir = config_path.parent
            canonical_domain = self._canonicalize_domain(manifest.domain)
            artifacts = resolve_artifacts(model_dir, manifest)
            registered_model = RegisteredModel(
                manifest=manifest,
                config_path=config_path,
                model_dir=model_dir,
                canonical_domain=canonical_domain,
                artifact_resolution=artifacts,
            )
            self._models_by_domain[canonical_domain].append(registered_model)

        for models in self._models_by_domain.values():
            models.sort(key=self._model_sort_key)

    def domains(self) -> list[str]:
        return self._ordered_domains(active_only=False)

    def active_domains(self) -> list[str]:
        return self._ordered_domains(active_only=True)

    def get_models(self, domain: str | None = None) -> list[RegisteredModel]:
        if domain is None:
            return [
                model
                for domain_name in self._ordered_domains(active_only=False)
                for model in self._models_by_domain.get(domain_name, [])
            ]
        return list(self._models_by_domain.get(domain, []))

    def get_active_models(self, domains: Iterable[str] | None = None) -> list[RegisteredModel]:
        selected_domains = list(domains) if domains is not None else self.active_domains()
        active_models: list[RegisteredModel] = []
        for domain in selected_domains:
            active_model = self._select_active_model(self._models_by_domain.get(domain, []))
            if active_model is not None:
                active_models.append(active_model)
        return active_models

    def get_model(self, model_id: str) -> RegisteredModel:
        for model in self.get_models():
            if model.manifest.model_id == model_id:
                return model
        raise KeyError(model_id)

    def get_runner(self, model: RegisteredModel):
        with self._lock:
            runner = self._runner_cache.get(model.manifest.model_id)
            if runner is None:
                runner = self._plugin_registry.create(model)
                self._runner_cache[model.manifest.model_id] = runner
            return runner

    def preload_active_models(self) -> None:
        for model in self.get_active_models():
            self.get_runner(model)

    def catalog(self, active_only: bool = False) -> list[dict]:
        entries = []
        for domain in self._ordered_domains(active_only=active_only):
            models = self._models_by_domain.get(domain, [])
            if not models:
                continue

            active_model = self._select_active_model(models)
            if active_only and active_model is None:
                continue

            entry_models = [active_model] if active_only and active_model is not None else models
            reference_model = active_model or entry_models[0]
            entries.append(
                {
                    "domain": domain,
                    "display_name": reference_model.manifest.ui.domain_display_name
                    or domain.title(),
                    "color_token": reference_model.manifest.ui.color_token or domain,
                    "group": reference_model.manifest.ui.group,
                    "active_model_id": active_model.manifest.model_id if active_model else None,
                    "active_model_name": active_model.manifest.display_name if active_model else None,
                    "active_model_version": active_model.manifest.version if active_model else None,
                    "model_count": len(entry_models),
                    "models": [self._serialize_model(model) for model in entry_models],
                }
            )
        return entries

    def snapshot(self) -> dict[str, list[dict]]:
        return {
            "active_domains": self.catalog(active_only=True),
            "management_domains": self.catalog(active_only=False),
        }

    def update_model(
        self,
        model_id: str,
        *,
        display_name: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, list[dict]]:
        model = self.get_model(model_id)
        changed = False

        if display_name is not None and display_name != model.manifest.display_name:
            model.manifest.display_name = display_name
            changed = True

        if is_active is not None and is_active != model.manifest.is_active:
            if is_active:
                _, can_activate, reason = self._model_runtime_status(model)
                if not can_activate:
                    raise ValueError(reason or "The model cannot be activated.")
                self._deactivate_domain_models(model.canonical_domain, except_model_id=model_id)
            model.manifest.is_active = is_active
            changed = True

        if changed:
            self._write_manifest(model.config_path, model.manifest)
            self.discover()
        return self.snapshot()

    def reorder_models(self, ordered_model_ids: list[str]) -> dict[str, list[dict]]:
        current_models = self.get_models()
        by_id = {model.manifest.model_id: model for model in current_models}
        if not by_id:
            return self.snapshot()

        normalized_ids: list[str] = []
        for model_id in ordered_model_ids:
            if model_id in by_id and model_id not in normalized_ids:
                normalized_ids.append(model_id)

        for model in current_models:
            if model.manifest.model_id not in normalized_ids:
                normalized_ids.append(model.manifest.model_id)

        next_priority = len(normalized_ids)
        for model_id in normalized_ids:
            model = by_id[model_id]
            model.manifest.priority = next_priority
            self._write_manifest(model.config_path, model.manifest)
            next_priority -= 1

        self.discover()
        return self.snapshot()

    def delete_model(self, model_id: str) -> dict[str, list[dict]]:
        model = self.get_model(model_id)
        shutil.rmtree(model.model_dir)
        self.discover()
        return self.snapshot()

    def preflight_local_upload(
        self,
        payload: LocalUploadPreflightRequest,
        *,
        registration_config_uploads: list[UploadedPayload],
    ) -> LocalUploadPreflightResponse:
        metadata = self._normalize_metadata(payload.metadata)
        self._ensure_model_id_available(metadata.model_id)

        warnings: list[str] = []
        if payload.registration_mode == "uploaded":
            registration_manifest, registration_warnings = self._load_uploaded_registration_manifest(
                metadata,
                registration_config_uploads,
            )
            metadata = self._merge_metadata_with_registration_manifest(
                metadata,
                registration_manifest,
            )
            warnings.extend(registration_warnings)

        artifact_checks, field_errors = self._artifact_selection_checks(
            metadata.framework_type,
            payload.artifact_manifest,
        )
        warnings.extend(self._dashboard_selection_warnings(payload.dashboard_manifest))
        if field_errors:
            raise RegistryValidationError(
                "Resolve the highlighted file issues before continuing.",
                field_errors=field_errors,
            )

        planned_artifacts = _planned_artifact_manifest(
            metadata.framework_type,
            {
                slot: [descriptor.name for descriptor in descriptors]
                for slot, descriptors in payload.artifact_manifest.items()
            },
        )
        manifest = self._build_manifest(
            metadata=metadata,
            canonical_domain=self._canonicalize_domain(metadata.domain),
            artifact_manifest=planned_artifacts,
        )
        return LocalUploadPreflightResponse(
            ready=True,
            config_source=payload.registration_mode,
            normalized_metadata=metadata,
            config_preview=self._build_manifest_preview(manifest),
            artifact_checks=artifact_checks,
            dashboard_attached=bool(payload.dashboard_manifest),
            warnings=list(dict.fromkeys(warnings)),
        )

    def register_local_upload(
        self,
        payload: LocalUploadPreflightRequest,
        *,
        artifact_uploads: list[UploadedPayload],
        dashboard_uploads: list[UploadedPayload],
        registration_config_uploads: list[UploadedPayload],
    ) -> RegistrationOutcome:
        metadata = self._normalize_metadata(payload.metadata)
        self._ensure_model_id_available(metadata.model_id)

        warnings: list[str] = []
        if payload.registration_mode == "uploaded":
            registration_manifest, registration_warnings = self._load_uploaded_registration_manifest(
                metadata,
                registration_config_uploads,
            )
            metadata = self._merge_metadata_with_registration_manifest(
                metadata,
                registration_manifest,
            )
            warnings.extend(registration_warnings)

        canonical_domain = self._canonicalize_domain(metadata.domain)
        model_dir = self._allocate_model_dir(canonical_domain, metadata.model_id)
        model_dir.mkdir(parents=True, exist_ok=False)

        try:
            artifact_manifest = _save_artifacts(
                model_dir=model_dir,
                framework_type=metadata.framework_type,
                artifact_uploads=artifact_uploads,
            )
            manifest = self._build_manifest(
                metadata=metadata,
                canonical_domain=canonical_domain,
                artifact_manifest=artifact_manifest,
            )

            self._write_manifest(model_dir / "model-config.yaml", manifest)
            if dashboard_uploads:
                _save_dashboard(model_dir, dashboard_uploads)

            registered = RegisteredModel(
                manifest=manifest,
                config_path=model_dir / "model-config.yaml",
                model_dir=model_dir,
                canonical_domain=canonical_domain,
                artifact_resolution=resolve_artifacts(model_dir, manifest),
            )
            _, can_activate, reason = self._model_runtime_status(registered)
            if metadata.enable_on_upload and not can_activate:
                raise RegistryValidationError(
                    reason or "The uploaded model cannot be activated.",
                    field_errors={"metadata.enable_on_upload": reason or "Activation is blocked."},
                )

            if metadata.enable_on_upload:
                self._deactivate_domain_models(canonical_domain)
                manifest.is_active = True
                self._write_manifest(model_dir / "model-config.yaml", manifest)

        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        self.discover()
        return RegistrationOutcome(
            snapshot=self.snapshot(),
            result=self._build_registration_result(
                metadata.model_id,
                source="local",
                branch=(
                    "local-config-upload"
                    if payload.registration_mode == "uploaded"
                    else "local-generated-config"
                ),
                config_source=payload.registration_mode,
                warnings=list(dict.fromkeys(warnings)),
            ),
        )

    def preflight_huggingface_import(
        self,
        payload: HuggingFacePreflightRequest,
    ) -> HuggingFacePreflightResponse:
        self._ensure_model_id_available(payload.metadata.model_id)
        try:
            inspection = self._hf_import_service.inspect(payload.repo)
        except ValueError as exc:
            raise RegistryValidationError(
                str(exc),
                field_errors={"huggingface.repo": str(exc)},
            ) from exc

        metadata = self._metadata_from_hf_request(payload.metadata, inspection)
        planned_artifacts = _planned_artifact_manifest(
            metadata.framework_type,
            {
                slot: [file.path for file in files]
                for slot, files in inspection.download_plan.items()
            },
        )
        manifest = self._build_manifest(
            metadata=metadata,
            canonical_domain=self._canonicalize_domain(metadata.domain),
            artifact_manifest=planned_artifacts,
        )

        return HuggingFacePreflightResponse(
            normalized_repo_id=inspection.repo_id,
            repo_url=inspection.repo_url,
            detected_framework_type=inspection.detected_framework_type,
            detected_task=inspection.detected_task,
            framework_library=inspection.framework_library,
            architecture=inspection.architecture,
            backbone=inspection.backbone,
            base_model=inspection.base_model,
            estimated_download_size_bytes=inspection.estimated_download_size_bytes,
            disk_free_bytes=inspection.disk_free_bytes,
            memory_total_bytes=inspection.memory_total_bytes,
            memory_estimate_bytes=inspection.memory_estimate_bytes,
            runtime_supported=inspection.runtime_supported,
            compatible=inspection.compatible,
            ready_to_import=inspection.ready_to_import,
            required_files=[
                HuggingFaceArtifactCheck(
                    path=file.path,
                    category=file.category,
                    required=file.required,
                    available=file.message is None,
                    size_bytes=file.size_bytes,
                    message=file.message,
                )
                for file in inspection.required_files
            ],
            warnings=inspection.warnings,
            blocking_reasons=inspection.blocking_reasons,
            normalized_metadata=metadata,
            config_preview=self._build_manifest_preview(manifest),
        )

    def import_huggingface_model(
        self,
        payload: HuggingFacePreflightRequest,
    ) -> RegistrationOutcome:
        self._ensure_model_id_available(payload.metadata.model_id)
        try:
            inspection = self._hf_import_service.inspect(payload.repo)
        except ValueError as exc:
            raise RegistryValidationError(
                str(exc),
                field_errors={"huggingface.repo": str(exc)},
            ) from exc

        if not inspection.ready_to_import:
            message = inspection.blocking_reasons[0] if inspection.blocking_reasons else (
                "This Hugging Face repo cannot be imported."
            )
            raise RegistryValidationError(
                message,
                field_errors={"huggingface.repo": message},
            )

        metadata = self._metadata_from_hf_request(payload.metadata, inspection)
        canonical_domain = self._canonicalize_domain(metadata.domain)
        model_dir = self._allocate_model_dir(canonical_domain, metadata.model_id)
        model_dir.mkdir(parents=True, exist_ok=False)

        try:
            artifact_manifest = self._hf_import_service.download_to_directory(inspection, model_dir)
            manifest = self._build_manifest(
                metadata=metadata,
                canonical_domain=canonical_domain,
                artifact_manifest=artifact_manifest,
            )
            self._write_manifest(model_dir / "model-config.yaml", manifest)

            registered = RegisteredModel(
                manifest=manifest,
                config_path=model_dir / "model-config.yaml",
                model_dir=model_dir,
                canonical_domain=canonical_domain,
                artifact_resolution=resolve_artifacts(model_dir, manifest),
            )
            _, can_activate, reason = self._model_runtime_status(registered)
            if metadata.enable_on_upload and not can_activate:
                raise RegistryValidationError(
                    reason or "The imported model cannot be activated.",
                    field_errors={"metadata.enable_on_upload": reason or "Activation is blocked."},
                )

            if metadata.enable_on_upload:
                self._deactivate_domain_models(canonical_domain)
                manifest.is_active = True
                self._write_manifest(model_dir / "model-config.yaml", manifest)

        except Exception:
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

        self.discover()
        return RegistrationOutcome(
            snapshot=self.snapshot(),
            result=self._build_registration_result(
                metadata.model_id,
                source="huggingface",
                branch="huggingface",
                config_source="generated",
                warnings=inspection.warnings,
            ),
        )

    def load_dashboard(self, model_id: str, asset_url_builder) -> object:
        model = self.get_model(model_id)
        return load_model_dashboard(model, asset_url_builder)

    def dashboard_asset_path(self, model_id: str, asset_path: str) -> Path:
        model = self.get_model(model_id)
        resolved = (model.model_dir / "dashboard" / asset_path).resolve()
        dashboard_root = (model.model_dir / "dashboard").resolve()
        if dashboard_root not in resolved.parents and resolved != dashboard_root:
            raise ValueError("Dashboard asset path is outside the model dashboard directory.")
        if not resolved.exists():
            raise FileNotFoundError(asset_path)
        return resolved

    def _build_manifest(
        self,
        metadata: UploadModelMetadata,
        canonical_domain: str,
        artifact_manifest: dict[str, object],
    ) -> ModelManifest:
        payload = {
            "model_id": metadata.model_id,
            "domain": canonical_domain,
            "display_name": metadata.display_name,
            "description": metadata.description,
            "version": metadata.version,
            "is_active": False,
            "priority": self._next_priority(),
            "framework": {
                "type": metadata.framework_type,
                "task": metadata.framework_task,
                "library": metadata.framework_library,
                "problem_type": metadata.framework_problem_type,
                "backbone": metadata.backbone,
                "architecture": metadata.architecture,
                "base_model": metadata.base_model,
                "embeddings": metadata.embeddings,
            },
            "artifacts": artifact_manifest,
            "model": metadata.model_payload,
            "runtime": {
                "max_sequence_length": metadata.runtime_max_sequence_length,
                "truncation": metadata.runtime_truncation,
                "padding": metadata.runtime_padding,
                "batch_size": metadata.runtime_batch_size,
                "device": metadata.runtime_device,
                "preprocessing": metadata.runtime_preprocessing,
            },
            "labels": {
                "type": metadata.output_type,
                "classes": [label.model_dump(exclude_none=True) for label in metadata.labels],
            },
            "inference": {
                "return_probabilities": True,
                "confidence_method": "softmax_max",
            },
            "ui": {
                "domain_display_name": metadata.ui_display_name or canonical_domain.title(),
                "color_token": metadata.color_token or canonical_domain,
                "group": metadata.group or f"{canonical_domain}-custom",
            },
        }
        return ModelManifest.from_yaml_dict(payload)

    def _build_manifest_preview(self, manifest: ModelManifest) -> str:
        return yaml.safe_dump(
            manifest.to_yaml_dict(),
            sort_keys=False,
            allow_unicode=False,
        )

    def _build_registration_result(
        self,
        model_id: str,
        *,
        source: str,
        branch: str,
        config_source: str,
        warnings: list[str],
    ) -> ModelRegistrationResult:
        model = self.get_model(model_id)
        serialized = self._serialize_model(model)
        return ModelRegistrationResult(
            model_id=model_id,
            source=source,
            branch=branch,
            config_source=config_source,
            framework_type=str(serialized["framework_type"]),
            display_name=str(serialized["display_name"]),
            domain=str(serialized["domain"]),
            is_active=bool(serialized["is_active"]),
            status=str(serialized["status"]),
            status_reason=serialized.get("status_reason"),
            dashboard_status=str(serialized["dashboard_status"]),
            warnings=warnings,
        )

    def _metadata_from_hf_request(
        self,
        metadata: UploadModelMetadata,
        inspection: HuggingFaceInspection,
    ) -> UploadModelMetadata:
        labels = metadata.labels
        if inspection.labels and _labels_are_placeholder(labels):
            labels = [UploadLabelClass.model_validate(label) for label in inspection.labels]

        return self._normalize_metadata(
            metadata.model_copy(
                update={
                    "framework_type": inspection.detected_framework_type
                    or metadata.framework_type,
                    "framework_task": inspection.detected_task or metadata.framework_task,
                    "framework_library": inspection.framework_library
                    or metadata.framework_library,
                    "framework_problem_type": metadata.framework_problem_type
                    or "single_label_classification",
                    "backbone": metadata.backbone or inspection.backbone,
                    "architecture": metadata.architecture or inspection.architecture,
                    "base_model": metadata.base_model or inspection.base_model,
                    "labels": labels,
                    "model_payload": metadata.model_payload or inspection.model_payload,
                }
            )
        )

    def _merge_metadata_with_registration_manifest(
        self,
        metadata: UploadModelMetadata,
        manifest: ModelManifest,
    ) -> UploadModelMetadata:
        labels = metadata.labels
        if _labels_are_placeholder(labels) and manifest.labels:
            labels = [
                UploadLabelClass(
                    id=label.id,
                    name=label.name,
                    display_name=label.display_name,
                )
                for label in manifest.labels
            ]

        return self._normalize_metadata(
            metadata.model_copy(
                update={
                    "description": metadata.description or manifest.description,
                    "version": metadata.version or manifest.version,
                    "framework_library": metadata.framework_library
                    or manifest.framework.library,
                    "framework_problem_type": metadata.framework_problem_type
                    or manifest.framework.problem_type,
                    "backbone": metadata.backbone or manifest.framework.backbone,
                    "architecture": metadata.architecture or manifest.framework.architecture,
                    "base_model": metadata.base_model or manifest.framework.base_model,
                    "embeddings": metadata.embeddings or manifest.framework.embeddings,
                    "output_type": metadata.output_type or manifest.label_type,
                    "runtime_device": _prefer_manifest_value(
                        metadata.runtime_device,
                        "auto",
                        manifest.runtime.device,
                    ),
                    "runtime_max_sequence_length": _prefer_manifest_value(
                        metadata.runtime_max_sequence_length,
                        256,
                        manifest.runtime.max_sequence_length,
                    ),
                    "runtime_batch_size": _prefer_manifest_value(
                        metadata.runtime_batch_size,
                        1,
                        manifest.runtime.batch_size,
                    ),
                    "runtime_truncation": _prefer_manifest_value(
                        metadata.runtime_truncation,
                        True,
                        manifest.runtime.truncation,
                    ),
                    "runtime_padding": _prefer_manifest_value(
                        metadata.runtime_padding,
                        True,
                        manifest.runtime.padding,
                    ),
                    "runtime_preprocessing": metadata.runtime_preprocessing
                    or manifest.runtime.preprocessing,
                    "ui_display_name": metadata.ui_display_name
                    or manifest.ui.domain_display_name,
                    "color_token": metadata.color_token or manifest.ui.color_token,
                    "group": metadata.group or manifest.ui.group,
                    "labels": labels,
                    "model_payload": metadata.model_payload or manifest.model,
                }
            )
        )

    def _load_uploaded_registration_manifest(
        self,
        metadata: UploadModelMetadata,
        uploads: list[UploadedPayload],
    ) -> tuple[ModelManifest, list[str]]:
        if not uploads:
            raise RegistryValidationError(
                "Upload the existing registration config before continuing.",
                field_errors={"registration_config": "Upload the existing registration config."},
            )

        warnings: list[str] = []
        if len(uploads) > 1:
            warnings.append(
                "Only the first uploaded registration config is used for validation and normalization."
            )

        manifest = _parse_uploaded_registration_manifest(uploads[0])
        if manifest.framework.type != metadata.framework_type:
            raise RegistryValidationError(
                "The uploaded config does not match the selected model type.",
                field_errors={
                    "registration_config": (
                        "The uploaded config framework does not match the selected model type."
                    )
                },
            )
        if manifest.model_id != metadata.model_id:
            warnings.append(
                "The uploaded config uses a different model id. The wizard value will be saved."
            )
        if self._canonicalize_domain(manifest.domain) != self._canonicalize_domain(metadata.domain):
            warnings.append(
                "The uploaded config uses a different domain. The wizard value will be saved."
            )
        if manifest.display_name != metadata.display_name:
            warnings.append(
                "The uploaded config uses a different display name. The wizard value will be saved."
            )
        return manifest, warnings

    def _normalize_metadata(self, metadata: UploadModelMetadata) -> UploadModelMetadata:
        framework_library = metadata.framework_library or _default_framework_library(
            metadata.framework_type
        )
        canonical_domain = self._canonicalize_domain(metadata.domain)
        return metadata.model_copy(
            update={
                "domain": canonical_domain,
                "framework_library": framework_library,
                "ui_display_name": metadata.ui_display_name or canonical_domain.title(),
                "color_token": metadata.color_token or canonical_domain,
                "group": metadata.group or f"{canonical_domain}-custom",
            }
        )

    def _ensure_model_id_available(self, model_id: str) -> None:
        if any(existing.manifest.model_id == model_id for existing in self.get_models()):
            raise RegistryValidationError(
                f"Model id '{model_id}' already exists.",
                field_errors={"metadata.model_id": "Choose a unique model id."},
            )

    def _artifact_selection_checks(
        self,
        framework_type: str,
        artifact_manifest: dict[str, list[UploadFileDescriptor]],
    ) -> tuple[list[ArtifactValidationSummary], dict[str, str]]:
        requirements = ARTIFACT_REQUIREMENTS.get(framework_type)
        if requirements is None:
            raise RegistryValidationError(
                f"Unsupported framework type '{framework_type}'.",
                field_errors={"metadata.framework_type": "Unsupported model type."},
            )

        summaries: list[ArtifactValidationSummary] = []
        field_errors: dict[str, str] = {}
        supported_slots = {requirement.slot for requirement in requirements}

        for slot in artifact_manifest:
            if slot not in supported_slots:
                field_errors[f"artifacts.{slot}"] = "This artifact slot is not used for the selected model type."

        for requirement in requirements:
            descriptors = artifact_manifest.get(requirement.slot, [])
            files = [Path(descriptor.name).name for descriptor in descriptors]
            error = _artifact_requirement_error(framework_type, requirement, files)
            if error is not None:
                field_errors[f"artifacts.{requirement.slot}"] = error
            summaries.append(
                ArtifactValidationSummary(
                    slot=requirement.slot,
                    title=requirement.title,
                    required=requirement.required,
                    valid=error is None,
                    message=error,
                    files=files,
                )
            )

        return summaries, field_errors

    def _dashboard_selection_warnings(
        self,
        dashboard_manifest: list[UploadFileDescriptor],
    ) -> list[str]:
        if not dashboard_manifest:
            return []

        references = [
            descriptor.relative_path or descriptor.name
            for descriptor in dashboard_manifest
        ]
        if not any(reference.replace("\\", "/").endswith("dashboard-manifest.json") for reference in references):
            raise RegistryValidationError(
                "Dashboard bundle must include dashboard-manifest.json.",
                field_errors={"dashboard": "Dashboard bundle must include dashboard-manifest.json."},
            )
        return []

    def _deactivate_domain_models(
        self,
        canonical_domain: str,
        *,
        except_model_id: str | None = None,
    ) -> None:
        for model in self._models_by_domain.get(canonical_domain, []):
            if model.manifest.model_id == except_model_id:
                continue
            if model.manifest.is_active:
                model.manifest.is_active = False
                self._write_manifest(model.config_path, model.manifest)

    def _next_priority(self) -> int:
        current_models = self.get_models()
        return max((model.manifest.priority for model in current_models), default=0) + 1

    def _allocate_model_dir(self, canonical_domain: str, model_id: str) -> Path:
        root = self._settings.model_discovery_root
        base_name = f"prod-model-{canonical_domain}"
        candidate = root / base_name
        if not candidate.exists():
            return candidate

        suffix = _slugify(model_id)
        candidate = root / f"{base_name}-{suffix}"
        counter = 2
        while candidate.exists():
            candidate = root / f"{base_name}-{suffix}-{counter}"
            counter += 1
        return candidate

    def _ordered_domains(self, *, active_only: bool) -> list[str]:
        alpha_rank = {domain: index for index, domain in enumerate(self._settings.alpha_domains)}

        domain_names = [
            domain
            for domain in self._models_by_domain
            if not active_only or self._select_active_model(self._models_by_domain[domain]) is not None
        ]

        def sort_key(domain: str) -> tuple[int, int, str]:
            models = self._models_by_domain.get(domain, [])
            primary = self._select_active_model(models) or (models[0] if models else None)
            priority = primary.manifest.priority if primary is not None else -1
            return (-priority, alpha_rank.get(domain, len(alpha_rank) + 1), domain)

        return sorted(domain_names, key=sort_key)

    def _select_active_model(self, models: list[RegisteredModel]) -> RegisteredModel | None:
        for model in models:
            _, can_activate, _ = self._model_runtime_status(model)
            if model.manifest.is_active and can_activate:
                return model
        return None

    def _model_sort_key(self, model: RegisteredModel) -> tuple[bool, int, str]:
        return (
            not model.manifest.is_active,
            -model.manifest.priority,
            model.manifest.display_name.lower(),
        )

    def _serialize_model(self, model: RegisteredModel) -> dict[str, object]:
        missing_artifacts = self._missing_runtime_artifacts(model)
        status, can_activate, status_reason = self._model_runtime_status(
            model,
            missing_artifacts=missing_artifacts,
        )
        dashboard = summarize_dashboard(model)
        return {
            "model_id": model.manifest.model_id,
            "domain": model.canonical_domain,
            "display_name": model.manifest.display_name,
            "description": model.manifest.description,
            "version": model.manifest.version,
            "framework_type": model.manifest.framework.type,
            "framework_task": model.manifest.framework.task,
            "framework_library": model.manifest.framework.library,
            "backbone": model.manifest.framework.backbone
            or model.manifest.framework.base_model,
            "architecture": model.manifest.framework.architecture,
            "runtime_device": model.manifest.runtime.device,
            "runtime_max_sequence_length": model.manifest.runtime.max_sequence_length,
            "output_type": model.manifest.label_type,
            "is_active": model.manifest.is_active and can_activate,
            "priority": model.manifest.priority,
            "notes": model.artifact_resolution.notes,
            "missing_artifacts": missing_artifacts,
            "status": status,
            "status_reason": status_reason,
            "can_activate": can_activate,
            "dashboard_status": dashboard["dashboard_status"],
            "dashboard_sections_available": dashboard["dashboard_sections_available"],
            "dashboard_sections_total": dashboard["dashboard_sections_total"],
            "dashboard_generated_at": dashboard["dashboard_generated_at"],
        }

    def _model_runtime_status(
        self,
        model: RegisteredModel,
        *,
        missing_artifacts: list[str] | None = None,
    ) -> tuple[str, bool, str | None]:
        effective_missing_artifacts = missing_artifacts or self._missing_runtime_artifacts(model)
        if effective_missing_artifacts:
            return (
                "missing_artifacts",
                False,
                f"{len(effective_missing_artifacts)} required runtime artifact(s) missing.",
            )

        if not self._plugin_registry.supports(model):
            framework = model.manifest.framework
            return (
                "incompatible",
                False,
                "No runtime plugin is registered for "
                f"{framework.type}/{framework.task or 'unknown'}"
                f"{' (' + framework.architecture + ')' if framework.architecture else ''}.",
            )

        return ("ready", True, None)

    def _missing_runtime_artifacts(self, model: RegisteredModel) -> list[str]:
        missing: list[str] = []
        for requirement in self._runtime_requirements(model):
            resolved_count = self._resolved_artifact_count(model, requirement.slot)
            if resolved_count >= requirement.min_files:
                continue

            configured_paths = self._configured_paths_for_slot(model, requirement.slot)
            if configured_paths:
                missing.extend(configured_paths)
            else:
                missing.append(requirement.slot)

        return list(dict.fromkeys(missing))

    def _runtime_requirements(self, model: RegisteredModel) -> tuple[ArtifactRequirement, ...]:
        framework = model.manifest.framework
        architecture = (framework.architecture or "").lower()

        if framework.type == "transformers" and framework.task == "sequence-classification":
            return tuple(
                requirement
                for requirement in ARTIFACT_REQUIREMENTS["transformers"]
                if requirement.slot in {"weights", "tokenizer", "config"}
            )

        if framework.type == "pytorch" and architecture == "bilstm-attention":
            return tuple(
                requirement
                for requirement in ARTIFACT_REQUIREMENTS["pytorch"]
                if requirement.slot in {"weights", "vocabulary"}
            )

        return (
            ArtifactRequirement(
                slot="weights",
                required=True,
                min_files=1,
                max_files=None,
                allowed_extensions=(),
                description="Fallback runtime requirement.",
            ),
        )

    def _configured_paths_for_slot(self, model: RegisteredModel, slot: str) -> list[str]:
        artifacts = model.manifest.artifacts
        match slot:
            case "weights" | "tokenizer" | "config" | "vocabulary":
                return list(getattr(artifacts, slot))
            case "label_map_file" | "label_classes_file" | "label_encoder_file":
                value = getattr(artifacts, slot)
                return [value] if value else []
            case _:
                return []

    def _resolved_artifact_count(self, model: RegisteredModel, slot: str) -> int:
        resolved = getattr(model.artifact_resolution, slot, None)
        if isinstance(resolved, list):
            return len(resolved)
        return 1 if resolved is not None else 0

    def _write_manifest(self, config_path: Path, manifest: ModelManifest) -> None:
        payload = manifest.to_yaml_dict()
        config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    def _canonicalize_domain(self, raw_domain: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", raw_domain.lower()).strip("-")
        return self._settings.domain_aliases.get(slug, slug)


def _load_manifest(config_path: Path) -> ModelManifest:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ModelManifest.from_yaml_dict(raw)


def _parse_uploaded_registration_manifest(upload: UploadedPayload) -> ModelManifest:
    try:
        raw = yaml.safe_load(upload.content.decode("utf-8"))
    except Exception as exc:
        raise RegistryValidationError(
            "The uploaded registration config is not valid YAML or JSON.",
            field_errors={"registration_config": "The uploaded registration config is invalid."},
        ) from exc

    if not isinstance(raw, dict):
        raise RegistryValidationError(
            "The uploaded registration config must be an object.",
            field_errors={"registration_config": "The uploaded registration config must be an object."},
        )

    try:
        return ModelManifest.from_yaml_dict(raw)
    except Exception as exc:
        raise RegistryValidationError(
            "The uploaded registration config does not match the expected manifest structure.",
            field_errors={
                "registration_config": (
                    "The uploaded registration config does not match the expected manifest structure."
                )
            },
        ) from exc


def _default_framework_library(framework_type: str) -> str:
    if framework_type == "transformers":
        return "huggingface"
    if framework_type == "pytorch":
        return "torch"
    if framework_type == "sklearn":
        return "sklearn"
    return framework_type


def _prefer_manifest_value(current: Any, default: Any, candidate: Any) -> Any:
    if current == default and candidate not in {None, ""}:
        return candidate
    return current


def _labels_are_placeholder(labels: list[UploadLabelClass]) -> bool:
    if len(labels) != 1:
        return False
    label = labels[0]
    return label.id == 0 and label.name == "class_0"


def _artifact_requirement_error(
    framework_type: str,
    requirement: ArtifactRequirement,
    files: list[str],
) -> str | None:
    if requirement.required and len(files) < requirement.min_files:
        return (
            f"{requirement.title} requires at least {requirement.min_files} file"
            f"{'' if requirement.min_files == 1 else 's'}."
        )

    if requirement.max_files is not None and len(files) > requirement.max_files:
        return f"{requirement.title} accepts at most {requirement.max_files} file(s)."

    for filename in files:
        suffix = Path(filename).suffix.lower()
        if requirement.allowed_extensions and suffix not in requirement.allowed_extensions:
            return (
                f"{filename} has an unsupported file type. Expected one of: "
                f"{', '.join(requirement.allowed_extensions)}."
            )

    normalized_names = {Path(filename).name.lower() for filename in files}
    if framework_type == "transformers" and requirement.slot == "config":
        if "config.json" not in normalized_names:
            return "Runtime Config Assets must include config.json for transformer models."

    return None


def _planned_artifact_manifest(
    framework_type: str,
    artifact_files: dict[str, list[str]],
) -> dict[str, object]:
    requirements = ARTIFACT_REQUIREMENTS.get(framework_type)
    if requirements is None:
        raise RegistryValidationError(
            f"Unsupported framework type '{framework_type}'.",
            field_errors={"metadata.framework_type": "Unsupported model type."},
        )

    planned_names: set[str] = set()
    artifact_manifest: dict[str, object] = {
        "weights": [],
        "tokenizer": [],
        "config": [],
        "vocabulary": [],
        "label_map_file": None,
        "label_classes_file": None,
        "label_encoder_file": None,
    }

    for requirement in requirements:
        slot_files = artifact_files.get(requirement.slot, [])
        normalized_paths = []
        for filename in slot_files:
            unique_name = _unique_name(planned_names, Path(filename).name)
            planned_names.add(unique_name)
            normalized_paths.append(unique_name)
        if requirement.max_files == 1:
            artifact_manifest[requirement.slot] = normalized_paths[0] if normalized_paths else None
        else:
            artifact_manifest[requirement.slot] = normalized_paths

    return artifact_manifest


def _save_artifacts(
    model_dir: Path,
    framework_type: str,
    artifact_uploads: list[UploadedPayload],
) -> dict[str, object]:
    requirements = ARTIFACT_REQUIREMENTS.get(framework_type)
    if requirements is None:
        raise ValueError(f"Unsupported framework type '{framework_type}'.")

    grouped: dict[str, list[UploadedPayload]] = defaultdict(list)
    for upload in artifact_uploads:
        slot, filename = _split_upload_slot(upload.path)
        grouped[slot].append(UploadedPayload(path=filename, content=upload.content))

    errors: list[str] = []
    field_errors: dict[str, str] = {}
    artifact_manifest: dict[str, object] = {
        "weights": [],
        "tokenizer": [],
        "config": [],
        "vocabulary": [],
        "label_map_file": None,
        "label_classes_file": None,
        "label_encoder_file": None,
    }

    for requirement in requirements:
        uploads = grouped.get(requirement.slot, [])
        upload_names = [upload.path for upload in uploads]
        requirement_error = _artifact_requirement_error(
            framework_type,
            requirement,
            upload_names,
        )
        if requirement_error is not None:
            errors.append(requirement_error)
            field_errors[f"artifacts.{requirement.slot}"] = requirement_error
            continue

        stored_paths: list[str] = []
        for upload in uploads:
            extension = Path(upload.path).suffix.lower()
            if requirement.allowed_extensions and extension not in requirement.allowed_extensions:
                message = (
                    f"{upload.path} is not valid for {requirement.title}. "
                    f"Expected one of: {', '.join(requirement.allowed_extensions)}."
                )
                errors.append(message)
                field_errors[f"artifacts.{requirement.slot}"] = message
                continue
            destination = _unique_file_path(model_dir, Path(upload.path).name)
            destination.write_bytes(upload.content)
            stored_paths.append(destination.name)

        if requirement.max_files == 1:
            artifact_manifest[requirement.slot] = stored_paths[0] if stored_paths else None
        else:
            artifact_manifest[requirement.slot] = stored_paths

    if errors:
        raise RegistryValidationError(
            " ".join(errors),
            field_errors=field_errors,
        )

    return artifact_manifest


def _save_dashboard(model_dir: Path, dashboard_uploads: list[UploadedPayload]) -> None:
    normalized_uploads = _normalize_dashboard_uploads(dashboard_uploads)
    if not normalized_uploads:
        return

    dashboard_dir = model_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    for upload in normalized_uploads:
        destination = (dashboard_dir / upload.path).resolve()
        if dashboard_dir.resolve() not in destination.parents:
            raise ValueError("Dashboard files must stay inside the dashboard directory.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(upload.content)

    manifest_path = dashboard_dir / "dashboard-manifest.json"
    if not manifest_path.exists():
        raise ValueError("Dashboard upload must include dashboard-manifest.json.")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Dashboard manifest is not valid JSON.") from exc

    path_map = {
        Path(upload.path).as_posix(): upload.path
        for upload in normalized_uploads
    }
    resolved_paths: dict[str, str] = {}

    for key, file_ref in dict(manifest.get("entrypoints", {})).items():
        resolved = _resolve_uploaded_dashboard_reference(path_map, str(file_ref))
        if resolved is None:
            raise ValueError(f"Dashboard entrypoint '{key}' points to a missing file.")
        resolved_paths[str(file_ref)] = resolved

    for section in manifest.get("sections", []):
        for file_ref in section.get("files", []):
            resolved = _resolve_uploaded_dashboard_reference(path_map, str(file_ref))
            if resolved is None:
                raise ValueError(
                    f"Dashboard section '{section.get('id', 'unknown')}' points to a missing file."
                )
            resolved_paths[str(file_ref)] = resolved

    dashboard_root = f"app/app-models/{model_dir.name}/dashboard"
    manifest["dashboard_root"] = dashboard_root
    manifest["entrypoints"] = {
        key: f"{dashboard_root}/{resolved_paths[str(value)]}"
        for key, value in dict(manifest.get("entrypoints", {})).items()
    }
    for section in manifest.get("sections", []):
        section["files"] = [
            f"{dashboard_root}/{resolved_paths[str(file_ref)]}"
            for file_ref in section.get("files", [])
        ]

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _normalize_dashboard_uploads(
    uploads: list[UploadedPayload],
) -> list[UploadedPayload]:
    if not uploads:
        return []

    parts_list = [Path(upload.path).parts for upload in uploads]
    strip_first_segment = bool(
        parts_list
        and all(len(parts) > 1 for parts in parts_list)
        and len({parts[0] for parts in parts_list}) == 1
    )

    normalized: list[UploadedPayload] = []
    for upload in uploads:
        parts = list(Path(upload.path).parts)
        if strip_first_segment and len(parts) > 1:
            parts = parts[1:]
        normalized.append(
            UploadedPayload(path=Path(*parts).as_posix(), content=upload.content)
        )
    return normalized


def _resolve_uploaded_dashboard_reference(
    upload_map: dict[str, str],
    file_ref: str,
) -> str | None:
    path = Path(file_ref)
    candidates = [path.as_posix()]
    if "dashboard" in path.parts:
        dashboard_index = path.parts.index("dashboard")
        candidates.append(Path(*path.parts[dashboard_index + 1 :]).as_posix())
    candidates.append(path.name)

    for candidate in candidates:
        if candidate in upload_map:
            return upload_map[candidate]

    for existing_path in upload_map.values():
        if Path(existing_path).name == path.name:
            return existing_path
    return None


def _split_upload_slot(raw_path: str) -> tuple[str, str]:
    normalized = Path(raw_path).as_posix()
    if "/" not in normalized:
        raise ValueError(
            "Each uploaded artifact must include an artifact slot prefix, e.g. 'weights/model.safetensors'."
        )
    slot, filename = normalized.split("/", 1)
    if not slot or not filename:
        raise ValueError("Artifact upload path is malformed.")
    return slot, filename


def _unique_file_path(root: Path, filename: str) -> Path:
    candidate = root / filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2
    while candidate.exists():
        candidate = root / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def _unique_name(existing: set[str], filename: str) -> str:
    candidate = filename
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2
    while candidate in existing:
        candidate = f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

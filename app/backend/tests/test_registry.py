from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.core.settings import Settings
from app.inference.factory import InferencePluginRegistry
from app.registry.model_registry import (
    ModelRegistry,
    RegistrationOutcome,
    RegistryValidationError,
    UploadedPayload,
)
from app.schemas.models import (
    HuggingFacePreflightRequest,
    LocalUploadPreflightRequest,
    UploadFileDescriptor,
    UploadLabelClass,
    UploadModelMetadata,
)
from app.services.huggingface_import import HuggingFaceInspection, HuggingFaceRepoFile


def _write_model_manifest(
    model_dir: Path,
    *,
    model_id: str,
    domain: str,
    display_name: str,
    is_active: bool,
    priority: int,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model-config.yaml").write_text(
        f"""
model_id: "{model_id}"
domain: "{domain}"
display_name: "{display_name}"
is_active: {"true" if is_active else "false"}
priority: {priority}
framework:
  type: "transformers"
  task: "sequence-classification"
artifacts:
  weights:
    - "model.safetensors"
  tokenizer:
    - "tokenizer.json"
  config:
    - "config.json"
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "neutral"
""",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")


def _build_registry(
    tmp_path: Path,
    *,
    hf_service: FakeHuggingFaceService | None = None,
) -> ModelRegistry:
    settings = Settings(model_discovery_root=tmp_path)
    registry = ModelRegistry(
        settings=settings,
        plugin_registry=InferencePluginRegistry(),
        hf_import_service=hf_service,
    )
    registry.discover()
    return registry


def _default_labels() -> list[UploadLabelClass]:
    return [
        UploadLabelClass(id=0, name="negative", display_name="Negative"),
        UploadLabelClass(id=1, name="positive", display_name="Positive"),
    ]


def _build_metadata(**overrides) -> UploadModelMetadata:
    payload = {
        "model_id": "sentiment-demo",
        "domain": "sentiment",
        "display_name": "Sentiment Demo",
        "description": "Demo model",
        "version": "v1",
        "enable_on_upload": False,
        "framework_type": "transformers",
        "framework_task": "sequence-classification",
        "framework_library": "huggingface",
        "framework_problem_type": "single_label_classification",
        "backbone": "distilbert-base-uncased",
        "architecture": "DistilBertForSequenceClassification",
        "base_model": "distilbert-base-uncased",
        "embeddings": None,
        "output_type": "single-label-classification",
        "runtime_device": "auto",
        "runtime_max_sequence_length": 128,
        "runtime_batch_size": 1,
        "runtime_truncation": True,
        "runtime_padding": True,
        "runtime_preprocessing": None,
        "ui_display_name": "Sentiment",
        "color_token": "sentiment",
        "group": "sentiment-custom",
        "labels": _default_labels(),
        "model_payload": {},
    }
    payload.update(overrides)
    return UploadModelMetadata(**payload)


def _build_local_payload(
    *,
    metadata: UploadModelMetadata | None = None,
    framework_type: str = "transformers",
    artifact_manifest: dict[str, list[UploadFileDescriptor]] | None = None,
) -> LocalUploadPreflightRequest:
    effective_metadata = metadata or _build_metadata(framework_type=framework_type)
    manifest = artifact_manifest or {
        "weights": [UploadFileDescriptor(name="model.safetensors", size_bytes=42)],
        "tokenizer": [
            UploadFileDescriptor(name="tokenizer.json", size_bytes=11),
            UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
        ],
        "config": [UploadFileDescriptor(name="config.json", size_bytes=9)],
    }
    return LocalUploadPreflightRequest(
        metadata=effective_metadata,
        artifact_manifest=manifest,
    )


def _build_local_uploads(
    *,
    framework_type: str = "transformers",
) -> list[UploadedPayload]:
    if framework_type == "transformers":
        uploads = [UploadedPayload(path="weights/model.safetensors", content=b"weights")]
        uploads.extend(
            [
                UploadedPayload(path="tokenizer/tokenizer.json", content=b"{}"),
                UploadedPayload(path="tokenizer/tokenizer_config.json", content=b"{}"),
                UploadedPayload(path="config/config.json", content=b"{}"),
            ]
        )
    elif framework_type == "pytorch":
        uploads = [UploadedPayload(path="weights/model.pt", content=b"weights")]
        uploads.extend(
            [
                UploadedPayload(path="vocabulary/vocab.pkl", content=b"vocab"),
                UploadedPayload(path="config/config.json", content=b"{}"),
            ]
        )
    else:
        uploads = [UploadedPayload(path="weights/model.pkl", content=b"weights")]
        uploads.append(UploadedPayload(path="config/features.json", content=b"{}"))
    return uploads


def _uploaded_registration_config(**overrides) -> UploadedPayload:
    manifest = {
        "model_id": "sentiment-demo",
        "domain": "sentiment",
        "display_name": "Sentiment Demo Config",
        "framework": {
            "type": "transformers",
            "task": "sequence-classification",
            "library": "huggingface",
            "backbone": "distilbert-base-uncased",
        },
        "artifacts": {
            "weights": ["model.safetensors"],
            "tokenizer": ["tokenizer.json"],
            "config": ["config.json"],
        },
        "runtime": {
            "max_sequence_length": 196,
            "truncation": True,
            "padding": True,
            "batch_size": 4,
            "device": "cpu",
        },
        "labels": {
            "type": "single-label-classification",
            "classes": [
                {"id": 0, "name": "negative", "display_name": "Negative"},
                {"id": 1, "name": "positive", "display_name": "Positive"},
            ],
        },
        "model": {"dropout": 0.1},
        "ui": {
            "domain_display_name": "Sentiment",
            "color_token": "sentiment",
            "group": "sentiment-core",
        },
    }
    manifest.update(overrides)
    return UploadedPayload(
        path="model-config.yaml",
        content=(yaml.safe_dump(manifest, sort_keys=False) + "\n").encode("utf-8"),
    )


class FakeHuggingFaceService:
    def __init__(self, inspection: HuggingFaceInspection | Exception) -> None:
        self._inspection = inspection
        self.download_calls = 0

    def inspect(self, repo_input: str) -> HuggingFaceInspection:
        if isinstance(self._inspection, Exception):
            raise self._inspection
        return self._inspection

    def download_to_directory(
        self,
        inspection: HuggingFaceInspection,
        destination_dir: Path,
    ) -> dict[str, object]:
        self.download_calls += 1
        destination_dir.mkdir(parents=True, exist_ok=True)
        artifact_manifest: dict[str, object] = {
            "weights": [],
            "tokenizer": [],
            "config": [],
            "vocabulary": [],
            "label_map_file": None,
            "label_classes_file": None,
            "label_encoder_file": None,
        }
        for slot, files in inspection.download_plan.items():
            stored: list[str] = []
            for file in files:
                filename = Path(file.path).name
                (destination_dir / filename).write_text("stub", encoding="utf-8")
                stored.append(filename)
            artifact_manifest[slot] = stored
        return artifact_manifest


def _hf_inspection(
    *,
    blocking_reasons: list[str] | None = None,
    detected_framework_type: str | None = "transformers",
    detected_task: str | None = "sequence-classification",
    required_files: list[HuggingFaceRepoFile] | None = None,
) -> HuggingFaceInspection:
    weights = HuggingFaceRepoFile(
        path="model.safetensors",
        category="weights",
        required=True,
        size_bytes=128,
    )
    tokenizer = HuggingFaceRepoFile(
        path="tokenizer.json",
        category="tokenizer",
        required=True,
        size_bytes=32,
    )
    config = HuggingFaceRepoFile(
        path="config.json",
        category="config",
        required=True,
        size_bytes=16,
    )
    return HuggingFaceInspection(
        repo_id="org/demo-model",
        repo_url="https://huggingface.co/org/demo-model",
        detected_framework_type=detected_framework_type,
        detected_task=detected_task,
        framework_library="huggingface",
        architecture="DistilBertForSequenceClassification",
        backbone="distilbert-base-uncased",
        base_model="distilbert-base-uncased",
        labels=[
            {"id": 0, "name": "negative", "display_name": "Negative"},
            {"id": 1, "name": "positive", "display_name": "Positive"},
        ],
        model_payload={"source_repo": "org/demo-model", "hidden_size": 768},
        required_files=required_files or [weights, tokenizer, config],
        download_plan={
            "weights": [weights],
            "tokenizer": [tokenizer],
            "config": [config],
        },
        estimated_download_size_bytes=176,
        disk_free_bytes=10_000_000,
        memory_total_bytes=10_000_000,
        memory_estimate_bytes=512,
        warnings=[],
        blocking_reasons=blocking_reasons or [],
    )


def test_registry_discovers_models_and_aliases_domains(tmp_path: Path) -> None:
    model_dir = tmp_path / "prod-model-abuse"
    model_dir.mkdir()
    (model_dir / "model-config.yaml").write_text(
        """
model_id: "abuse-model"
domain: "abuse-detection"
display_name: "Abuse"
is_active: true
priority: 100
framework:
  type: "transformers"
  task: "sequence-classification"
artifacts:
  weights:
    - "model.safetensors"
  tokenizer:
    - "tokenizer.json"
  config:
    - "config.json"
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "clean"
""",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    registry = _build_registry(tmp_path)
    catalog = registry.catalog()

    assert registry.domains() == ["abuse"]
    assert registry.get_active_models()[0].canonical_domain == "abuse"
    assert catalog[0]["models"][0]["framework_task"] == "sequence-classification"
    assert catalog[0]["models"][0]["runtime_device"] == "auto"
    assert catalog[0]["models"][0]["output_type"] == "single-label-classification"


def test_flattened_artifacts_are_resolved_from_basename(tmp_path: Path) -> None:
    model_dir = tmp_path / "prod-model-sentiment"
    model_dir.mkdir()
    (model_dir / "model-config.yaml").write_text(
        """
model_id: "sentiment-model"
domain: "sentiment"
display_name: "Sentiment"
is_active: true
priority: 100
framework:
  type: "transformers"
  task: "sequence-classification"
artifacts:
  base_dir: "models/nested/export"
  weights:
    - "model.safetensors"
  tokenizer:
    - "tokenizer.json"
  config:
    - "config.json"
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "neutral"
""",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    registry = _build_registry(tmp_path)
    active = registry.get_active_models()[0]
    assert active.artifact_resolution.weights[0] == model_dir / "model.safetensors"


def test_active_catalog_skips_domains_without_active_models(tmp_path: Path) -> None:
    _write_model_manifest(
        tmp_path / "prod-model-age",
        model_id="age-model",
        domain="age",
        display_name="Age Model",
        is_active=False,
        priority=100,
    )

    registry = _build_registry(tmp_path)

    assert registry.active_domains() == []
    assert registry.get_active_models() == []
    assert registry.catalog(active_only=True) == []
    assert registry.catalog(active_only=False)[0]["active_model_id"] is None


def test_updating_model_activation_switches_home_model(tmp_path: Path) -> None:
    _write_model_manifest(
        tmp_path / "prod-model-sentiment-a",
        model_id="sentiment-a",
        domain="sentiment",
        display_name="Sentiment A",
        is_active=True,
        priority=200,
    )
    _write_model_manifest(
        tmp_path / "prod-model-sentiment-b",
        model_id="sentiment-b",
        domain="sentiment",
        display_name="Sentiment B",
        is_active=False,
        priority=100,
    )

    registry = _build_registry(tmp_path)
    snapshot = registry.update_model("sentiment-b", is_active=True)

    assert snapshot["active_domains"][0]["active_model_id"] == "sentiment-b"
    updated_models = {
        model["model_id"]: model for model in snapshot["management_domains"][0]["models"]
    }
    assert updated_models["sentiment-a"]["is_active"] is False
    assert updated_models["sentiment-b"]["is_active"] is True


def test_dashboard_config_is_preserved_on_manifest_write(tmp_path: Path) -> None:
    model_dir = tmp_path / "prod-model-demo"
    model_dir.mkdir()
    (model_dir / "model-config.yaml").write_text(
        """
model_id: "demo-model"
domain: "demo"
display_name: "Demo"
is_active: true
priority: 10
framework:
  type: "transformers"
  task: "sequence-classification"
artifacts:
  weights:
    - "model.safetensors"
dashboard:
  builder: "generic-v1"
  sources:
    primary_evaluation: "inputs/primary.json"
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "neutral"
""",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")

    registry = _build_registry(tmp_path)
    registry.update_model("demo-model", display_name="Demo Updated")

    updated = yaml.safe_load((model_dir / "model-config.yaml").read_text(encoding="utf-8"))
    assert updated["display_name"] == "Demo Updated"
    assert updated["dashboard"]["builder"] == "generic-v1"
    assert updated["dashboard"]["sources"]["primary_evaluation"] == "inputs/primary.json"


def test_local_uploaded_config_happy_path(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload()
    config_upload = _uploaded_registration_config()

    preflight = registry.preflight_local_upload(
        payload,
        registration_config_uploads=[config_upload],
    )
    assert preflight.config_source == "uploaded"
    assert "runtime:" in preflight.config_preview
    assert preflight.normalized_metadata.runtime_batch_size == 4

    outcome = registry.register_local_upload(
        payload,
        artifact_uploads=_build_local_uploads(),
        dashboard_uploads=[],
        registration_config_uploads=[config_upload],
    )

    assert outcome.result.branch == "local"
    assert outcome.result.config_source == "uploaded"
    manifest_path = tmp_path / "prod-model-sentiment" / "model-config.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["display_name"] == "Sentiment Demo"
    assert manifest["runtime"]["batch_size"] == 4
    assert manifest["artifacts"]["weights"] == ["model.safetensors"]


def test_local_uploaded_config_happy_path_without_manual_metadata(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = LocalUploadPreflightRequest(
        metadata=None,
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.safetensors", size_bytes=42)],
            "tokenizer": [
                UploadFileDescriptor(name="tokenizer.json", size_bytes=11),
                UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
            ],
            "config": [UploadFileDescriptor(name="config.json", size_bytes=9)],
        },
    )
    config_upload = _uploaded_registration_config()

    preflight = registry.preflight_local_upload(
        payload,
        registration_config_uploads=[config_upload],
    )
    assert preflight.config_source == "uploaded"
    assert preflight.normalized_metadata.display_name == "Sentiment Demo Config"
    assert preflight.normalized_metadata.framework_type == "transformers"

    outcome = registry.register_local_upload(
        payload,
        artifact_uploads=_build_local_uploads(),
        dashboard_uploads=[],
        registration_config_uploads=[config_upload],
    )

    assert outcome.result.branch == "local"
    manifest_path = tmp_path / "prod-model-sentiment" / "model-config.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["display_name"] == "Sentiment Demo Config"
    assert manifest["framework"]["type"] == "transformers"


def test_local_uploaded_config_missing_required_file_raises_field_error(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload()

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.register_local_upload(
            payload,
            artifact_uploads=[UploadedPayload(path="weights/model.safetensors", content=b"weights")],
            dashboard_uploads=[],
            registration_config_uploads=[_uploaded_registration_config()],
        )

    assert "artifacts.tokenizer" in exc_info.value.field_errors


def test_local_uploaded_config_invalid_manifest_is_rejected(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload()

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.preflight_local_upload(
            payload,
            registration_config_uploads=[
                UploadedPayload(path="model-config.yaml", content=b"labels: ["),
            ],
        )

    assert "registration_config" in exc_info.value.field_errors


def test_local_generated_config_happy_path(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        metadata=_build_metadata(
            model_id="complexity-demo",
            domain="complexity",
            display_name="Complexity Demo",
            framework_type="pytorch",
            framework_library="torch",
            architecture="bilstm-attention",
            runtime_device="cpu",
            runtime_preprocessing="normalize_text + texts_to_sequences",
            labels=[
                UploadLabelClass(id=0, name="simple", display_name="Simple"),
                UploadLabelClass(id=1, name="complex", display_name="Complex"),
            ],
        ),
        framework_type="pytorch",
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.pt", size_bytes=100)],
            "vocabulary": [UploadFileDescriptor(name="vocab.pkl", size_bytes=10)],
            "config": [UploadFileDescriptor(name="config.json", size_bytes=10)],
        },
    )

    preflight = registry.preflight_local_upload(payload, registration_config_uploads=[])
    assert preflight.config_source == "generated"
    assert "preprocessing: normalize_text + texts_to_sequences" in preflight.config_preview

    outcome = registry.register_local_upload(
        payload,
        artifact_uploads=_build_local_uploads(framework_type="pytorch"),
        dashboard_uploads=[],
        registration_config_uploads=[],
    )

    assert outcome.result.branch == "local"
    manifest = yaml.safe_load(
        (tmp_path / "prod-model-complexity" / "model-config.yaml").read_text(encoding="utf-8")
    )
    assert manifest["framework"]["type"] == "pytorch"
    assert manifest["runtime"]["preprocessing"] == "normalize_text + texts_to_sequences"


def test_local_generated_config_defaults_label_display_name_to_name(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        metadata=_build_metadata(
            labels=[
                UploadLabelClass(id=0, name="negative", display_name=None),
                UploadLabelClass(id=1, name="positive", display_name="Class 1"),
            ],
        ),
    )

    preflight = registry.preflight_local_upload(payload, registration_config_uploads=[])

    assert preflight.normalized_metadata.labels[0].display_name == "negative"
    assert preflight.normalized_metadata.labels[1].display_name == "positive"
    assert 'display_name: positive' in preflight.config_preview


def test_transformer_preflight_requires_config_json(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.safetensors", size_bytes=42)],
            "tokenizer": [
                UploadFileDescriptor(name="tokenizer.json", size_bytes=11),
                UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
            ],
            "config": [UploadFileDescriptor(name="training_args.bin", size_bytes=9)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.preflight_local_upload(payload, registration_config_uploads=[])

    assert (
        exc_info.value.field_errors["artifacts.config"]
        == "Runtime Config Assets must include config.json. "
        "training_args.bin is not a supported runtime config file for this slot."
    )


def test_transformer_preflight_rejects_non_weight_file_in_weights_slot(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="training_args.bin", size_bytes=9)],
            "tokenizer": [
                UploadFileDescriptor(name="tokenizer.json", size_bytes=11),
                UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
            ],
            "config": [UploadFileDescriptor(name="config.json", size_bytes=9)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.preflight_local_upload(payload, registration_config_uploads=[])

    assert (
        exc_info.value.field_errors["artifacts.weights"]
        == "Weights only accept model.safetensors or pytorch_model.bin for transformer "
        "sequence-classification uploads. training_args.bin is not a valid transformer weight file."
    )


def test_transformer_preflight_rejects_incomplete_tokenizer_bundle(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.safetensors", size_bytes=42)],
            "tokenizer": [
                UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
            ],
            "config": [UploadFileDescriptor(name="config.json", size_bytes=9)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.preflight_local_upload(payload, registration_config_uploads=[])

    assert (
        exc_info.value.field_errors["artifacts.tokenizer"]
        == "Tokenizer Assets are incomplete. Missing one tokenizer definition file "
        "(tokenizer.json, vocab.txt, tokenizer.model, spiece.model, sentencepiece.bpe.model, "
        "or vocab.json with merges.txt)."
    )


def test_transformer_import_requires_config_json(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.safetensors", size_bytes=42)],
            "tokenizer": [
                UploadFileDescriptor(name="tokenizer.json", size_bytes=11),
                UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
            ],
            "config": [UploadFileDescriptor(name="training_args.bin", size_bytes=9)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.register_local_upload(
            payload,
            artifact_uploads=[
                UploadedPayload(path="weights/model.safetensors", content=b"weights"),
                UploadedPayload(path="tokenizer/tokenizer.json", content=b"{}"),
                UploadedPayload(path="tokenizer/tokenizer_config.json", content=b"{}"),
                UploadedPayload(path="config/training_args.bin", content=b"bin"),
            ],
            dashboard_uploads=[],
            registration_config_uploads=[],
        )

    assert (
        exc_info.value.field_errors["artifacts.config"]
        == "Runtime Config Assets must include config.json. "
        "training_args.bin is not a supported runtime config file for this slot."
    )


def test_transformer_import_rejects_non_weight_file_in_weights_slot(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="training_args.bin", size_bytes=9)],
            "tokenizer": [
                UploadFileDescriptor(name="tokenizer.json", size_bytes=11),
                UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
            ],
            "config": [UploadFileDescriptor(name="config.json", size_bytes=9)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.register_local_upload(
            payload,
            artifact_uploads=[
                UploadedPayload(path="weights/training_args.bin", content=b"bin"),
                UploadedPayload(path="tokenizer/tokenizer.json", content=b"{}"),
                UploadedPayload(path="tokenizer/tokenizer_config.json", content=b"{}"),
                UploadedPayload(path="config/config.json", content=b"{}"),
            ],
            dashboard_uploads=[],
            registration_config_uploads=[],
        )

    assert (
        exc_info.value.field_errors["artifacts.weights"]
        == "Weights only accept model.safetensors or pytorch_model.bin for transformer "
        "sequence-classification uploads. training_args.bin is not a valid transformer weight file."
    )


def test_transformer_import_rejects_incomplete_tokenizer_bundle(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.safetensors", size_bytes=42)],
            "tokenizer": [
                UploadFileDescriptor(name="tokenizer_config.json", size_bytes=8),
            ],
            "config": [UploadFileDescriptor(name="config.json", size_bytes=9)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.register_local_upload(
            payload,
            artifact_uploads=[
                UploadedPayload(path="weights/model.safetensors", content=b"weights"),
                UploadedPayload(path="tokenizer/tokenizer_config.json", content=b"{}"),
                UploadedPayload(path="config/config.json", content=b"{}"),
            ],
            dashboard_uploads=[],
            registration_config_uploads=[],
        )

    assert (
        exc_info.value.field_errors["artifacts.tokenizer"]
        == "Tokenizer Assets are incomplete. Missing one tokenizer definition file "
        "(tokenizer.json, vocab.txt, tokenizer.model, spiece.model, sentencepiece.bpe.model, "
        "or vocab.json with merges.txt)."
    )


def test_local_generated_config_duplicate_model_id_is_rejected(tmp_path: Path) -> None:
    _write_model_manifest(
        tmp_path / "prod-model-sentiment",
        model_id="sentiment-demo",
        domain="sentiment",
        display_name="Existing",
        is_active=True,
        priority=10,
    )
    registry = _build_registry(tmp_path)

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.preflight_local_upload(
            _build_local_payload(),
            registration_config_uploads=[],
        )

    assert exc_info.value.field_errors["metadata.model_id"] == "Choose a unique model id."


def test_huggingface_happy_path_imports_into_registry(tmp_path: Path) -> None:
    fake_service = FakeHuggingFaceService(_hf_inspection())
    registry = _build_registry(tmp_path, hf_service=fake_service)
    payload = HuggingFacePreflightRequest(
        repo="org/demo-model",
        metadata=_build_metadata(
            model_id="hf-demo",
            display_name="HF Demo",
            domain="sentiment",
            ui_display_name="Sentiment",
        ),
    )

    preflight = registry.preflight_huggingface_import(payload)
    assert preflight.ready_to_import is True
    assert preflight.detected_framework_type == "transformers"
    assert "source_repo" in preflight.config_preview

    outcome = registry.import_huggingface_model(payload)
    assert isinstance(outcome, RegistrationOutcome)
    assert outcome.result.source == "huggingface"
    assert outcome.result.config_source == "generated"
    assert fake_service.download_calls == 1
    manifest = yaml.safe_load(
        (tmp_path / "prod-model-sentiment" / "model-config.yaml").read_text(encoding="utf-8")
    )
    assert manifest["framework"]["type"] == "transformers"
    assert manifest["model"]["source_repo"] == "org/demo-model"


def test_huggingface_invalid_link_returns_field_error(tmp_path: Path) -> None:
    fake_service = FakeHuggingFaceService(
        ValueError("Paste a full model URL or a repo id like org/model-name.")
    )
    registry = _build_registry(tmp_path, hf_service=fake_service)
    payload = HuggingFacePreflightRequest(repo="broken", metadata=_build_metadata(model_id="hf-bad"))

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.preflight_huggingface_import(payload)

    assert exc_info.value.field_errors["huggingface.repo"].startswith("Paste a full model URL")


def test_huggingface_unsupported_repo_is_reported(tmp_path: Path) -> None:
    fake_service = FakeHuggingFaceService(
        _hf_inspection(
            detected_framework_type=None,
            detected_task="feature-extraction",
            blocking_reasons=["Only sequence-classification models are supported for Hugging Face import."],
        )
    )
    registry = _build_registry(tmp_path, hf_service=fake_service)
    payload = HuggingFacePreflightRequest(repo="org/unsupported", metadata=_build_metadata(model_id="hf-unsupported"))

    preflight = registry.preflight_huggingface_import(payload)
    assert preflight.ready_to_import is False
    assert preflight.blocking_reasons


def test_huggingface_insufficient_space_is_reported(tmp_path: Path) -> None:
    fake_service = FakeHuggingFaceService(
        _hf_inspection(
            blocking_reasons=["Not enough free disk space is available for this import."]
        )
    )
    registry = _build_registry(tmp_path, hf_service=fake_service)
    payload = HuggingFacePreflightRequest(repo="org/huge-model", metadata=_build_metadata(model_id="hf-disk"))

    preflight = registry.preflight_huggingface_import(payload)
    assert preflight.ready_to_import is False
    assert preflight.blocking_reasons == ["Not enough free disk space is available for this import."]


def test_huggingface_missing_required_files_are_reported(tmp_path: Path) -> None:
    required_files = [
        HuggingFaceRepoFile(
            path="model.safetensors",
            category="weights",
            required=True,
            size_bytes=128,
        ),
        HuggingFaceRepoFile(
            path="tokenizer.json | vocab.txt | tokenizer.model",
            category="tokenizer",
            required=True,
            size_bytes=None,
            message="Missing tokenizer assets.",
        ),
    ]
    fake_service = FakeHuggingFaceService(
        _hf_inspection(
            required_files=required_files,
            blocking_reasons=["Required tokenizer assets were not found in the repo."],
        )
    )
    registry = _build_registry(tmp_path, hf_service=fake_service)
    payload = HuggingFacePreflightRequest(repo="org/missing-tokenizer", metadata=_build_metadata(model_id="hf-missing"))

    preflight = registry.preflight_huggingface_import(payload)
    assert preflight.ready_to_import is False
    assert any(not item.available for item in preflight.required_files)

from __future__ import annotations

import json
import pickle
from io import BytesIO
from pathlib import Path

import pytest
import yaml
import numpy as np
try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on test env
    torch = None

from app.core.settings import Settings
from app.inference.factory import InferencePluginRegistry
from app.inference.runners.torch_sequence import EmbeddingMLP
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


class FakeRunner:
    def predict(self, text: str):  # pragma: no cover - tiny test stub
        return {"text": text}


class KeywordTransformer:
    def __init__(self, keyword: str) -> None:
        self.keyword = keyword

    def transform(self, texts):
        return np.asarray(
            [[1.0 if self.keyword in str(text).lower() else 0.0] for text in texts],
            dtype=float,
        )


class ThresholdEstimator:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        score = float(np.asarray(features)[0][0])
        if score >= 0.5:
            return np.asarray([[0.1, 0.9]], dtype=float)
        return np.asarray([[0.85, 0.15]], dtype=float)


class LogisticRegression:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features):
        _ = self.multi_class
        first_item = features[0]
        if isinstance(first_item, str):
            score = 1.0 if "old" in first_item.lower() else 0.0
        else:
            score = float(np.asarray(features)[0][0])
        if score >= 0.5:
            return np.asarray([[0.2, 0.8]], dtype=float)
        return np.asarray([[0.7, 0.3]], dtype=float)


class SimpleLabelEncoder:
    def __init__(self, classes: list[str]) -> None:
        self.classes_ = np.asarray(classes)

    def inverse_transform(self, values):
        return np.asarray([self.classes_[int(value)] for value in values])


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


def _pickle_bytes(value: object) -> bytes:
    return pickle.dumps(value)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value).encode("utf-8")


def _build_torch_embedding_checkpoint(num_classes: int = 2) -> bytes:
    if torch is None:
        pytest.skip("torch is not installed in the current test environment.")
    model = EmbeddingMLP(
        torch_module=torch,
        embedding_matrix=np.zeros((8, 4), dtype=np.float32),
        num_classes=num_classes,
        hidden_dim=6,
        dropout=0.0,
    )
    buffer = BytesIO()
    torch.save({"model_state": model.state_dict()}, buffer)
    return buffer.getvalue()


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


def test_updating_model_activation_switches_home_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    monkeypatch.setattr(registry, "get_runner", lambda model: FakeRunner())
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


def test_local_sklearn_upload_can_activate_and_derive_labels_from_encoder(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        metadata=_build_metadata(
            model_id="complexity-sklearn",
            domain="complexity",
            display_name="Complexity Sklearn",
            enable_on_upload=True,
            framework_type="sklearn",
            framework_library="sklearn",
            architecture="tf-idf-logreg",
            runtime_device="cpu",
            runtime_preprocessing="normalize_text",
            labels=[UploadLabelClass(id=0, name="class_0", display_name="Class 0")],
        ),
        framework_type="sklearn",
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.pkl", size_bytes=96)],
            "config": [UploadFileDescriptor(name="vectorizer.pkl", size_bytes=64)],
            "label_encoder_file": [
                UploadFileDescriptor(name="label_encoder.pkl", size_bytes=24)
            ],
        },
    )

    preflight = registry.preflight_local_upload(payload, registration_config_uploads=[])
    assert preflight.ready is True
    assert preflight.normalized_metadata.framework_type == "sklearn"

    outcome = registry.register_local_upload(
        payload,
        artifact_uploads=[
            UploadedPayload(
                path="weights/model.pkl",
                content=_pickle_bytes(ThresholdEstimator()),
            ),
            UploadedPayload(
                path="config/vectorizer.pkl",
                content=_pickle_bytes(KeywordTransformer("complex")),
            ),
            UploadedPayload(
                path="label_encoder_file/label_encoder.pkl",
                content=_pickle_bytes(SimpleLabelEncoder(["easy", "complex"])),
            ),
        ],
        dashboard_uploads=[],
        registration_config_uploads=[],
    )

    assert outcome.result.is_active is True
    assert outcome.result.status == "ready"
    manifest = yaml.safe_load(
        (tmp_path / "prod-model-complexity" / "model-config.yaml").read_text(encoding="utf-8")
    )
    assert manifest["labels"]["classes"] == [
        {"id": 0, "name": "easy", "display_name": "easy"},
        {"id": 1, "name": "complex", "display_name": "complex"},
    ]

    model = registry.get_model("complexity-sklearn")
    prediction = registry.get_runner(model).predict("This looks complex.")
    assert prediction.predicted_label == "complex"
    assert prediction.confidence > 0.5


def test_local_sklearn_activation_handles_legacy_logistic_regression_defaults(
    tmp_path: Path,
) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        metadata=_build_metadata(
            model_id="age-logreg-legacy",
            domain="age",
            display_name="Age LogReg Legacy",
            enable_on_upload=True,
            framework_type="sklearn",
            framework_library="sklearn",
            labels=[UploadLabelClass(id=0, name="class_0", display_name="Class 0")],
        ),
        framework_type="sklearn",
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.pkl", size_bytes=96)],
            "config": [UploadFileDescriptor(name="feature_config.json", size_bytes=16)],
            "label_encoder_file": [
                UploadFileDescriptor(name="label_encoder.pkl", size_bytes=24)
            ],
        },
    )

    outcome = registry.register_local_upload(
        payload,
        artifact_uploads=[
            UploadedPayload(
                path="weights/model.pkl",
                content=_pickle_bytes(LogisticRegression()),
            ),
            UploadedPayload(path="config/feature_config.json", content=b"{}"),
            UploadedPayload(
                path="label_encoder_file/label_encoder.pkl",
                content=_pickle_bytes(SimpleLabelEncoder(["young", "old"])),
            ),
        ],
        dashboard_uploads=[],
        registration_config_uploads=[],
    )

    assert outcome.result.is_active is True
    assert outcome.result.status == "ready"


def test_local_pytorch_upload_can_activate_embedding_mlp(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        metadata=_build_metadata(
            model_id="complexity-torch",
            domain="complexity",
            display_name="Complexity Torch",
            enable_on_upload=True,
            framework_type="pytorch",
            framework_library="torch",
            architecture="glove_mlp",
            embeddings="fasttext-wiki-news-subwords-300",
            runtime_device="cpu",
            runtime_preprocessing="normalize_text + preprocess_from_normalized",
            labels=[UploadLabelClass(id=0, name="class_0", display_name="Class 0")],
            model_config={"hidden_dim": 6, "dropout": 0.0},
        ),
        framework_type="pytorch",
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.pt", size_bytes=256)],
            "vocabulary": [UploadFileDescriptor(name="vocab.json", size_bytes=32)],
            "config": [UploadFileDescriptor(name="config.json", size_bytes=16)],
            "label_map_file": [UploadFileDescriptor(name="class2id.json", size_bytes=24)],
        },
    )

    outcome = registry.register_local_upload(
        payload,
        artifact_uploads=[
            UploadedPayload(
                path="weights/model.pt",
                content=_build_torch_embedding_checkpoint(),
            ),
            UploadedPayload(
                path="vocabulary/vocab.json",
                content=_json_bytes(
                    {
                        "<PAD>": 0,
                        "<UNK>": 1,
                        "registry": 2,
                        "activation": 3,
                        "smoke": 4,
                        "test": 5,
                    }
                ),
            ),
            UploadedPayload(path="config/config.json", content=b"{}"),
            UploadedPayload(
                path="label_map_file/class2id.json",
                content=_json_bytes({"negative": 0, "positive": 1}),
            ),
        ],
        dashboard_uploads=[],
        registration_config_uploads=[],
    )

    assert outcome.result.is_active is True
    assert outcome.result.status == "ready"
    manifest = yaml.safe_load(
        (tmp_path / "prod-model-complexity" / "model-config.yaml").read_text(encoding="utf-8")
    )
    assert manifest["framework"]["architecture"] == "embedding-mlp"
    assert manifest["labels"]["classes"] == [
        {"id": 0, "name": "negative", "display_name": "negative"},
        {"id": 1, "name": "positive", "display_name": "positive"},
    ]

    model = registry.get_model("complexity-torch")
    prediction = registry.get_runner(model).predict("Registry activation smoke test.")
    assert prediction.predicted_label in {"negative", "positive"}
    assert prediction.confidence > 0


def test_preflight_rejects_unsupported_preprocessing_for_sklearn(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        metadata=_build_metadata(
            model_id="invalid-sklearn",
            framework_type="sklearn",
            framework_library="sklearn",
            runtime_preprocessing="normalize_text + texts_to_sequences",
        ),
        framework_type="sklearn",
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.pkl", size_bytes=64)],
            "config": [UploadFileDescriptor(name="feature_config.json", size_bytes=16)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.preflight_local_upload(payload, registration_config_uploads=[])

    assert exc_info.value.field_errors["metadata.runtime_preprocessing"].startswith(
        "Unsupported preprocessing steps for the selected model type"
    )


def test_local_import_rejects_unsupported_artifact_slot_for_sklearn(tmp_path: Path) -> None:
    registry = _build_registry(tmp_path)
    payload = _build_local_payload(
        metadata=_build_metadata(
            model_id="sklearn-slot-check",
            framework_type="sklearn",
            framework_library="sklearn",
        ),
        framework_type="sklearn",
        artifact_manifest={
            "weights": [UploadFileDescriptor(name="model.pkl", size_bytes=64)],
            "config": [UploadFileDescriptor(name="feature_config.json", size_bytes=16)],
            "tokenizer": [UploadFileDescriptor(name="tokenizer.json", size_bytes=8)],
        },
    )

    with pytest.raises(RegistryValidationError) as exc_info:
        registry.register_local_upload(
            payload,
            artifact_uploads=[
                UploadedPayload(path="weights/model.pkl", content=_pickle_bytes(ThresholdEstimator())),
                UploadedPayload(path="config/feature_config.json", content=b"{}"),
                UploadedPayload(path="tokenizer/tokenizer.json", content=b"{}"),
            ],
            dashboard_uploads=[],
            registration_config_uploads=[],
        )

    assert exc_info.value.field_errors["artifacts.tokenizer"] == (
        "This artifact slot is not used for the selected model type."
    )


def test_catalog_marks_unsupported_pytorch_architecture_incompatible(tmp_path: Path) -> None:
    model_dir = tmp_path / "prod-model-complexity"
    model_dir.mkdir()
    (model_dir / "model-config.yaml").write_text(
        """
model_id: "torch-unsupported"
domain: "complexity"
display_name: "Unsupported Torch"
is_active: false
priority: 9
framework:
  type: "pytorch"
  task: "sequence-classification"
  library: "torch"
  architecture: "capsule-net"
artifacts:
  weights:
    - "model.pt"
  vocabulary:
    - "vocab.json"
  config:
    - "config.json"
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "negative"
""",
        encoding="utf-8",
    )
    (model_dir / "model.pt").write_bytes(b"weights")
    (model_dir / "vocab.json").write_text('{"<PAD>": 0, "<UNK>": 1}', encoding="utf-8")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    registry = _build_registry(tmp_path)
    catalog_model = registry.catalog(active_only=False)[0]["models"][0]

    assert catalog_model["status"] == "incompatible"
    assert "Unsupported PyTorch architecture 'capsule-net'" in str(catalog_model["status_reason"])

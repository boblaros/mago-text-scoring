from __future__ import annotations

from pathlib import Path

from app.core.settings import Settings
from app.inference.factory import InferencePluginRegistry
from app.registry.model_registry import ModelRegistry


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
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "neutral"
""",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")


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
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "clean"
""",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")

    settings = Settings(model_discovery_root=tmp_path)
    registry = ModelRegistry(settings=settings, plugin_registry=InferencePluginRegistry())
    registry.discover()
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
labels:
  type: "single-label-classification"
  classes:
    - id: 0
      name: "neutral"
""",
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")

    settings = Settings(model_discovery_root=tmp_path)
    registry = ModelRegistry(settings=settings, plugin_registry=InferencePluginRegistry())
    registry.discover()

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

    settings = Settings(model_discovery_root=tmp_path)
    registry = ModelRegistry(settings=settings, plugin_registry=InferencePluginRegistry())
    registry.discover()

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

    settings = Settings(model_discovery_root=tmp_path)
    registry = ModelRegistry(settings=settings, plugin_registry=InferencePluginRegistry())
    registry.discover()

    snapshot = registry.update_model("sentiment-b", is_active=True)

    assert snapshot["active_domains"][0]["active_model_id"] == "sentiment-b"
    updated_models = {model["model_id"]: model for model in snapshot["management_domains"][0]["models"]}
    assert updated_models["sentiment-a"]["is_active"] is False
    assert updated_models["sentiment-b"]["is_active"] is True

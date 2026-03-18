from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from app.registry.contracts import ModelManifest, RegisteredModel, ResolvedArtifacts
from app.registry.dashboard_builder import build_model_dashboard
from app.registry.dashboard_loader import load_dashboard_manifest, load_model_dashboard, summarize_dashboard


ONE_BY_ONE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc`\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_model_config(model_dir: Path, dashboard_config: dict) -> None:
    payload = {
        "model_id": "demo-model",
        "domain": "demo",
        "display_name": "Demo Model",
        "description": "Demo model for dashboard builder tests.",
        "is_active": True,
        "priority": 1,
        "framework": {
            "type": "transformers",
            "task": "sequence-classification",
            "library": "huggingface",
            "backbone": "demo-backbone",
        },
        "artifacts": {
            "weights": ["model.safetensors"],
        },
        "runtime": {
            "device": "auto",
            "max_sequence_length": 256,
            "batch_size": 8,
            "padding": True,
            "truncation": True,
        },
        "labels": {
            "type": "single-label-classification",
            "classes": [
                {"id": 0, "name": "negative", "display_name": "Negative"},
                {"id": 1, "name": "positive", "display_name": "Positive"},
            ],
        },
        "dashboard": dashboard_config,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model-config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_text("stub", encoding="utf-8")


def _registered_model(model_dir: Path) -> RegisteredModel:
    raw = yaml.safe_load((model_dir / "model-config.yaml").read_text(encoding="utf-8"))
    manifest = ModelManifest.from_yaml_dict(raw)
    return RegisteredModel(
        manifest=manifest,
        config_path=model_dir / "model-config.yaml",
        model_dir=model_dir,
        canonical_domain=manifest.domain,
        artifact_resolution=ResolvedArtifacts(),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_model_dashboard_with_minimal_sources(tmp_path: Path) -> None:
    repo_root = tmp_path
    app_root = repo_root / "app"
    model_dir = app_root / "app-models" / "prod-model-demo"

    _write_model_config(
        model_dir,
        {
            "builder": "generic-v1",
            "sources": {
                "experiment_config": "inputs/experiment.json",
                "primary_evaluation": {
                    "path": "inputs/primary-evaluation.json",
                    "model_name": "Demo Model",
                },
            },
        },
    )
    (model_dir / "inputs").mkdir()
    (model_dir / "inputs" / "experiment.json").write_text(
        '{"seed": 7, "output_paths": {"reports": "outputs/reports"}}',
        encoding="utf-8",
    )
    (model_dir / "inputs" / "primary-evaluation.json").write_text(
        (
            '{"val": {"accuracy": 0.81, "f1_macro": 0.79, "f1_weighted": 0.8}, '
            '"test": {"accuracy": 0.83, "f1_macro": 0.82, "f1_weighted": 0.825}, '
            '"best_model_dir": "exports/best"}'
        ),
        encoding="utf-8",
    )

    result = build_model_dashboard(model_dir, repo_root=repo_root, app_root=app_root)

    assert result.generated is True
    assert result.manifest_path == model_dir / "dashboard" / "dashboard-manifest.json"

    manifest = load_dashboard_manifest(model_dir)
    assert manifest is not None
    section_status = {section["id"]: section["status"] for section in manifest["sections"]}
    assert section_status["metadata"] == "available"
    assert section_status["summary"] == "available"
    assert section_status["evaluation"] == "available"
    assert section_status["benchmark"] == "missing"

    dashboard = load_model_dashboard(
        _registered_model(model_dir),
        lambda asset_path: f"/assets/{asset_path}",
    )
    assert dashboard.available is True
    assert "metadata/model.json" in dashboard.documents
    assert "metadata/experiment-config.json" in dashboard.documents
    assert "metrics/primary-evaluation.json" in dashboard.documents
    assert dashboard.overview is not None
    assert dashboard.source_audit is not None


def test_build_model_dashboard_supports_partial_sections(tmp_path: Path) -> None:
    repo_root = tmp_path
    app_root = repo_root / "app"
    model_dir = app_root / "app-models" / "prod-model-demo"

    _write_model_config(
        model_dir,
        {
            "builder": "generic-v1",
            "notes": ["Partial build is expected for this model."],
            "sources": {
                "primary_evaluation": "inputs/primary-evaluation.json",
                "benchmark": {
                    "path": "inputs/benchmark.csv",
                    "model_name": "Demo Model",
                },
                "training_history": "inputs/missing-history.json",
                "learning_curve": {
                    "status": "not_applicable",
                    "reason": "This model does not expose sample-size checkpoints.",
                },
                "class_distribution": {
                    "train": "inputs/train.csv",
                    "val": "inputs/val.csv",
                    "test": "inputs/test.csv",
                    "label_field": "label",
                    "dataset_field": "source_dataset",
                },
                "prediction_samples": {
                    "path": "inputs/prediction-samples.csv",
                    "production_prediction": {
                        "model": "Demo Model",
                        "label_field": "production_label",
                        "confidence_field": "production_conf",
                    },
                    "reference_prediction": {
                        "model": "Reference Model",
                        "label_field": "reference_label",
                        "confidence_field": "reference_conf",
                    },
                },
                "confusion_matrix": {
                    "path": "inputs/confusion.png",
                },
            },
        },
    )
    (model_dir / "inputs").mkdir()
    (model_dir / "inputs" / "primary-evaluation.json").write_text(
        (
            '{"val": {"accuracy": 0.75, "f1_macro": 0.74, "f1_weighted": 0.74}, '
            '"test": {"accuracy": 0.77, "f1_macro": 0.76, "f1_weighted": 0.76}}'
        ),
        encoding="utf-8",
    )
    _write_csv(
        model_dir / "inputs" / "benchmark.csv",
        [
            {
                "name": "Demo Model | test",
                "accuracy": "0.77",
                "f1_macro": "0.76",
                "f1_weighted": "0.76",
                "loss": "",
            },
            {
                "name": "Baseline | test",
                "accuracy": "0.62",
                "f1_macro": "0.61",
                "f1_weighted": "0.61",
                "loss": "",
            },
        ],
    )
    for split_name, rows in {
        "train": [
            {"label": 0, "source_dataset": "reddit"},
            {"label": 1, "source_dataset": "reddit"},
            {"label": 1, "source_dataset": "forum"},
        ],
        "val": [
            {"label": 0, "source_dataset": "reddit"},
            {"label": 1, "source_dataset": "forum"},
        ],
        "test": [
            {"label": 0, "source_dataset": "forum"},
            {"label": 1, "source_dataset": "forum"},
        ],
    }.items():
        _write_csv(model_dir / "inputs" / f"{split_name}.csv", rows)

    _write_csv(
        model_dir / "inputs" / "prediction-samples.csv",
        [
            {
                "example_id": 1,
                "text": "A calm example",
                "production_label": "Positive",
                "production_conf": "0.91",
                "reference_label": "Positive",
                "reference_conf": "0.74",
            },
            {
                "example_id": 2,
                "text": "A difficult example",
                "production_label": "Negative",
                "production_conf": "0.66",
                "reference_label": "Positive",
                "reference_conf": "0.51",
            },
        ],
    )
    (model_dir / "inputs" / "confusion.png").write_bytes(ONE_BY_ONE_PNG)

    result = build_model_dashboard(model_dir, repo_root=repo_root, app_root=app_root)

    assert result.generated is True
    manifest = load_dashboard_manifest(model_dir)
    assert manifest is not None
    section_status = {section["id"]: section["status"] for section in manifest["sections"]}
    assert section_status["benchmark"] == "available"
    assert section_status["training_curves"] == "missing"
    assert section_status["learning_curves"] == "not_applicable"
    assert section_status["confusion_matrix"] == "image_only"
    assert section_status["class_distribution"] == "available"
    assert section_status["samples"] == "available"

    dashboard = load_model_dashboard(
        _registered_model(model_dir),
        lambda asset_path: f"/assets/{asset_path}",
    )
    assert "metrics/benchmark-test.json" in dashboard.documents
    assert "distributions/class-distribution.json" in dashboard.documents
    assert "distributions/source-dataset-distribution.json" in dashboard.documents
    assert "samples/prediction-samples.json" in dashboard.documents
    assert any(image.section_id == "confusion_matrix" for image in dashboard.images)
    assert any(figure.section_id == "benchmark" for figure in dashboard.figures)


def test_build_model_dashboard_without_sources_creates_generic_skeleton(tmp_path: Path) -> None:
    repo_root = tmp_path
    app_root = repo_root / "app"
    model_dir = app_root / "app-models" / "prod-model-demo"

    _write_model_config(
        model_dir,
        {
            "builder": "generic-v1",
        },
    )

    result = build_model_dashboard(model_dir, repo_root=repo_root, app_root=app_root)

    assert result.generated is True
    manifest = load_dashboard_manifest(model_dir)
    assert manifest is not None
    section_status = {section["id"]: section["status"] for section in manifest["sections"]}
    assert section_status["metadata"] == "available"
    assert section_status["summary"] == "available"
    assert section_status["evaluation"] == "missing"
    assert section_status["benchmark"] == "missing"

    summary = summarize_dashboard(_registered_model(model_dir))
    assert summary["dashboard_status"] == "partial"
    assert summary["dashboard_sections_available"] == 2


def test_load_model_dashboard_recovers_distribution_assets_from_dashboard_bundle(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    app_root = repo_root / "app"
    model_dir = app_root / "app-models" / "prod-model-demo"
    dashboard_dir = model_dir / "dashboard"

    _write_model_config(
        model_dir,
        {
            "builder": "generic-v1",
        },
    )

    (dashboard_dir / "summary").mkdir(parents=True)
    (dashboard_dir / "distributions").mkdir(parents=True)
    (dashboard_dir / "figures").mkdir(parents=True)

    (dashboard_dir / "summary" / "overview.json").write_text(
        json.dumps({"generated_at": "2026-03-19T10:30:00+00:00"}, indent=2),
        encoding="utf-8",
    )
    (dashboard_dir / "summary" / "source-audit.json").write_text(
        json.dumps({"artifact_counts": {"distributions": 2}}, indent=2),
        encoding="utf-8",
    )
    (dashboard_dir / "distributions" / "class-distribution.json").write_text(
        json.dumps(
            {
                "overall": [
                    {"split": "overall", "label": "negative", "count": 14, "share": 0.56},
                    {"split": "overall", "label": "positive", "count": 11, "share": 0.44},
                ],
                "splits": [
                    {"split": "train", "label": "negative", "count": 8, "share": 0.57},
                    {"split": "train", "label": "positive", "count": 6, "share": 0.43},
                    {"split": "test", "label": "negative", "count": 6, "share": 0.55},
                    {"split": "test", "label": "positive", "count": 5, "share": 0.45},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (dashboard_dir / "distributions" / "source-dataset-distribution.json").write_text(
        json.dumps(
            {
                "overall": [
                    {"split": "overall", "source_dataset": "reddit", "count": 18, "share": 0.72},
                    {"split": "overall", "source_dataset": "forum", "count": 7, "share": 0.28},
                ],
                "splits": [
                    {"split": "train", "source_dataset": "reddit", "count": 10, "share": 0.71},
                    {"split": "train", "source_dataset": "forum", "count": 4, "share": 0.29},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (dashboard_dir / "figures" / "source-dataset-distribution.plotly.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": ["reddit", "forum"],
                        "y": [18, 7],
                    }
                ],
                "layout": {"title": {"text": "Source Dataset Distribution"}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "1.0.0",
        "generated_at": "2026-03-19T10:30:00+00:00",
        "dashboard_root": "app/app-models/prod-model-demo/dashboard",
        "entrypoints": {
            "overview": "app/app-models/prod-model-demo/dashboard/summary/overview.json",
            "source_audit": "app/app-models/prod-model-demo/dashboard/summary/source-audit.json",
        },
        "sections": [
            {
                "id": "summary",
                "title": "Summary",
                "status": "available",
                "files": [
                    "app/app-models/prod-model-demo/dashboard/summary/overview.json",
                    "app/app-models/prod-model-demo/dashboard/summary/source-audit.json",
                ],
            },
            {
                "id": "class_distribution",
                "title": "Class Distribution",
                "status": "missing",
                "files": [],
                "reason": "No class_distribution source was configured.",
            },
        ],
    }
    (dashboard_dir / "dashboard-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = summarize_dashboard(_registered_model(model_dir))
    assert summary["dashboard_status"] == "available"
    assert summary["dashboard_sections_available"] == 2
    assert summary["dashboard_sections_total"] == 2

    dashboard = load_model_dashboard(
        _registered_model(model_dir),
        lambda asset_path: f"/assets/{asset_path}",
    )

    assert dashboard.available is True
    assert "distributions/class-distribution.json" in dashboard.documents
    assert "distributions/source-dataset-distribution.json" in dashboard.documents

    class_section = next(
        section for section in (dashboard.manifest.sections if dashboard.manifest else []) if section.id == "class_distribution"
    )
    assert class_section.status == "available"
    assert class_section.reason is None
    assert "distributions/class-distribution.json" in class_section.files
    assert "distributions/source-dataset-distribution.json" in class_section.files
    assert "figures/source-dataset-distribution.plotly.json" in class_section.files
    assert "class-distribution" in class_section.charts
    assert "source-dataset-distribution" in class_section.charts

    distribution_figures = [
        figure for figure in dashboard.figures if figure.section_id == "class_distribution"
    ]
    assert {figure.id for figure in distribution_figures} == {
        "class-distribution",
        "source-dataset-distribution",
    }

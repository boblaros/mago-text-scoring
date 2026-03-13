from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from app.core.settings import APP_ROOT
from app.registry.contracts import LabelClass, ModelManifest


REPO_ROOT = APP_ROOT.parent
SECTION_ORDER = [
    "metadata",
    "summary",
    "evaluation",
    "benchmark",
    "training_curves",
    "learning_curves",
    "cross_dataset",
    "confusion_matrix",
    "class_distribution",
    "samples",
]
SECTION_TITLES = {
    "metadata": "Model Metadata",
    "summary": "Summary",
    "evaluation": "Primary Evaluation",
    "benchmark": "Benchmark",
    "training_curves": "Training Curves",
    "learning_curves": "Learning Curves",
    "cross_dataset": "Cross Dataset Evaluation",
    "confusion_matrix": "Confusion Matrix",
    "class_distribution": "Class Distribution",
    "samples": "Prediction Samples",
}
SECTION_DESCRIPTIONS = {
    "metadata": "Model identity, runtime settings, and optional experiment configuration.",
    "summary": "Dashboard status overview and source audit for the generated bundle.",
    "evaluation": "Primary validation and test metrics for the production model.",
    "benchmark": "Comparison rows that place the production model in the wider benchmark context.",
    "training_curves": "Training history exported from the model training run.",
    "learning_curves": "Sample-size or checkpoint learning-curve artifacts for the domain.",
    "cross_dataset": "Evaluation rows produced on datasets outside the primary split.",
    "confusion_matrix": "Static confusion-matrix image assets copied into the dashboard bundle.",
    "class_distribution": "Class and optional source-dataset distributions derived from explicit split inputs.",
    "samples": "Prediction examples for qualitative inspection.",
}
MISSING_SOURCE_REASONS = {
    "evaluation": "No primary_evaluation source was configured.",
    "benchmark": "No benchmark source was configured.",
    "training_curves": "No training_history source was configured.",
    "learning_curves": "No learning_curve source was configured.",
    "cross_dataset": "No cross_dataset source was configured.",
    "confusion_matrix": "No confusion_matrix source was configured.",
    "class_distribution": "No class_distribution source was configured.",
    "samples": "No prediction_samples source was configured.",
}
DEFAULT_SOURCE_REASONS = {
    "metadata": "Production model identity, runtime settings, and label mapping.",
    "runtime_config": "Runtime export config attached to the dashboard metadata bundle.",
    "experiment_config": "Experiment or evaluation config attached for traceability.",
    "primary_evaluation": "Primary validation/test metrics for the production model.",
    "benchmark": "Benchmark rows used to compare the production model against peers.",
    "training_history": "Training-history artifact used to build learning and loss curves.",
    "learning_curve": "Learning-curve source used for sample-size or checkpoint comparisons.",
    "cross_dataset": "Cross-dataset evaluation summary rows.",
    "class_distribution": "Explicit split artifacts used to derive class distributions.",
    "source_dataset_distribution": "Explicit source-distribution artifact for the dashboard bundle.",
    "prediction_samples": "Prediction samples used for qualitative dashboard review.",
    "confusion_matrix": "Confusion-matrix image copied into the dashboard bundle.",
}
SOURCE_ALIASES = {
    "learning_curves": "learning_curve",
    "samples": "prediction_samples",
    "primary-evaluation": "primary_evaluation",
    "training-history": "training_history",
    "learning-curve": "learning_curve",
    "cross-dataset": "cross_dataset",
    "class-distribution": "class_distribution",
    "source-dataset-distribution": "source_dataset_distribution",
    "prediction-samples": "prediction_samples",
    "confusion-matrix": "confusion_matrix",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


@dataclass(slots=True)
class DashboardSectionOutcome:
    id: str
    title: str
    status: str
    description: str
    files: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    reason: str | None = None

    def to_manifest_record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "description": self.description,
            "files": self.files,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.charts:
            payload["charts"] = self.charts
        return payload


@dataclass(slots=True)
class DashboardBuildResult:
    model_id: str
    model_dir: Path
    dashboard_dir: Path
    builder: str | None
    generated: bool
    skipped: bool = False
    manifest_path: Path | None = None
    generated_at: str | None = None
    notes: list[str] = field(default_factory=list)
    sections: list[DashboardSectionOutcome] = field(default_factory=list)


def build_configured_dashboards(
    model_root: Path,
    *,
    repo_root: Path | None = None,
    app_root: Path | None = None,
) -> list[DashboardBuildResult]:
    results: list[DashboardBuildResult] = []
    for config_path in sorted(model_root.rglob("model-config.yaml")):
        results.append(
            build_model_dashboard(
                config_path.parent,
                repo_root=repo_root,
                app_root=app_root,
            )
        )
    return results


def build_model_dashboard(
    model_dir: Path,
    *,
    repo_root: Path | None = None,
    app_root: Path | None = None,
) -> DashboardBuildResult:
    resolved_model_dir = model_dir.resolve()
    effective_repo_root = (repo_root or REPO_ROOT).resolve()
    effective_app_root = (app_root or APP_ROOT).resolve()
    config_path = resolved_model_dir / "model-config.yaml"
    raw_config = _load_yaml(config_path)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Model config at '{config_path}' must contain a mapping.")

    manifest = ModelManifest.from_yaml_dict(raw_config)
    dashboard_config = _as_dict(raw_config.get("dashboard"))
    builder_name = str(dashboard_config.get("builder") or "").strip() or None
    result = DashboardBuildResult(
        model_id=manifest.model_id,
        model_dir=resolved_model_dir,
        dashboard_dir=resolved_model_dir / "dashboard",
        builder=builder_name,
        generated=False,
    )

    if builder_name is None:
        result.skipped = True
        result.notes.append("No dashboard builder configured.")
        return result

    if builder_name != "generic-v1":
        result.skipped = True
        result.notes.append(f"Unsupported dashboard builder '{builder_name}'.")
        return result

    return _build_generic_v1(
        manifest=manifest,
        raw_config=raw_config,
        model_dir=resolved_model_dir,
        dashboard_config=dashboard_config,
        repo_root=effective_repo_root,
        app_root=effective_app_root,
    )


def _build_generic_v1(
    *,
    manifest: ModelManifest,
    raw_config: dict[str, Any],
    model_dir: Path,
    dashboard_config: dict[str, Any],
    repo_root: Path,
    app_root: Path,
) -> DashboardBuildResult:
    dashboard_dir = model_dir / "dashboard"
    generated_at = _utc_now_iso()
    notes = _dedupe_strings(_string_list(dashboard_config.get("notes")))
    selected_sources: list[dict[str, Any]] = []
    missing_sources: list[dict[str, Any]] = []
    scan_roots: set[Path] = {model_dir.resolve()}
    sources = _normalize_sources_map(_as_dict(dashboard_config.get("sources")))

    runtime_spec = _get_source_spec(
        sources,
        "runtime_config",
        "model_runtime_config",
        "export_config",
    )
    runtime_payload = _load_optional_mapping_source(
        category="runtime_config",
        raw_spec=runtime_spec,
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    metadata_payload = _build_model_metadata(
        raw_config=raw_config,
        model_dir=model_dir,
        repo_root=repo_root,
        runtime_payload=runtime_payload,
    )
    model_config_path = model_dir / "model-config.yaml"
    selected_sources.append(
        {
            "category": "metadata",
            "path": _rel(model_config_path, repo_root),
            "reason": DEFAULT_SOURCE_REASONS["metadata"],
        }
    )

    experiment_payload = _load_optional_mapping_source(
        category="experiment_config",
        raw_spec=_get_source_spec(sources, "experiment_config"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )

    metadata_files = _write_metadata_documents(
        dashboard_dir=dashboard_dir,
        metadata_payload=metadata_payload,
        experiment_payload=experiment_payload,
        repo_root=repo_root,
    )
    metadata_section = DashboardSectionOutcome(
        id="metadata",
        title=SECTION_TITLES["metadata"],
        status="available",
        description=SECTION_DESCRIPTIONS["metadata"],
        files=metadata_files,
    )

    evaluation_section, evaluation_payload = _build_primary_evaluation_section(
        manifest=manifest,
        dashboard_dir=dashboard_dir,
        raw_spec=_get_source_spec(sources, "primary_evaluation"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    benchmark_section, benchmark_rows, production_benchmark_name = _build_benchmark_section(
        manifest=manifest,
        dashboard_dir=dashboard_dir,
        raw_spec=_get_source_spec(sources, "benchmark"),
        production_model_name=_preferred_model_name(
            _get_source_spec(sources, "benchmark"),
            _get_source_spec(sources, "primary_evaluation"),
            evaluation_payload.get("model") if isinstance(evaluation_payload, dict) else None,
            manifest.display_name,
        ),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    training_section, training_payload = _build_training_history_section(
        dashboard_dir=dashboard_dir,
        raw_spec=_get_source_spec(sources, "training_history"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    learning_section, learning_rows = _build_learning_curve_section(
        dashboard_dir=dashboard_dir,
        raw_spec=_get_source_spec(sources, "learning_curve"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    cross_dataset_section, cross_dataset_rows = _build_cross_dataset_section(
        dashboard_dir=dashboard_dir,
        raw_spec=_get_source_spec(sources, "cross_dataset"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    class_distribution_section = _build_class_distribution_section(
        manifest=manifest,
        dashboard_dir=dashboard_dir,
        class_spec=_get_source_spec(sources, "class_distribution"),
        source_spec=_get_source_spec(sources, "source_dataset_distribution"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    samples_section = _build_prediction_samples_section(
        manifest=manifest,
        dashboard_dir=dashboard_dir,
        raw_spec=_get_source_spec(sources, "prediction_samples"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    confusion_section = _build_confusion_matrix_section(
        dashboard_dir=dashboard_dir,
        raw_spec=_get_source_spec(sources, "confusion_matrix"),
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )

    section_map = {
        "metadata": metadata_section,
        "evaluation": evaluation_section,
        "benchmark": benchmark_section,
        "training_curves": training_section,
        "learning_curves": learning_section,
        "cross_dataset": cross_dataset_section,
        "confusion_matrix": confusion_section,
        "class_distribution": class_distribution_section,
        "samples": samples_section,
    }

    section_status = {
        section_id: outcome.status for section_id, outcome in section_map.items()
    }
    source_audit_payload = _build_source_audit(
        scan_roots=scan_roots,
        repo_root=repo_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        notes=notes,
        generated_at=generated_at,
    )
    source_audit_path = dashboard_dir / "summary" / "source-audit.json"
    _write_json(source_audit_path, source_audit_payload)

    overview_payload = _build_overview(
        manifest=manifest,
        evaluation=evaluation_payload if isinstance(evaluation_payload, dict) else {},
        benchmark_rows=benchmark_rows,
        production_benchmark_name=production_benchmark_name,
        cross_dataset_rows=cross_dataset_rows,
        section_status=section_status,
        notes=notes,
        generated_at=generated_at,
    )
    overview_path = dashboard_dir / "summary" / "overview.json"
    _write_json(overview_path, overview_payload)

    summary_section = DashboardSectionOutcome(
        id="summary",
        title=SECTION_TITLES["summary"],
        status="available",
        description=SECTION_DESCRIPTIONS["summary"],
        files=[_rel(overview_path, repo_root), _rel(source_audit_path, repo_root)],
    )
    section_map["summary"] = summary_section

    ordered_sections = [section_map[section_id] for section_id in SECTION_ORDER]
    manifest_path = _write_manifest(
        dashboard_dir=dashboard_dir,
        repo_root=repo_root,
        generated_at=generated_at,
        manifest=manifest,
        sections=ordered_sections,
        selected_sources=_dedupe_source_items(selected_sources),
        notes=_dedupe_strings(notes),
    )

    return DashboardBuildResult(
        model_id=manifest.model_id,
        model_dir=model_dir,
        dashboard_dir=dashboard_dir,
        builder="generic-v1",
        generated=True,
        manifest_path=manifest_path,
        generated_at=generated_at,
        notes=_dedupe_strings(notes),
        sections=ordered_sections,
    )


def _build_primary_evaluation_section(
    *,
    manifest: ModelManifest,
    dashboard_dir: Path,
    raw_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> tuple[DashboardSectionOutcome, dict[str, Any]]:
    section_id = "evaluation"
    override = _section_override(section_id, raw_spec)
    if override is not None:
        return override, {}

    if raw_spec is None:
        return _missing_section(section_id), {}

    path, payload = _load_structured_source(
        category="primary_evaluation",
        raw_spec=raw_spec,
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    if path is None or payload is None:
        return _missing_section(section_id, reason=_missing_path_reason(raw_spec, section_id)), {}

    evaluation_payload: dict[str, Any]
    if isinstance(payload, dict) and isinstance(payload.get("splits"), dict):
        evaluation_payload = dict(payload)
        evaluation_payload.setdefault("model", _preferred_model_name(raw_spec, None, None, manifest.display_name))
        source_files = _string_list(evaluation_payload.get("source_files"))
        if not source_files:
            evaluation_payload["source_files"] = [_rel(path, repo_root)]
    elif isinstance(payload, dict) and _has_metric_splits(payload):
        splits = _collect_metric_splits(payload)
        if not splits:
            return _missing_section(
                section_id,
                reason="Configured primary_evaluation source did not contain val/test metrics.",
            ), {}
        evaluation_payload = {
            "model": _preferred_model_name(raw_spec, None, None, manifest.display_name),
            "source_files": [_rel(path, repo_root)],
            "splits": splits,
        }
        artifact_paths = {
            key: _normalize_slashes(value)
            for key, value in payload.items()
            if key not in {"val", "validation", "test"}
        }
        if artifact_paths:
            evaluation_payload["artifact_paths"] = artifact_paths
    else:
        rows = _parse_metric_rows_from_payload(
            payload=payload,
            source_path=path,
            item_spec=raw_spec,
            repo_root=repo_root,
        )
        model_name = _preferred_model_name(raw_spec, None, None, manifest.display_name)
        matching_rows = [
            row for row in rows if str(row.get("model", "")).strip() == model_name
        ]
        if not matching_rows:
            return _missing_section(
                section_id,
                reason=(
                    "Configured primary_evaluation source did not contain rows for "
                    f"'{model_name}'."
                ),
            ), {}
        evaluation_payload = {
            "model": model_name,
            "source_files": sorted(
                {
                    str(row.get("source_file"))
                    for row in matching_rows
                    if row.get("source_file")
                }
            ),
            "splits": {
                str(row.get("split")): {
                    metric: row.get(metric)
                    for metric in ("accuracy", "f1_macro", "f1_weighted", "loss")
                }
                for row in matching_rows
                if row.get("split")
            },
        }

    evaluation_path = dashboard_dir / "metrics" / "primary-evaluation.json"
    _write_json(evaluation_path, evaluation_payload)
    return (
        DashboardSectionOutcome(
            id=section_id,
            title=SECTION_TITLES[section_id],
            status="available",
            description=SECTION_DESCRIPTIONS[section_id],
            files=[_rel(evaluation_path, repo_root)],
        ),
        evaluation_payload,
    )


def _build_benchmark_section(
    *,
    manifest: ModelManifest,
    dashboard_dir: Path,
    raw_spec: dict[str, Any] | None,
    production_model_name: str | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> tuple[DashboardSectionOutcome, list[dict[str, Any]], str | None]:
    section_id = "benchmark"
    override = _section_override(section_id, raw_spec)
    if override is not None:
        return override, [], production_model_name

    if raw_spec is None:
        return _missing_section(section_id), [], production_model_name

    all_rows: list[dict[str, Any]] = []
    for item_spec in _iter_source_items(raw_spec):
        path, payload = _load_structured_source(
            category="benchmark",
            raw_spec=item_spec,
            model_dir=model_dir,
            repo_root=repo_root,
            app_root=app_root,
            selected_sources=selected_sources,
            missing_sources=missing_sources,
            scan_roots=scan_roots,
            notes=notes,
        )
        if path is None or payload is None:
            continue
        all_rows.extend(
            _parse_metric_rows_from_payload(
                payload=payload,
                source_path=path,
                item_spec=item_spec,
                repo_root=repo_root,
            )
        )

    if not all_rows:
        return _missing_section(section_id, reason=_missing_path_reason(raw_spec, section_id)), [], production_model_name

    effective_production_name = _resolve_matching_model_name(
        candidates=[
            production_model_name,
            manifest.display_name,
            manifest.model_id,
        ],
        rows=all_rows,
    )
    benchmark_test = _top_test_rows(all_rows, limit=_int_value(raw_spec.get("top_k"), default=10))
    for row in benchmark_test:
        row["is_production"] = (
            effective_production_name is not None
            and str(row.get("model", "")).strip() == effective_production_name
        )

    benchmark_path = dashboard_dir / "metrics" / "benchmark-test.json"
    _write_json(benchmark_path, benchmark_test)
    files = [_rel(benchmark_path, repo_root)]
    charts: list[str] = []

    benchmark_figure = _make_benchmark_figure(benchmark_test)
    if benchmark_figure is not None:
        figure_path = _write_figure(
            dashboard_dir=dashboard_dir,
            repo_root=repo_root,
            figure_id="benchmark-test-f1",
            figure=benchmark_figure,
        )
        files.append(figure_path)
        charts.append("benchmark-test-f1")

    return (
        DashboardSectionOutcome(
            id=section_id,
            title=SECTION_TITLES[section_id],
            status="available",
            description=SECTION_DESCRIPTIONS[section_id],
            files=files,
            charts=charts,
        ),
        all_rows,
        effective_production_name,
    )


def _build_training_history_section(
    *,
    dashboard_dir: Path,
    raw_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> tuple[DashboardSectionOutcome, dict[str, Any]]:
    section_id = "training_curves"
    override = _section_override(section_id, raw_spec)
    if override is not None:
        return override, {}

    if raw_spec is None:
        return _missing_section(section_id), {}

    path = _resolve_source_path(raw_spec.get("path"), model_dir=model_dir, repo_root=repo_root, app_root=app_root)
    if path is None:
        _record_missing_source(
            category="training_history",
            raw_path=raw_spec.get("path"),
            reason=_default_reason("training_history", raw_spec),
            missing_sources=missing_sources,
            repo_root=repo_root,
        )
        return _missing_section(section_id, reason=_missing_path_reason(raw_spec, section_id)), {}

    _record_selected_source(
        category="training_history",
        path=path,
        reason=_default_reason("training_history", raw_spec),
        selected_sources=selected_sources,
        scan_roots=scan_roots,
        repo_root=repo_root,
    )

    try:
        training_payload = _parse_training_history(
            path=path,
            raw_spec=raw_spec,
            repo_root=repo_root,
        )
    except Exception as exc:
        notes.append(f"training_history parsing failed for {_rel(path, repo_root)}: {exc}")
        return _missing_section(section_id, reason="Configured training_history source could not be parsed."), {}

    curves_path = dashboard_dir / "curves" / "training-history.json"
    _write_json(curves_path, training_payload)
    files = [_rel(curves_path, repo_root)]
    charts: list[str] = []

    for figure_id, figure in _make_training_history_figures(training_payload).items():
        figure_path = _write_figure(
            dashboard_dir=dashboard_dir,
            repo_root=repo_root,
            figure_id=figure_id,
            figure=figure,
        )
        files.append(figure_path)
        charts.append(figure_id)

    return (
        DashboardSectionOutcome(
            id=section_id,
            title=SECTION_TITLES[section_id],
            status="available",
            description=SECTION_DESCRIPTIONS[section_id],
            files=files,
            charts=charts,
        ),
        training_payload,
    )


def _build_learning_curve_section(
    *,
    dashboard_dir: Path,
    raw_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> tuple[DashboardSectionOutcome, list[dict[str, Any]]]:
    section_id = "learning_curves"
    override = _section_override(section_id, raw_spec)
    if override is not None:
        return override, []

    if raw_spec is None:
        return _missing_section(section_id), []

    path, payload = _load_structured_source(
        category="learning_curve",
        raw_spec=raw_spec,
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    if path is None or payload is None:
        return _missing_section(section_id, reason=_missing_path_reason(raw_spec, section_id)), []

    rows = _parse_learning_curve_rows(payload, path, repo_root)
    if not rows:
        return _missing_section(
            section_id,
            reason="Configured learning_curve source did not produce any rows.",
        ), []

    learning_curve_path = dashboard_dir / "curves" / "learning-curve.json"
    _write_json(learning_curve_path, rows)
    files = [_rel(learning_curve_path, repo_root)]
    charts: list[str] = []

    figure = _make_learning_curve_figure(rows)
    if figure is not None:
        figure_path = _write_figure(
            dashboard_dir=dashboard_dir,
            repo_root=repo_root,
            figure_id="learning-curve-f1",
            figure=figure,
        )
        files.append(figure_path)
        charts.append("learning-curve-f1")

    return (
        DashboardSectionOutcome(
            id=section_id,
            title=SECTION_TITLES[section_id],
            status="available",
            description=SECTION_DESCRIPTIONS[section_id],
            files=files,
            charts=charts,
        ),
        rows,
    )


def _build_cross_dataset_section(
    *,
    dashboard_dir: Path,
    raw_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> tuple[DashboardSectionOutcome, list[dict[str, Any]]]:
    section_id = "cross_dataset"
    override = _section_override(section_id, raw_spec)
    if override is not None:
        return override, []

    if raw_spec is None:
        return _missing_section(section_id), []

    all_rows: list[dict[str, Any]] = []
    for item_spec in _iter_source_items(raw_spec):
        path, payload = _load_structured_source(
            category="cross_dataset",
            raw_spec=item_spec,
            model_dir=model_dir,
            repo_root=repo_root,
            app_root=app_root,
            selected_sources=selected_sources,
            missing_sources=missing_sources,
            scan_roots=scan_roots,
            notes=notes,
        )
        if path is None or payload is None:
            continue
        all_rows.extend(_parse_cross_dataset_rows(payload, path, item_spec, repo_root))

    if not all_rows:
        return _missing_section(section_id, reason=_missing_path_reason(raw_spec, section_id)), []

    cross_dataset_path = dashboard_dir / "metrics" / "cross-dataset.json"
    _write_json(cross_dataset_path, all_rows)
    files = [_rel(cross_dataset_path, repo_root)]
    charts: list[str] = []

    figure = _make_cross_dataset_figure(all_rows)
    if figure is not None:
        figure_path = _write_figure(
            dashboard_dir=dashboard_dir,
            repo_root=repo_root,
            figure_id="cross-dataset-f1",
            figure=figure,
        )
        files.append(figure_path)
        charts.append("cross-dataset-f1")

    return (
        DashboardSectionOutcome(
            id=section_id,
            title=SECTION_TITLES[section_id],
            status="available",
            description=SECTION_DESCRIPTIONS[section_id],
            files=files,
            charts=charts,
        ),
        all_rows,
    )


def _build_class_distribution_section(
    *,
    manifest: ModelManifest,
    dashboard_dir: Path,
    class_spec: dict[str, Any] | None,
    source_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> DashboardSectionOutcome:
    section_id = "class_distribution"
    override = _section_override(section_id, class_spec or source_spec)
    if override is not None:
        return override

    class_distribution: dict[str, Any] | None = None
    source_distribution: dict[str, Any] | None = None
    files: list[str] = []
    charts: list[str] = []

    if class_spec is not None:
        class_distribution, derived_source_distribution = _build_distribution_payloads(
            manifest=manifest,
            category="class_distribution",
            raw_spec=class_spec,
            metric_key="label",
            model_dir=model_dir,
            repo_root=repo_root,
            app_root=app_root,
            selected_sources=selected_sources,
            missing_sources=missing_sources,
            scan_roots=scan_roots,
            notes=notes,
        )
        if derived_source_distribution is not None:
            source_distribution = derived_source_distribution

    if source_distribution is None and source_spec is not None:
        _, source_distribution = _build_distribution_payloads(
            manifest=manifest,
            category="source_dataset_distribution",
            raw_spec=source_spec,
            metric_key="source_dataset",
            model_dir=model_dir,
            repo_root=repo_root,
            app_root=app_root,
            selected_sources=selected_sources,
            missing_sources=missing_sources,
            scan_roots=scan_roots,
            notes=notes,
        )

    if class_distribution is not None:
        class_distribution_path = dashboard_dir / "distributions" / "class-distribution.json"
        _write_json(class_distribution_path, class_distribution)
        files.append(_rel(class_distribution_path, repo_root))
        figure = _make_distribution_figure(
            distribution=class_distribution,
            value_key="label",
            title="Class Distribution",
        )
        if figure is not None:
            figure_path = _write_figure(
                dashboard_dir=dashboard_dir,
                repo_root=repo_root,
                figure_id="class-distribution",
                figure=figure,
            )
            files.append(figure_path)
            charts.append("class-distribution")

    if source_distribution is not None:
        source_distribution_path = (
            dashboard_dir / "distributions" / "source-dataset-distribution.json"
        )
        _write_json(source_distribution_path, source_distribution)
        files.append(_rel(source_distribution_path, repo_root))
        figure = _make_distribution_figure(
            distribution=source_distribution,
            value_key="source_dataset",
            title="Source Dataset Distribution",
        )
        if figure is not None:
            figure_path = _write_figure(
                dashboard_dir=dashboard_dir,
                repo_root=repo_root,
                figure_id="source-dataset-distribution",
                figure=figure,
            )
            files.append(figure_path)
            charts.append("source-dataset-distribution")

    if not files:
        if class_spec is None and source_spec is None:
            return _missing_section(section_id)
        return _missing_section(
            section_id,
            reason=_missing_path_reason(class_spec or source_spec, section_id),
        )

    return DashboardSectionOutcome(
        id=section_id,
        title=SECTION_TITLES[section_id],
        status="available",
        description=SECTION_DESCRIPTIONS[section_id],
        files=files,
        charts=charts,
    )


def _build_prediction_samples_section(
    *,
    manifest: ModelManifest,
    dashboard_dir: Path,
    raw_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> DashboardSectionOutcome:
    section_id = "samples"
    override = _section_override(section_id, raw_spec)
    if override is not None:
        return override

    if raw_spec is None:
        return _missing_section(section_id)

    path, payload = _load_structured_source(
        category="prediction_samples",
        raw_spec=raw_spec,
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    if path is None or payload is None:
        return _missing_section(section_id, reason=_missing_path_reason(raw_spec, section_id))

    rows = _parse_prediction_samples(
        manifest=manifest,
        payload=payload,
        path=path,
        raw_spec=raw_spec,
    )
    if not rows:
        return _missing_section(
            section_id,
            reason="Configured prediction_samples source did not produce any rows.",
        )

    samples_path = dashboard_dir / "samples" / "prediction-samples.json"
    _write_json(samples_path, rows)
    return DashboardSectionOutcome(
        id=section_id,
        title=SECTION_TITLES[section_id],
        status="available",
        description=SECTION_DESCRIPTIONS[section_id],
        files=[_rel(samples_path, repo_root)],
    )


def _build_confusion_matrix_section(
    *,
    dashboard_dir: Path,
    raw_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> DashboardSectionOutcome:
    section_id = "confusion_matrix"
    override = _section_override(section_id, raw_spec)
    if override is not None:
        return override

    if raw_spec is None:
        return _missing_section(section_id)

    files: list[str] = []
    for item_spec in _iter_confusion_items(raw_spec):
        raw_path = item_spec.get("path")
        resolved = _resolve_source_path(raw_path, model_dir=model_dir, repo_root=repo_root, app_root=app_root)
        if resolved is None:
            _record_missing_source(
                category="confusion_matrix",
                raw_path=raw_path,
                reason=_default_reason("confusion_matrix", item_spec),
                missing_sources=missing_sources,
                repo_root=repo_root,
            )
            continue
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            notes.append(
                "confusion_matrix source "
                f"{_rel(resolved, repo_root)} was ignored because it is not an image."
            )
            continue
        _record_selected_source(
            category="confusion_matrix",
            path=resolved,
            reason=_default_reason("confusion_matrix", item_spec),
            selected_sources=selected_sources,
            scan_roots=scan_roots,
            repo_root=repo_root,
        )
        image_id = _sanitize_id(str(item_spec.get("id") or resolved.stem))
        destination = dashboard_dir / "confusion" / f"{image_id}{resolved.suffix.lower()}"
        files.append(_copy_file(resolved, destination, repo_root))

    if not files:
        return _missing_section(section_id, reason=_missing_path_reason(raw_spec, section_id))

    return DashboardSectionOutcome(
        id=section_id,
        title=SECTION_TITLES[section_id],
        status="image_only",
        description=SECTION_DESCRIPTIONS[section_id],
        files=files,
    )


def _build_model_metadata(
    *,
    raw_config: dict[str, Any],
    model_dir: Path,
    repo_root: Path,
    runtime_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = dict(raw_config)
    metadata.pop("dashboard", None)
    if runtime_payload is not None:
        metadata["runtime_config"] = runtime_payload
    metadata["prod_model_dir"] = _rel(model_dir, repo_root)
    metadata["artifacts_present"] = sorted(
        _rel(path, repo_root)
        for path in model_dir.iterdir()
        if path.name != "dashboard" and not path.name.startswith(".ipynb_checkpoints")
    )
    return metadata


def _write_metadata_documents(
    *,
    dashboard_dir: Path,
    metadata_payload: dict[str, Any],
    experiment_payload: dict[str, Any] | None,
    repo_root: Path,
) -> list[str]:
    files: list[str] = []
    model_path = dashboard_dir / "metadata" / "model.json"
    _write_json(model_path, metadata_payload)
    files.append(_rel(model_path, repo_root))
    if experiment_payload is not None:
        experiment_path = dashboard_dir / "metadata" / "experiment-config.json"
        _write_json(experiment_path, experiment_payload)
        files.append(_rel(experiment_path, repo_root))
    return files


def _build_distribution_payloads(
    *,
    manifest: ModelManifest,
    category: str,
    raw_spec: dict[str, Any],
    metric_key: str,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path_value = raw_spec.get("path")
    if path_value:
        path, payload = _load_structured_source(
            category=category,
            raw_spec=raw_spec,
            model_dir=model_dir,
            repo_root=repo_root,
            app_root=app_root,
            selected_sources=selected_sources,
            missing_sources=missing_sources,
            scan_roots=scan_roots,
            notes=notes,
        )
        if path is not None and isinstance(payload, dict):
            if metric_key == "label":
                class_distribution = dict(payload)
                source_distribution = class_distribution.pop("source_distribution", None)
                return (
                    class_distribution,
                    dict(source_distribution) if isinstance(source_distribution, dict) else None,
                )
            return None, dict(payload)

    split_specs = {
        split_name: raw_spec.get(split_name)
        for split_name in ("train", "val", "test")
        if raw_spec.get(split_name) is not None
    }
    if not split_specs:
        return None, None

    label_field = str(raw_spec.get("label_field") or "label")
    dataset_field = raw_spec.get("dataset_field")
    split_rows: dict[str, list[dict[str, Any]]] = {}

    for split_name, raw_path in split_specs.items():
        resolved = _resolve_source_path(raw_path, model_dir=model_dir, repo_root=repo_root, app_root=app_root)
        if resolved is None:
            _record_missing_source(
                category=category,
                raw_path=raw_path,
                reason=_default_reason(category, raw_spec),
                missing_sources=missing_sources,
                repo_root=repo_root,
            )
            continue
        _record_selected_source(
            category=category,
            path=resolved,
            reason=_default_reason(category, raw_spec),
            selected_sources=selected_sources,
            scan_roots=scan_roots,
            repo_root=repo_root,
        )
        try:
            split_rows[split_name] = _load_table_rows(resolved)
        except Exception as exc:
            notes.append(f"{category} parsing failed for {_rel(resolved, repo_root)}: {exc}")

    if not split_rows:
        return None, None

    class_distribution = _distribution_from_split_rows(
        split_rows=split_rows,
        label_field=label_field,
        value_key="label",
        value_transform=lambda value: _humanize_label_value(value, manifest.labels or []),
    )
    source_distribution = None
    if dataset_field:
        source_distribution = _distribution_from_split_rows(
            split_rows=split_rows,
            label_field=str(dataset_field),
            value_key="source_dataset",
            value_transform=lambda value: str(value),
        )
    return class_distribution, source_distribution


def _build_source_audit(
    *,
    scan_roots: set[Path],
    repo_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    notes: list[str],
    generated_at: str,
) -> dict[str, Any]:
    candidate_categories: dict[str, list[str]] = defaultdict(list)
    interesting_exts = {
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".parquet",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".log",
        ".txt",
    }
    seen_files: set[Path] = set()

    for root in sorted(scan_roots, key=lambda item: item.as_posix()):
        if not root.exists():
            continue
        scan_iterable = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in scan_iterable:
            candidate = path.resolve()
            if candidate in seen_files or not candidate.is_file():
                continue
            seen_files.add(candidate)
            if candidate.name == ".DS_Store":
                continue
            suffix = candidate.suffix.lower()
            if suffix not in interesting_exts:
                continue
            if suffix in IMAGE_SUFFIXES:
                category = "plot_images"
            elif suffix in {".log", ".txt"}:
                category = "logs"
            elif suffix == ".parquet":
                category = "structured_data"
            elif suffix == ".csv":
                category = "tabular_data"
            elif "metric" in candidate.name.lower() or "result" in candidate.name.lower():
                category = "metrics_and_configs"
            else:
                category = "other_structured_files"
            candidate_categories[category].append(_rel(candidate, repo_root))

    return {
        "generated_at": generated_at,
        "scanned_roots": [
            _rel(path, repo_root)
            for path in sorted(scan_roots, key=lambda item: item.as_posix())
        ],
        "artifact_counts": {
            key: len(value) for key, value in sorted(candidate_categories.items())
        },
        "candidate_artifacts_by_category": dict(sorted(candidate_categories.items())),
        "selected_dashboard_sources": _dedupe_source_items(selected_sources),
        "missing_dashboard_sources": _dedupe_source_items(missing_sources),
        "notes": _dedupe_strings(notes),
    }


def _build_overview(
    *,
    manifest: ModelManifest,
    evaluation: dict[str, Any],
    benchmark_rows: list[dict[str, Any]],
    production_benchmark_name: str | None,
    cross_dataset_rows: list[dict[str, Any]],
    section_status: dict[str, str],
    notes: list[str],
    generated_at: str,
) -> dict[str, Any]:
    ranked_rows = _sort_benchmark_rows(
        [
            row
            for row in benchmark_rows
            if str(row.get("split", "")).strip() == "test"
            and _clean_float(row.get("f1_macro")) is not None
        ]
    )
    rank_info = None
    if production_benchmark_name is not None:
        for index, row in enumerate(ranked_rows, start=1):
            if str(row.get("model", "")).strip() == production_benchmark_name:
                rank_info = {
                    "rank": index,
                    "out_of": len(ranked_rows),
                    "metric": "f1_macro",
                    "score": row.get("f1_macro"),
                }
                break

    cross_dataset_summary = None
    if cross_dataset_rows:
        valid_rows = [
            row
            for row in cross_dataset_rows
            if _clean_float(row.get("f1_macro")) is not None
        ]
        if valid_rows:
            dataset_count = len(valid_rows)
            cross_dataset_summary = {
                "datasets": dataset_count,
                "mean_accuracy": round(
                    sum(_clean_float(row.get("accuracy")) or 0.0 for row in valid_rows)
                    / dataset_count,
                    6,
                ),
                "mean_f1_macro": round(
                    sum(_clean_float(row.get("f1_macro")) or 0.0 for row in valid_rows)
                    / dataset_count,
                    6,
                ),
                "mean_f1_weighted": round(
                    sum(_clean_float(row.get("f1_weighted")) or 0.0 for row in valid_rows)
                    / dataset_count,
                    6,
                ),
            }

    evaluation_splits = _as_dict(evaluation.get("splits"))
    return {
        "generated_at": generated_at,
        "model": {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "domain": manifest.domain,
            "framework_type": manifest.framework.type,
            "framework_architecture": manifest.framework.architecture
            or manifest.framework.backbone
            or manifest.framework.base_model,
        },
        "evaluation_highlights": {
            "validation": evaluation_splits.get("validation") or evaluation_splits.get("val"),
            "test": evaluation_splits.get("test"),
            "benchmark_rank": rank_info,
        },
        "cross_dataset_highlights": cross_dataset_summary,
        "artifact_status": section_status,
        "notes": _dedupe_strings(notes),
    }


def _write_manifest(
    *,
    dashboard_dir: Path,
    repo_root: Path,
    generated_at: str,
    manifest: ModelManifest,
    sections: list[DashboardSectionOutcome],
    selected_sources: list[dict[str, Any]],
    notes: list[str],
) -> Path:
    manifest_path = dashboard_dir / "dashboard-manifest.json"
    payload = {
        "schema_version": "1.0.0",
        "generated_at": generated_at,
        "bundle_builder": "generic-v1",
        "dashboard_root": _rel(dashboard_dir, repo_root),
        "model": {
            "model_id": manifest.model_id,
            "display_name": manifest.display_name,
            "domain": manifest.domain,
            "description": manifest.description,
        },
        "entrypoints": {
            "overview": _rel(dashboard_dir / "summary" / "overview.json", repo_root),
            "source_audit": _rel(dashboard_dir / "summary" / "source-audit.json", repo_root),
        },
        "sections": [section.to_manifest_record() for section in sections],
        "selected_sources": selected_sources,
        "notes": notes,
    }
    _write_json(manifest_path, payload)
    return manifest_path


def _load_optional_mapping_source(
    *,
    category: str,
    raw_spec: dict[str, Any] | None,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> dict[str, Any] | None:
    if raw_spec is None:
        return None
    if _section_override("metadata", raw_spec) is not None:
        return None

    path, payload = _load_structured_source(
        category=category,
        raw_spec=raw_spec,
        model_dir=model_dir,
        repo_root=repo_root,
        app_root=app_root,
        selected_sources=selected_sources,
        missing_sources=missing_sources,
        scan_roots=scan_roots,
        notes=notes,
    )
    if path is None or payload is None:
        return None
    if not isinstance(payload, dict):
        notes.append(
            f"{category} at {_rel(path, repo_root)} was ignored because it did not contain a mapping."
        )
        return None
    return payload


def _load_structured_source(
    *,
    category: str,
    raw_spec: dict[str, Any],
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
    selected_sources: list[dict[str, Any]],
    missing_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    notes: list[str],
) -> tuple[Path | None, Any]:
    raw_path = raw_spec.get("path")
    resolved = _resolve_source_path(raw_path, model_dir=model_dir, repo_root=repo_root, app_root=app_root)
    if resolved is None:
        _record_missing_source(
            category=category,
            raw_path=raw_path,
            reason=_default_reason(category, raw_spec),
            missing_sources=missing_sources,
            repo_root=repo_root,
        )
        return None, None

    _record_selected_source(
        category=category,
        path=resolved,
        reason=_default_reason(category, raw_spec),
        selected_sources=selected_sources,
        scan_roots=scan_roots,
        repo_root=repo_root,
    )

    try:
        return resolved, _load_structured_file(resolved)
    except Exception as exc:
        notes.append(f"{category} parsing failed for {_rel(resolved, repo_root)}: {exc}")
        return resolved, None


def _parse_metric_rows_from_payload(
    *,
    payload: Any,
    source_path: Path,
    item_spec: dict[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and any(" | " in str(key) for key in payload):
        return _parse_results_json_payload(payload, source_path, repo_root)

    if isinstance(payload, dict) and _has_metric_splits(payload):
        return _rows_from_metric_summary(
            payload=payload,
            source_path=source_path,
            item_spec=item_spec,
            repo_root=repo_root,
        )

    if isinstance(payload, list):
        if _looks_like_results_csv_rows(payload):
            return _parse_pipe_results_csv_rows(payload, source_path, repo_root)
        return _normalize_metric_rows(payload, source_path, item_spec, repo_root)

    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return _normalize_metric_rows(
            payload["rows"],
            source_path,
            item_spec,
            repo_root,
        )

    return []


def _parse_cross_dataset_rows(
    payload: Any,
    source_path: Path,
    item_spec: dict[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]]
    if isinstance(payload, list):
        raw_rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        raw_rows = [row for row in payload["rows"] if isinstance(row, dict)]
    else:
        return rows

    filter_field = str(item_spec.get("filter_field") or "model")
    model_name = item_spec.get("model_name")
    dataset_field = str(item_spec.get("dataset_field") or "dataset")
    for row in raw_rows:
        if model_name is not None and str(row.get(filter_field, "")).strip() != str(model_name):
            continue
        normalized = dict(row)
        if dataset_field in normalized and dataset_field != "dataset":
            normalized["dataset"] = normalized.get(dataset_field)
        for metric in ("accuracy", "f1_macro", "f1_weighted", "loss"):
            if metric in normalized:
                normalized[metric] = _clean_float(normalized.get(metric))
        normalized.setdefault("source_file", _rel(source_path, repo_root))
        rows.append(normalized)
    return rows


def _parse_learning_curve_rows(
    payload: Any,
    source_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]]
    if isinstance(payload, list):
        raw_rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        raw_rows = [row for row in payload["rows"] if isinstance(row, dict)]
    else:
        return []

    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        model_name = str(item.get("model", "")).strip()
        if not model_name:
            continue
        split_key = str(item.get("split_key", "")).strip()
        train_size = _int_value(item.get("train_size"))
        if train_size is None and split_key:
            train_size = _parse_train_size(split_key)
        rows.append(
            {
                "model": model_name,
                "display_name": str(item.get("display_name") or _normalize_display_name(model_name)),
                "family": str(item.get("family") or _infer_family(model_name)),
                "split_key": split_key,
                "train_size": train_size,
                "f1_macro": _clean_float(item.get("f1_macro")),
                "source_file": str(item.get("source_file") or _rel(source_path, repo_root)),
            }
        )
    return rows


def _parse_training_history(
    *,
    path: Path,
    raw_spec: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    format_name = str(raw_spec.get("format") or "").strip().lower()
    if format_name in {"epoch_log", "epoch-log"} or path.suffix.lower() in {".log", ".txt"}:
        return _parse_epoch_log(path, repo_root)

    payload = _load_structured_file(path)
    if isinstance(payload, dict) and (
        isinstance(payload.get("train_events"), list) or isinstance(payload.get("points"), list)
    ):
        return dict(payload)
    if isinstance(payload, dict) and isinstance(payload.get("log_history"), list):
        return _parse_hf_log_history(payload["log_history"], path, repo_root)
    if isinstance(payload, list):
        return _parse_hf_log_history(payload, path, repo_root)
    raise ValueError("training_history format is not supported")


def _parse_prediction_samples(
    *,
    manifest: ModelManifest,
    payload: Any,
    path: Path,
    raw_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]]
    if isinstance(payload, list):
        raw_rows = [row for row in payload if isinstance(row, dict)]
        if raw_rows and "production_prediction" in raw_rows[0]:
            return raw_rows
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        raw_rows = [row for row in payload["rows"] if isinstance(row, dict)]
    else:
        return []

    text_field = str(raw_spec.get("text_field") or "text")
    example_id_field = str(raw_spec.get("example_id_field") or "example_id")
    production_spec = _as_dict(raw_spec.get("production_prediction"))
    reference_spec = _as_dict(raw_spec.get("reference_prediction"))
    if not production_spec:
        return []

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, start=1):
        text = row.get(text_field)
        if text is None:
            continue
        payload_row: dict[str, Any] = {
            "example_id": row.get(example_id_field, index),
            "text": str(text),
            "production_prediction": _build_prediction_payload(
                row=row,
                spec=production_spec,
                fallback_model=manifest.display_name,
            ),
        }
        reference_payload = _build_prediction_payload(
            row=row,
            spec=reference_spec,
            fallback_model="Reference",
        )
        if reference_payload is not None:
            payload_row["reference_prediction"] = reference_payload
        rows.append(payload_row)
    return rows


def _build_prediction_payload(
    *,
    row: Mapping[str, Any],
    spec: dict[str, Any],
    fallback_model: str,
) -> dict[str, Any] | None:
    if not spec:
        return None
    label_field = spec.get("label_field")
    if not label_field:
        return None
    label = row.get(str(label_field))
    if label is None:
        return None
    confidence = None
    confidence_field = spec.get("confidence_field")
    if confidence_field:
        confidence = _clean_float(row.get(str(confidence_field)))
    return {
        "model": str(spec.get("model") or fallback_model),
        "label": str(label),
        "confidence": confidence,
    }


def _parse_results_json_payload(
    payload: Mapping[str, Any],
    source_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, metrics in payload.items():
        if " | " not in str(key) or not isinstance(metrics, Mapping):
            continue
        model_name, split = str(key).rsplit(" | ", 1)
        rows.append(
            {
                "model": model_name,
                "display_name": _normalize_display_name(model_name),
                "family": _infer_family(model_name),
                "split": split,
                "accuracy": _clean_float(metrics.get("accuracy")),
                "f1_macro": _clean_float(metrics.get("f1_macro")),
                "f1_weighted": _clean_float(metrics.get("f1_weighted")),
                "loss": _clean_float(metrics.get("loss")),
                "source_file": _rel(source_path, repo_root),
            }
        )
    return rows


def _parse_pipe_results_csv_rows(
    rows: Sequence[Mapping[str, Any]],
    source_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    first_row = rows[0]
    first_col = next(iter(first_row.keys()), "")
    parsed_rows: list[dict[str, Any]] = []
    for item in rows:
        model_split = str(item.get(first_col, "")).strip()
        if not model_split or model_split.startswith("LC | "):
            continue
        if " | " not in model_split:
            continue
        model_name, split = model_split.rsplit(" | ", 1)
        parsed_rows.append(
            {
                "model": model_name,
                "display_name": _normalize_display_name(model_name),
                "family": _infer_family(model_name),
                "split": split,
                "accuracy": _clean_float(item.get("accuracy")),
                "f1_macro": _clean_float(item.get("f1_macro")),
                "f1_weighted": _clean_float(item.get("f1_weighted")),
                "loss": _clean_float(item.get("loss")),
                "source_file": _rel(source_path, repo_root),
            }
        )
    return parsed_rows


def _rows_from_metric_summary(
    *,
    payload: Mapping[str, Any],
    source_path: Path,
    item_spec: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    model_name = str(
        item_spec.get("model_name")
        or item_spec.get("display_name")
        or source_path.stem
    )
    display_name = str(item_spec.get("display_name") or _normalize_display_name(model_name))
    family = str(item_spec.get("family") or _infer_family(model_name))
    rows: list[dict[str, Any]] = []
    for split_name, metrics in _collect_metric_splits(payload).items():
        rows.append(
            {
                "model": model_name,
                "display_name": display_name,
                "family": family,
                "split": split_name,
                "accuracy": _clean_float(_as_dict(metrics).get("accuracy")),
                "f1_macro": _clean_float(_as_dict(metrics).get("f1_macro")),
                "f1_weighted": _clean_float(_as_dict(metrics).get("f1_weighted")),
                "loss": _clean_float(_as_dict(metrics).get("loss")),
                "source_file": _rel(source_path, repo_root),
            }
        )
    return rows


def _normalize_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    source_path: Path,
    item_spec: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        model_name = str(
            row.get("model")
            or item_spec.get("model_name")
            or item_spec.get("display_name")
            or ""
        ).strip()
        split = str(row.get("split") or "").strip()
        if not model_name or not split:
            continue
        normalized_rows.append(
            {
                "model": model_name,
                "display_name": str(row.get("display_name") or _normalize_display_name(model_name)),
                "family": str(row.get("family") or _infer_family(model_name)),
                "split": split,
                "accuracy": _clean_float(row.get("accuracy")),
                "f1_macro": _clean_float(row.get("f1_macro")),
                "f1_weighted": _clean_float(row.get("f1_weighted")),
                "loss": _clean_float(row.get("loss")),
                "source_file": str(row.get("source_file") or _rel(source_path, repo_root)),
            }
        )
    return normalized_rows


def _parse_hf_log_history(
    entries: Sequence[Mapping[str, Any]],
    source_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    train_events: list[dict[str, Any]] = []
    eval_events: list[dict[str, Any]] = []
    final_summary: dict[str, Any] | None = None

    for entry in entries:
        base = {
            "epoch": _clean_float(entry.get("epoch")),
            "step": _clean_float(entry.get("step")),
        }
        if any(key in entry for key in ("eval_loss", "eval_f1_macro", "eval_accuracy")):
            eval_events.append(
                {
                    **base,
                    "eval_loss": _clean_float(entry.get("eval_loss")),
                    "eval_f1_macro": _clean_float(entry.get("eval_f1_macro")),
                    "eval_accuracy": _clean_float(entry.get("eval_accuracy")),
                    "learning_rate": _clean_float(entry.get("learning_rate")),
                    "grad_norm": _clean_float(entry.get("grad_norm")),
                }
            )
        elif "loss" in entry:
            train_events.append(
                {
                    **base,
                    "loss": _clean_float(entry.get("loss")),
                    "learning_rate": _clean_float(entry.get("learning_rate")),
                    "grad_norm": _clean_float(entry.get("grad_norm")),
                }
            )
        elif "train_loss" in entry:
            final_summary = {
                "train_loss": _clean_float(entry.get("train_loss")),
                "train_runtime": _clean_float(entry.get("train_runtime")),
                "train_steps_per_second": _clean_float(entry.get("train_steps_per_second")),
                "train_samples_per_second": _clean_float(entry.get("train_samples_per_second")),
                "epoch": _clean_float(entry.get("epoch")),
                "step": _clean_float(entry.get("step")),
            }

    eval_with_metric = [
        item for item in eval_events if _clean_float(item.get("eval_f1_macro")) is not None
    ]
    best_eval = (
        max(eval_with_metric, key=lambda item: item.get("eval_f1_macro") or float("-inf"))
        if eval_with_metric
        else None
    )

    return {
        "source_file": _rel(source_path, repo_root),
        "x_axis": "step",
        "train_events": train_events,
        "eval_events": eval_events,
        "best_eval_event": best_eval,
        "final_summary": final_summary,
    }


def _parse_epoch_log(path: Path, repo_root: Path) -> dict[str, Any]:
    pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\s+\|\s+tr_loss=(?P<tr_loss>[0-9.]+)\s+\|\s+"
        r"va_loss=(?P<va_loss>[0-9.]+)\s+\|\s+val_f1=(?P<val_f1>[0-9.]+)\s+\|\s+(?P<seconds>[0-9.]+)s"
    )
    points: list[dict[str, Any]] = []
    notes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            points.append(
                {
                    "epoch": int(match.group("epoch")),
                    "total_epochs": int(match.group("epochs")),
                    "train_loss": float(match.group("tr_loss")),
                    "val_loss": float(match.group("va_loss")),
                    "val_f1_macro": float(match.group("val_f1")),
                    "epoch_seconds": float(match.group("seconds")),
                }
            )
        elif "Early stopping triggered" in line or "Restored best checkpoint" in line:
            notes.append(line.split("|", 2)[-1].strip())

    best_point = (
        max(points, key=lambda item: item.get("val_f1_macro") or float("-inf"))
        if points
        else None
    )
    return {
        "source_file": _rel(path, repo_root),
        "x_axis": "epoch",
        "points": points,
        "best_eval_event": best_point,
        "notes": notes,
    }


def _make_benchmark_figure(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    metric_key = "f1_macro"
    if not any(_clean_float(row.get(metric_key)) is not None for row in rows):
        metric_key = "accuracy"
    values = [_clean_float(row.get(metric_key)) for row in rows]
    if not any(value is not None for value in values):
        return None
    return _make_bar_figure(
        title="Benchmark Comparison",
        x_values=[str(row.get("display_name") or row.get("model") or "Model") for row in rows],
        y_values=[value or 0.0 for value in values],
        x_title="Model",
        y_title=_format_metric_label(metric_key),
        text_values=[
            f"{value:.3f}" if value is not None else "n/a" for value in values
        ],
    )


def _make_training_history_figures(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    figures: dict[str, dict[str, Any]] = {}
    train_events = [
        item for item in _record_list(payload.get("train_events")) if item
    ]
    eval_events = [item for item in _record_list(payload.get("eval_events")) if item]
    points = [item for item in _record_list(payload.get("points")) if item]
    x_axis = str(payload.get("x_axis") or "step")
    x_title = "Epoch" if x_axis == "epoch" else "Training Step"

    if train_events or eval_events:
        train_x = [item.get(x_axis) for item in train_events if item.get(x_axis) is not None]
        train_loss = [item.get("loss") for item in train_events if item.get(x_axis) is not None]
        eval_x = [item.get(x_axis) for item in eval_events if item.get(x_axis) is not None]
        eval_loss = [item.get("eval_loss") for item in eval_events if item.get(x_axis) is not None]
        series = []
        if train_x:
            series.append({"name": "Train Loss", "x": train_x, "y": train_loss})
        if eval_x:
            series.append({"name": "Eval Loss", "x": eval_x, "y": eval_loss})
        if series:
            figures["training-loss"] = _make_multi_line_figure(
                title="Training Loss",
                series=series,
                x_title=x_title,
                y_title="Loss",
            )

        metric_series = []
        eval_metric_x = [item.get(x_axis) for item in eval_events if item.get(x_axis) is not None]
        eval_f1 = [item.get("eval_f1_macro") for item in eval_events if item.get(x_axis) is not None]
        eval_accuracy = [item.get("eval_accuracy") for item in eval_events if item.get(x_axis) is not None]
        if eval_metric_x:
            metric_series.append({"name": "Eval F1 Macro", "x": eval_metric_x, "y": eval_f1})
            if any(value is not None for value in eval_accuracy):
                metric_series.append(
                    {"name": "Eval Accuracy", "x": eval_metric_x, "y": eval_accuracy}
                )
        if metric_series:
            figures["eval-metrics"] = _make_multi_line_figure(
                title="Evaluation Metrics",
                series=metric_series,
                x_title=x_title,
                y_title="Score",
            )

    elif points:
        epochs = [item.get("epoch") for item in points if item.get("epoch") is not None]
        if epochs:
            figures["training-loss"] = _make_multi_line_figure(
                title="Training Loss",
                series=[
                    {"name": "Train Loss", "x": epochs, "y": [item.get("train_loss") for item in points]},
                    {"name": "Validation Loss", "x": epochs, "y": [item.get("val_loss") for item in points]},
                ],
                x_title="Epoch",
                y_title="Loss",
            )
            metric_series = [
                {"name": "Validation F1 Macro", "x": epochs, "y": [item.get("val_f1_macro") for item in points]}
            ]
            val_accuracy = [item.get("val_accuracy") for item in points]
            if any(value is not None for value in val_accuracy):
                metric_series.append({"name": "Validation Accuracy", "x": epochs, "y": val_accuracy})
            figures["eval-metrics"] = _make_multi_line_figure(
                title="Evaluation Metrics",
                series=metric_series,
                x_title="Epoch",
                y_title="Score",
            )

    return figures


def _make_learning_curve_figure(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("display_name") or row.get("model") or "Model")].append(row)
    if not grouped:
        return None

    series = []
    for model_name, items in sorted(grouped.items()):
        ordered = sorted(
            items,
            key=lambda item: (
                item.get("train_size") is None,
                item.get("train_size") or 0,
                str(item.get("split_key") or ""),
            ),
        )
        series.append(
            {
                "name": model_name,
                "x": [item.get("train_size") for item in ordered],
                "y": [item.get("f1_macro") for item in ordered],
            }
        )
    return _make_multi_line_figure(
        title="Learning Curve",
        series=series,
        x_title="Train Size",
        y_title="F1 Macro",
    )


def _make_cross_dataset_figure(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    metric_key = "f1_macro"
    if not any(_clean_float(row.get(metric_key)) is not None for row in rows):
        metric_key = "accuracy"
    values = [_clean_float(row.get(metric_key)) for row in rows]
    if not any(value is not None for value in values):
        return None
    labels = [str(row.get("dataset") or row.get("model") or "Dataset") for row in rows]
    return _make_bar_figure(
        title="Cross Dataset Evaluation",
        x_values=labels,
        y_values=[value or 0.0 for value in values],
        x_title="Dataset",
        y_title=_format_metric_label(metric_key),
        text_values=[f"{value:.3f}" if value is not None else "n/a" for value in values],
        color="#b45309",
    )


def _make_distribution_figure(
    *,
    distribution: Mapping[str, Any],
    value_key: str,
    title: str,
) -> dict[str, Any] | None:
    overall_rows = [row for row in _record_list(distribution.get("overall")) if row]
    if not overall_rows:
        return None
    return _make_bar_figure(
        title=title,
        x_values=[str(row.get(value_key) or "Unknown") for row in overall_rows],
        y_values=[_int_value(row.get("count"), default=0) or 0 for row in overall_rows],
        x_title=_format_metric_label(value_key),
        y_title="Count",
        text_values=[str(_int_value(row.get("count"), default=0) or 0) for row in overall_rows],
    )


def _make_bar_figure(
    title: str,
    x_values: list[Any],
    y_values: list[Any],
    *,
    x_title: str,
    y_title: str,
    text_values: list[str] | None = None,
    color: str = "#0f766e",
    orientation: str = "v",
) -> dict[str, Any]:
    data = [
        {
            "type": "bar",
            "orientation": orientation,
            "x": x_values if orientation == "v" else y_values,
            "y": y_values if orientation == "v" else x_values,
            "marker": {"color": color},
            "text": text_values,
            "textposition": "auto",
        }
    ]
    layout = {
        "title": {"text": title},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "size": 12},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
        "xaxis": {"title": {"text": x_title}, "automargin": True},
        "yaxis": {"title": {"text": y_title}, "automargin": True},
    }
    return {"data": data, "layout": layout}


def _make_multi_line_figure(
    *,
    title: str,
    series: list[dict[str, Any]],
    x_title: str,
    y_title: str,
) -> dict[str, Any]:
    colors = ["#0f766e", "#b45309", "#1d4ed8", "#be123c", "#7c3aed", "#475569"]
    data = []
    for index, item in enumerate(series):
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": item["name"],
                "x": item["x"],
                "y": item["y"],
                "line": {"color": colors[index % len(colors)], "width": 2},
                "marker": {"size": 6},
            }
        )
    layout = {
        "title": {"text": title},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "size": 12},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
        "xaxis": {"title": {"text": x_title}, "automargin": True},
        "yaxis": {"title": {"text": y_title}, "automargin": True},
        "legend": {"orientation": "h", "y": 1.12},
    }
    return {"data": data, "layout": layout}


def _write_figure(
    *,
    dashboard_dir: Path,
    repo_root: Path,
    figure_id: str,
    figure: dict[str, Any],
) -> str:
    figure_path = dashboard_dir / "figures" / f"{figure_id}.plotly.json"
    _write_json(figure_path, figure)
    return _rel(figure_path, repo_root)


def _distribution_from_split_rows(
    *,
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    label_field: str,
    value_key: str,
    value_transform,
) -> dict[str, Any] | None:
    split_payloads: list[dict[str, Any]] = []
    overall_counter: Counter[str] = Counter()
    total_count = 0

    for split_name, rows in split_rows.items():
        counter: Counter[str] = Counter()
        for row in rows:
            if label_field not in row:
                continue
            normalized_label = value_transform(row[label_field])
            counter[normalized_label] += 1
        split_total = sum(counter.values())
        total_count += split_total
        overall_counter.update(counter)
        for label, count in sorted(counter.items()):
            split_payloads.append(
                {
                    "split": split_name,
                    value_key: label,
                    "count": count,
                    "share": round(count / split_total, 6) if split_total else 0.0,
                }
            )

    if total_count == 0:
        return None

    overall_payload = [
        {
            "split": "overall",
            value_key: label,
            "count": count,
            "share": round(count / total_count, 6),
        }
        for label, count in sorted(overall_counter.items())
    ]
    return {
        "overall": overall_payload,
        "splits": split_payloads,
    }


def _load_structured_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
        return rows
    if suffix == ".parquet":
        return _load_parquet_rows(path)
    raise ValueError(f"Unsupported source file type '{path.suffix}'.")


def _load_table_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_structured_file(path)
    if not isinstance(payload, list):
        raise ValueError("Tabular source must decode to a list of records.")
    return [row for row in payload if isinstance(row, dict)]


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet dashboard sources require the optional pandas dependency."
        ) from exc
    dataframe = pd.read_parquet(path)
    return [
        row
        for row in dataframe.to_dict(orient="records")
        if isinstance(row, dict)
    ]


def _resolve_source_path(
    raw_path: Any,
    *,
    model_dir: Path,
    repo_root: Path,
    app_root: Path,
) -> Path | None:
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([model_dir / path, app_root / path, repo_root / path])
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _copy_file(source: Path, destination: Path, repo_root: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _rel(destination, repo_root)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _missing_section(section_id: str, reason: str | None = None) -> DashboardSectionOutcome:
    return DashboardSectionOutcome(
        id=section_id,
        title=SECTION_TITLES[section_id],
        status="missing",
        description=SECTION_DESCRIPTIONS[section_id],
        reason=reason or MISSING_SOURCE_REASONS[section_id],
    )


def _section_override(
    section_id: str,
    raw_spec: dict[str, Any] | None,
) -> DashboardSectionOutcome | None:
    if not raw_spec:
        return None
    status = str(raw_spec.get("status") or "").strip()
    if status not in {"missing", "not_applicable"}:
        return None
    return DashboardSectionOutcome(
        id=section_id,
        title=SECTION_TITLES[section_id],
        status=status,
        description=str(raw_spec.get("description") or SECTION_DESCRIPTIONS[section_id]),
        reason=str(raw_spec.get("reason") or "") or None,
    )


def _get_source_spec(sources: Mapping[str, dict[str, Any]], *aliases: str) -> dict[str, Any] | None:
    for alias in aliases:
        normalized = _normalize_source_key(alias)
        if normalized in sources:
            spec = dict(sources[normalized])
            spec.setdefault("section_id", _section_id_for_source(normalized))
            return spec
    return None


def _normalize_sources_map(raw_sources: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in raw_sources.items():
        source_key = _normalize_source_key(key)
        normalized[source_key] = _normalize_source_spec(value)
        normalized[source_key].setdefault("section_id", _section_id_for_source(source_key))
    return normalized


def _normalize_source_key(key: str) -> str:
    base = key.strip().lower().replace("-", "_")
    return SOURCE_ALIASES.get(base, base)


def _section_id_for_source(source_key: str) -> str:
    return {
        "primary_evaluation": "evaluation",
        "benchmark": "benchmark",
        "training_history": "training_curves",
        "learning_curve": "learning_curves",
        "cross_dataset": "cross_dataset",
        "class_distribution": "class_distribution",
        "source_dataset_distribution": "class_distribution",
        "prediction_samples": "samples",
        "confusion_matrix": "confusion_matrix",
        "experiment_config": "metadata",
        "runtime_config": "metadata",
    }.get(source_key, source_key)


def _normalize_source_spec(raw_value: Any) -> dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, str):
        return {"path": raw_value}
    if isinstance(raw_value, list):
        return {
            "items": [
                _normalize_source_spec(item) if not isinstance(item, dict) else dict(item)
                for item in raw_value
            ]
        }
    if isinstance(raw_value, dict):
        return dict(raw_value)
    return {"path": str(raw_value)}


def _iter_source_items(raw_spec: dict[str, Any]) -> list[dict[str, Any]]:
    items = raw_spec.get("items")
    if isinstance(items, list):
        normalized_items = []
        for item in items:
            item_spec = _normalize_source_spec(item)
            for key, value in raw_spec.items():
                if key not in item_spec and key not in {"items", "path", "paths"}:
                    item_spec[key] = value
            normalized_items.append(item_spec)
        return normalized_items
    return [raw_spec]


def _iter_confusion_items(raw_spec: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw_spec.get("items"), list):
        return _iter_source_items(raw_spec)
    if isinstance(raw_spec.get("paths"), list):
        return [
            {
                **{key: value for key, value in raw_spec.items() if key != "paths"},
                "path": path,
            }
            for path in raw_spec["paths"]
        ]
    return [raw_spec]


def _record_selected_source(
    *,
    category: str,
    path: Path,
    reason: str,
    selected_sources: list[dict[str, Any]],
    scan_roots: set[Path],
    repo_root: Path,
) -> None:
    selected_sources.append(
        {
            "category": category,
            "path": _rel(path, repo_root),
            "reason": reason,
        }
    )
    scan_roots.add(path if path.is_dir() else path.parent)


def _record_missing_source(
    *,
    category: str,
    raw_path: Any,
    reason: str,
    missing_sources: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    if raw_path is None:
        return
    path_string = str(raw_path)
    maybe_path = Path(path_string)
    missing_sources.append(
        {
            "category": category,
            "path": _rel(maybe_path, repo_root) if maybe_path.is_absolute() else path_string,
            "reason": reason,
        }
    )


def _default_reason(category: str, raw_spec: Mapping[str, Any]) -> str:
    custom_reason = raw_spec.get("reason")
    if custom_reason:
        return str(custom_reason)
    return DEFAULT_SOURCE_REASONS.get(category, "Dashboard source configured for this section.")


def _missing_path_reason(raw_spec: dict[str, Any], section_id: str) -> str:
    if raw_spec.get("reason"):
        return str(raw_spec["reason"])
    if raw_spec.get("path") or raw_spec.get("paths") or raw_spec.get("items"):
        return "One or more configured dashboard sources for this section were missing or unreadable."
    return MISSING_SOURCE_REASONS[section_id]


def _preferred_model_name(
    benchmark_spec: Mapping[str, Any] | None,
    evaluation_spec: Mapping[str, Any] | None,
    evaluation_model: str | None,
    fallback: str,
) -> str:
    for candidate in (
        benchmark_spec.get("production_model_name") if benchmark_spec else None,
        benchmark_spec.get("model_name") if benchmark_spec else None,
        evaluation_spec.get("model_name") if evaluation_spec else None,
        evaluation_model,
        fallback,
    ):
        if candidate:
            return str(candidate)
    return fallback


def _resolve_matching_model_name(
    *,
    candidates: Iterable[str | None],
    rows: Sequence[Mapping[str, Any]],
) -> str | None:
    available_names = {str(row.get("model", "")).strip() for row in rows}
    for candidate in candidates:
        if candidate and str(candidate).strip() in available_names:
            return str(candidate).strip()
    return None


def _sort_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[float, float]:
        return (
            _clean_float(row.get("f1_macro")) or float("-inf"),
            _clean_float(row.get("accuracy")) or float("-inf"),
        )

    return sorted(rows, key=sort_key, reverse=True)


def _top_test_rows(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    test_rows = [
        row
        for row in rows
        if str(row.get("split", "")).strip() == "test"
    ]
    return _sort_benchmark_rows(test_rows)[:limit]


def _has_metric_splits(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload and isinstance(payload.get(key), Mapping)
        for key in ("validation", "val", "test")
    )


def _collect_metric_splits(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    splits: dict[str, dict[str, Any]] = {}
    for source_key, target_key in (("validation", "validation"), ("val", "val"), ("test", "test")):
        metrics = _as_dict(payload.get(source_key))
        if metrics:
            splits[target_key] = {
                key: _clean_float(value)
                for key, value in metrics.items()
            }
    return splits


def _looks_like_results_csv_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    first_col = next(iter(rows[0].keys()), "")
    return bool(first_col) and any(" | " in str(row.get(first_col, "")) for row in rows)


def _record_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            ordered.append(stripped)
    return ordered


def _dedupe_source_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for item in items:
        category = str(item.get("category") or "")
        path = str(item.get("path") or "")
        reason = str(item.get("reason")) if item.get("reason") is not None else None
        key = (category, path, reason)
        if not category or not path or key in seen:
            continue
        seen.add(key)
        ordered.append({"category": category, "path": path, "reason": reason})
    return ordered


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "asset"


def _infer_family(model_name: str) -> str:
    normalized = model_name.lower()
    if any(token in normalized for token in ("deberta", "distilbert", "roberta", "bert")):
        return "transformer"
    if "glove" in normalized or "bilstm" in normalized or "cnn" in normalized or "mlp" in normalized:
        return "deep"
    if "tf-idf" in normalized or "majority" in normalized or "logreg" in normalized or "svc" in normalized:
        return "classical"
    return "unknown"


def _normalize_display_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.replace("_", " ")).strip()


def _parse_train_size(token: str) -> int | None:
    normalized = token.lower()
    if normalized == "train_pool":
        return 31305
    match = re.search(r"(\d+)(k)?", normalized)
    if not match:
        return None
    value = int(match.group(1))
    if match.group(2):
        value *= 1000
    return value


def _humanize_label_value(value: Any, labels: Sequence[LabelClass]) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            mapped = _label_for_id(int(stripped), labels)
            if mapped is not None:
                return mapped
        mapped_name = _label_for_name(stripped, labels)
        return mapped_name or stripped
    if isinstance(value, int):
        mapped = _label_for_id(value, labels)
        return mapped or str(value)
    return str(value)


def _label_for_id(label_id: int, labels: Sequence[LabelClass]) -> str | None:
    for label in labels:
        if label.id == label_id:
            return label.effective_name
    return None


def _label_for_name(name: str, labels: Sequence[LabelClass]) -> str | None:
    normalized = name.strip().lower()
    for label in labels:
        if label.name.lower() == normalized or (label.display_name or "").lower() == normalized:
            return label.effective_name
    return None


def _format_metric_label(metric_key: str) -> str:
    return (
        metric_key.replace("_", " ")
        .replace("f1", "F1")
        .replace("source dataset", "Source Dataset")
        .title()
    )


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    string_value = str(value).strip()
    if not string_value or string_value.lower() in {"nan", "none", "null", "n/a"}:
        return None
    try:
        numeric = float(string_value)
    except ValueError:
        return None
    return None if math.isnan(numeric) else numeric


def _int_value(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    string_value = str(value).strip()
    if not string_value:
        return default
    try:
        return int(float(string_value))
    except ValueError:
        return default


def _normalize_slashes(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\\", "/")
    return value

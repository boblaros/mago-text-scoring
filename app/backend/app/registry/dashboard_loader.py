from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from app.core.settings import APP_ROOT
from app.registry.contracts import RegisteredModel
from app.schemas.models import (
    DashboardFigure,
    DashboardImageAsset,
    DashboardManifestSummary,
    DashboardSectionSummary,
    DashboardSourceItem,
    ModelDashboardResponse,
)


REPO_ROOT = APP_ROOT.parent
AVAILABLE_SECTION_STATUSES = {"available", "image_only", "not_applicable"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
SECTION_ID_BY_TOP_LEVEL_DIR = {
    "summary": "summary",
    "metadata": "metadata",
    "distributions": "class_distribution",
    "confusion": "confusion_matrix",
    "samples": "samples",
}
SECTION_ID_BY_METRIC_FILE = {
    "primary-evaluation": "evaluation",
    "benchmark": "benchmark",
    "cross-dataset": "cross_dataset",
}
SECTION_ID_BY_CURVE_FILE = {
    "training-history": "training_curves",
    "learning-curve": "learning_curves",
}
SECTION_ID_BY_FIGURE_TOKEN = (
    ("class-distribution", "class_distribution"),
    ("source-dataset-distribution", "class_distribution"),
    ("benchmark", "benchmark"),
    ("cross-dataset", "cross_dataset"),
    ("learning-curve", "learning_curves"),
    ("training-loss", "training_curves"),
    ("eval-metrics", "training_curves"),
)
SECTION_STATUS_WITH_DISCOVERABLE_ASSETS = {"missing", "image_only", "available"}
DEFAULT_DISTRIBUTION_FIGURE_COLORS = {
    "class-distribution": "#1d4ed8",
    "source-dataset-distribution": "#475569",
}


def summarize_dashboard(model: RegisteredModel) -> dict[str, Any]:
    manifest = _load_normalized_dashboard_manifest(model.model_dir)
    if manifest is None:
        return {
            "dashboard_status": "missing",
            "dashboard_sections_available": 0,
            "dashboard_sections_total": 0,
            "dashboard_generated_at": None,
        }

    sections = manifest.get("sections", [])
    available_count = sum(
        1 for section in sections if section.get("status") in AVAILABLE_SECTION_STATUSES
    )
    total = len(sections)
    if available_count == 0:
        status = "missing"
    elif available_count == total:
        status = "available"
    else:
        status = "partial"

    return {
        "dashboard_status": status,
        "dashboard_sections_available": available_count,
        "dashboard_sections_total": total,
        "dashboard_generated_at": manifest.get("generated_at"),
    }


def load_dashboard_manifest(model_dir: Path) -> dict[str, Any] | None:
    manifest_path = model_dir / "dashboard" / "dashboard-manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_model_dashboard(
    model: RegisteredModel,
    asset_url_builder: Callable[[str], str],
) -> ModelDashboardResponse:
    manifest_raw = _load_normalized_dashboard_manifest(model.model_dir)
    if manifest_raw is None:
        return ModelDashboardResponse(
            model_id=model.manifest.model_id,
            available=False,
        )

    dashboard_dir = model.model_dir / "dashboard"
    sections = [
        DashboardSectionSummary(
            id=str(section.get("id")),
            title=str(section.get("title")),
            status=str(section.get("status")),
            description=section.get("description"),
            reason=section.get("reason"),
            files=list(section.get("files", [])),
            charts=list(section.get("charts", [])),
        )
        for section in manifest_raw.get("sections", [])
    ]

    figures: list[DashboardFigure] = []
    images: list[DashboardImageAsset] = []
    documents: dict[str, Any] = {}

    for section in sections:
        for file_ref in section.files:
            resolved = _resolve_dashboard_file(dashboard_dir, file_ref)
            if resolved is None:
                continue
            relative_path = _relative_to_dashboard(dashboard_dir, resolved)
            if resolved.suffix == ".json" and resolved.name.endswith(".plotly.json"):
                figure = _load_json(resolved)
                if figure is None:
                    continue
                figures.append(
                    DashboardFigure(
                        id=resolved.name.replace(".plotly.json", ""),
                        path=relative_path,
                        title=_extract_figure_title(figure),
                        section_id=section.id,
                        figure=figure,
                    )
                )
                continue
            if resolved.suffix.lower() in IMAGE_SUFFIXES:
                images.append(
                    DashboardImageAsset(
                        title=_humanize_filename(resolved.stem),
                        path=relative_path,
                        url=asset_url_builder(relative_path),
                        section_id=section.id,
                    )
                )
                continue
            if resolved.suffix == ".json" and relative_path != "summary/source-audit.json":
                document = _load_json(resolved)
                if document is not None:
                    documents[relative_path] = document

    synthetic_distribution_figures = _build_distribution_figures_from_documents(
        documents=documents,
        existing_figures=figures,
    )
    if synthetic_distribution_figures:
        figures.extend(synthetic_distribution_figures)
        for section in sections:
            if section.id != "class_distribution":
                continue
            section.charts = _dedupe_strings(
                [*section.charts, *[figure.id for figure in synthetic_distribution_figures]],
            )
            if section.status == "missing":
                section.status = "available"
                section.reason = None
            break

    overview = _load_entrypoint_json(manifest_raw, dashboard_dir, "overview")
    source_audit = _load_source_audit_summary(manifest_raw, dashboard_dir)

    return ModelDashboardResponse(
        model_id=model.manifest.model_id,
        available=True,
        manifest=DashboardManifestSummary(
            schema_version=str(manifest_raw.get("schema_version", "1.0.0")),
            generated_at=manifest_raw.get("generated_at"),
            dashboard_root=str(manifest_raw.get("dashboard_root", "")),
            model=dict(manifest_raw.get("model", {})),
            entrypoints={
                key: str(value) for key, value in dict(manifest_raw.get("entrypoints", {})).items()
            },
            sections=sections,
            selected_sources=[
                DashboardSourceItem(
                    category=str(item.get("category")),
                    path=str(item.get("path")),
                    reason=item.get("reason"),
                )
                for item in manifest_raw.get("selected_sources", [])
            ],
            notes=[str(item) for item in manifest_raw.get("notes", [])],
        ),
        overview=overview,
        source_audit=source_audit,
        documents=documents,
        figures=figures,
        images=images,
    )


def _load_entrypoint_json(
    manifest_raw: dict[str, Any],
    dashboard_dir: Path,
    key: str,
) -> dict[str, Any] | None:
    entrypoints = manifest_raw.get("entrypoints", {})
    file_ref = entrypoints.get(key)
    if not file_ref:
        return None
    resolved = _resolve_dashboard_file(dashboard_dir, str(file_ref))
    if resolved is None:
        return None
    document = _load_json(resolved)
    return document if isinstance(document, dict) else None


def _load_source_audit_summary(
    manifest_raw: dict[str, Any],
    dashboard_dir: Path,
) -> dict[str, Any] | None:
    entrypoints = manifest_raw.get("entrypoints", {})
    file_ref = entrypoints.get("source_audit")
    if not file_ref:
        return None
    resolved = _resolve_dashboard_file(dashboard_dir, str(file_ref))
    if resolved is None:
        return None
    document = _load_json(resolved)
    if not isinstance(document, dict):
        return None
    return {
        "generated_at": document.get("generated_at"),
        "scanned_roots": document.get("scanned_roots", []),
        "artifact_counts": document.get("artifact_counts", {}),
    }


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_normalized_dashboard_manifest(model_dir: Path) -> dict[str, Any] | None:
    manifest = load_dashboard_manifest(model_dir)
    if manifest is None:
        return None
    return _normalize_dashboard_manifest(
        manifest_raw=manifest,
        dashboard_dir=model_dir / "dashboard",
    )


def _normalize_dashboard_manifest(
    *,
    manifest_raw: Mapping[str, Any],
    dashboard_dir: Path,
) -> dict[str, Any]:
    manifest = dict(manifest_raw)
    sections = manifest.get("sections", [])
    if not isinstance(sections, list):
        manifest["sections"] = []
        return manifest

    discovered_assets = _discover_section_assets(dashboard_dir)
    normalized_sections: list[dict[str, Any]] = []

    for raw_section in sections:
        if not isinstance(raw_section, dict):
            continue

        section = dict(raw_section)
        section_id = str(section.get("id", ""))
        existing_files = _normalize_section_file_refs(
            dashboard_dir,
            section.get("files", []),
        )
        existing_charts = _string_list(section.get("charts", []))
        discovered = discovered_assets.get(section_id, {"files": [], "charts": []})
        files = _dedupe_strings([*existing_files, *discovered["files"]])
        charts = _dedupe_strings([*existing_charts, *discovered["charts"]])
        current_status = str(section.get("status", "missing"))
        normalized_status = _normalize_section_status(
            current_status=current_status,
            file_refs=files,
            dashboard_dir=dashboard_dir,
        )

        section["files"] = files
        section["charts"] = charts
        section["status"] = normalized_status
        if normalized_status in SECTION_STATUS_WITH_DISCOVERABLE_ASSETS and normalized_status != "missing":
            section["reason"] = None

        normalized_sections.append(section)

    manifest["sections"] = normalized_sections
    return manifest


def _discover_section_assets(dashboard_dir: Path) -> dict[str, dict[str, list[str]]]:
    discovered: dict[str, dict[str, list[str]]] = {}
    if not dashboard_dir.exists():
        return discovered

    for path in sorted(dashboard_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = _relative_to_dashboard(dashboard_dir, path)
        section_id = _infer_section_id_from_asset(relative_path)
        if section_id is None:
            continue
        bucket = discovered.setdefault(section_id, {"files": [], "charts": []})
        bucket["files"].append(relative_path)
        if path.suffix == ".json" and path.name.endswith(".plotly.json"):
            bucket["charts"].append(path.name.replace(".plotly.json", ""))

    for bucket in discovered.values():
        bucket["files"] = _dedupe_strings(bucket["files"])
        bucket["charts"] = _dedupe_strings(bucket["charts"])

    return discovered


def _infer_section_id_from_asset(relative_path: str) -> str | None:
    parts = Path(relative_path).parts
    if not parts:
        return None

    top_level = parts[0]
    if top_level in SECTION_ID_BY_TOP_LEVEL_DIR:
        return SECTION_ID_BY_TOP_LEVEL_DIR[top_level]

    filename = parts[-1]
    stem = filename.replace(".plotly.json", "")

    if top_level == "metrics":
        for token, section_id in SECTION_ID_BY_METRIC_FILE.items():
            if token in stem:
                return section_id
        return None

    if top_level == "curves":
        for token, section_id in SECTION_ID_BY_CURVE_FILE.items():
            if token in stem:
                return section_id
        return None

    if top_level == "figures":
        for token, section_id in SECTION_ID_BY_FIGURE_TOKEN:
            if token in stem:
                return section_id
        if "distribution" in stem:
            return "class_distribution"
        return None

    return None


def _normalize_section_file_refs(dashboard_dir: Path, file_refs: Any) -> list[str]:
    normalized: list[str] = []
    for file_ref in _string_list(file_refs):
        resolved = _resolve_dashboard_file(dashboard_dir, file_ref)
        if resolved is not None:
            normalized.append(_relative_to_dashboard(dashboard_dir, resolved))
        else:
            normalized.append(file_ref)
    return _dedupe_strings(normalized)


def _normalize_section_status(
    *,
    current_status: str,
    file_refs: list[str],
    dashboard_dir: Path,
) -> str:
    resolved_files = [
        resolved
        for file_ref in file_refs
        if (resolved := _resolve_dashboard_file(dashboard_dir, file_ref)) is not None
    ]
    if not resolved_files:
        return current_status

    has_non_image_asset = any(path.suffix.lower() not in IMAGE_SUFFIXES for path in resolved_files)
    if has_non_image_asset:
        return "available"

    has_image_asset = any(path.suffix.lower() in IMAGE_SUFFIXES for path in resolved_files)
    if has_image_asset:
        return "image_only"

    return current_status


def _build_distribution_figures_from_documents(
    *,
    documents: Mapping[str, Any],
    existing_figures: list[DashboardFigure],
) -> list[DashboardFigure]:
    existing_ids = {
        figure.id
        for figure in existing_figures
        if figure.section_id == "class_distribution"
    }
    figures: list[DashboardFigure] = []

    for path, document in sorted(documents.items()):
        if not path.startswith("distributions/"):
            continue
        if not isinstance(document, dict):
            continue

        figure_id = Path(path).stem
        if figure_id in existing_ids:
            continue

        figure = _make_distribution_figure(
            distribution=document,
            figure_id=figure_id,
        )
        if figure is None:
            continue

        figures.append(
            DashboardFigure(
                id=figure_id,
                path=path,
                title=_extract_figure_title(figure),
                section_id="class_distribution",
                figure=figure,
            )
        )
        existing_ids.add(figure_id)

    return figures


def _make_distribution_figure(
    *,
    distribution: Mapping[str, Any],
    figure_id: str,
) -> dict[str, Any] | None:
    rows = _record_rows(distribution.get("overall"))
    value_key = _infer_distribution_value_key(rows)

    if not rows or value_key is None:
        split_rows = _record_rows(distribution.get("splits"))
        value_key = _infer_distribution_value_key(split_rows)
        if value_key is None:
            return None
        rows = _aggregate_distribution_rows(split_rows, value_key)
        if not rows:
            return None

    x_values = [str(row.get(value_key) or "Unknown") for row in rows]
    y_values = [_int_value(row.get("count")) for row in rows]
    if not any(value > 0 for value in y_values):
        return None

    return {
        "data": [
            {
                "type": "bar",
                "orientation": "v",
                "x": x_values,
                "y": y_values,
                "marker": {"color": _distribution_figure_color(figure_id, value_key)},
                "text": [str(value) for value in y_values],
                "textposition": "auto",
            }
        ],
        "layout": {
            "title": {"text": _distribution_figure_title(figure_id)},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "font": {"family": "Arial, sans-serif", "size": 12},
            "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
            "xaxis": {"title": {"text": _humanize_filename(value_key)}, "automargin": True},
            "yaxis": {"title": {"text": "Count"}, "automargin": True},
        },
    }


def _record_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _infer_distribution_value_key(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        for key in row:
            if key not in {"split", "count", "share"}:
                return str(key)
    return None


def _aggregate_distribution_rows(
    rows: list[dict[str, Any]],
    value_key: str,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get(value_key) or "Unknown")
        counts[label] = counts.get(label, 0) + _int_value(row.get("count"))
    return [
        {"split": "overall", value_key: label, "count": count}
        for label, count in sorted(counts.items())
    ]


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def _distribution_figure_color(figure_id: str, value_key: str) -> str:
    if figure_id in DEFAULT_DISTRIBUTION_FIGURE_COLORS:
        return DEFAULT_DISTRIBUTION_FIGURE_COLORS[figure_id]
    if value_key == "source_dataset":
        return "#475569"
    if value_key == "label":
        return "#1d4ed8"
    return "#0f766e"


def _distribution_figure_title(figure_id: str) -> str:
    return _humanize_filename(figure_id)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _dedupe_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _resolve_dashboard_file(dashboard_dir: Path, file_ref: str) -> Path | None:
    candidate = Path(file_ref)
    options = [
        dashboard_dir / candidate,
        REPO_ROOT / candidate,
        APP_ROOT / candidate,
    ]
    for option in options:
        if option.exists():
            return option
    if candidate.name:
        for match in sorted(dashboard_dir.rglob(candidate.name)):
            return match
    return None


def _relative_to_dashboard(dashboard_dir: Path, resolved: Path) -> str:
    try:
        return resolved.resolve().relative_to(dashboard_dir.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _extract_figure_title(figure: dict[str, Any]) -> str | None:
    layout = figure.get("layout", {})
    title = layout.get("title")
    if isinstance(title, dict):
        text = title.get("text")
        return str(text) if text else None
    if isinstance(title, str):
        return title
    return None


def _humanize_filename(value: str) -> str:
    return value.replace("-", " ").replace("_", " ").strip().title()

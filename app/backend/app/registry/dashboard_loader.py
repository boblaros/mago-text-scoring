from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

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


def summarize_dashboard(model: RegisteredModel) -> dict[str, Any]:
    manifest = load_dashboard_manifest(model.model_dir)
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
    manifest_raw = load_dashboard_manifest(model.model_dir)
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
        return resolved.relative_to(dashboard_dir).as_posix()
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

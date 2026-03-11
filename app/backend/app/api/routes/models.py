from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from app.api.dependencies import get_registry_dependency
from app.registry.model_registry import ModelRegistry, UploadedPayload
from app.schemas.models import (
    CatalogSnapshotResponse,
    DomainCatalogEntry,
    DomainCatalogModel,
    ModelDashboardResponse,
    ModelPatchRequest,
    ModelReorderRequest,
    UploadModelMetadata,
)


router = APIRouter(tags=["models"])


@router.get("/models/catalog", response_model=CatalogSnapshotResponse)
def get_models_catalog(
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> CatalogSnapshotResponse:
    return _build_snapshot_response(registry.snapshot())


@router.patch("/models/{model_id}", response_model=CatalogSnapshotResponse)
def patch_model(
    model_id: str,
    payload: ModelPatchRequest,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> CatalogSnapshotResponse:
    try:
        snapshot = registry.update_model(
            model_id,
            display_name=payload.display_name,
            is_active=payload.is_active,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _build_snapshot_response(snapshot)


@router.post("/models/reorder", response_model=CatalogSnapshotResponse)
def reorder_models(
    payload: ModelReorderRequest,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> CatalogSnapshotResponse:
    try:
        snapshot = registry.reorder_models(payload.ordered_model_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _build_snapshot_response(snapshot)


@router.delete("/models/{model_id}", response_model=CatalogSnapshotResponse)
def delete_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> CatalogSnapshotResponse:
    try:
        snapshot = registry.delete_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'.") from exc
    return _build_snapshot_response(snapshot)


@router.post("/models/upload", response_model=CatalogSnapshotResponse)
async def upload_model(
    request: Request,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> CatalogSnapshotResponse:
    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Multipart form parsing is unavailable in the backend environment. "
                "Install backend dependencies and restart the server."
            ),
        ) from exc

    metadata = form.get("metadata")
    if not isinstance(metadata, str):
        raise HTTPException(status_code=422, detail="Upload metadata is missing.")

    artifact_files = _collect_uploads(form.getlist("artifact_files"))
    dashboard_files = _collect_uploads(form.getlist("dashboard_files"))

    try:
        payload = UploadModelMetadata.model_validate_json(metadata)
        snapshot = registry.upload_model(
            payload,
            artifact_uploads=await _consume_uploads(artifact_files),
            dashboard_uploads=await _consume_uploads(dashboard_files),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _build_snapshot_response(snapshot)


@router.get("/models/{model_id}/dashboard", response_model=ModelDashboardResponse)
def get_model_dashboard(
    model_id: str,
    request: Request,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> ModelDashboardResponse:
    try:
        return registry.load_dashboard(
            model_id,
            asset_url_builder=lambda asset_path: str(
                request.url_for(
                    "get_model_dashboard_asset",
                    model_id=model_id,
                    asset_path=asset_path,
                )
            ),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'.") from exc


@router.get(
    "/models/{model_id}/dashboard/assets/{asset_path:path}",
    name="get_model_dashboard_asset",
)
def get_model_dashboard_asset(
    model_id: str,
    asset_path: str,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> FileResponse:
    try:
        asset = registry.dashboard_asset_path(model_id, asset_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_id}'.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(asset)


async def _consume_uploads(uploads: list[UploadFile]) -> list[UploadedPayload]:
    payloads: list[UploadedPayload] = []
    for upload in uploads:
        payloads.append(
            UploadedPayload(
                path=upload.filename or "upload.bin",
                content=await upload.read(),
            )
        )
    return payloads


def _collect_uploads(items: list[object]) -> list[UploadFile]:
    return [item for item in items if isinstance(item, UploadFile)]


def _build_snapshot_response(snapshot: dict[str, list[dict]]) -> CatalogSnapshotResponse:
    return CatalogSnapshotResponse(
        active_domains=_build_domain_entries(snapshot["active_domains"]),
        management_domains=_build_domain_entries(snapshot["management_domains"]),
    )


def _build_domain_entries(entries: list[dict]) -> list[DomainCatalogEntry]:
    return [
        DomainCatalogEntry(
            domain=entry["domain"],
            display_name=entry["display_name"],
            color_token=entry["color_token"],
            group=entry["group"],
            active_model_id=entry["active_model_id"],
            active_model_name=entry["active_model_name"],
            active_model_version=entry["active_model_version"],
            model_count=entry["model_count"],
            models=[DomainCatalogModel(**model) for model in entry["models"]],
        )
        for entry in entries
    ]

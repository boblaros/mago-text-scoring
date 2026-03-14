from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from app.api.dependencies import get_registry_dependency
from app.registry.model_registry import (
    ModelRegistry,
    RegistryValidationError,
    RegistrationOutcome,
    UploadedPayload,
)
from app.schemas.models import (
    CatalogSnapshotResponse,
    DomainCatalogEntry,
    DomainCatalogModel,
    HuggingFacePreflightRequest,
    HuggingFacePreflightResponse,
    LocalUploadPreflightRequest,
    LocalUploadPreflightResponse,
    ModelRegistrationResponse,
    ModelDashboardResponse,
    ModelPatchRequest,
    ModelReorderRequest,
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


@router.post("/models/local/preflight", response_model=LocalUploadPreflightResponse)
async def preflight_local_upload(
    request: Request,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> LocalUploadPreflightResponse:
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

    raw_payload = form.get("payload")
    if not isinstance(raw_payload, str):
        raise HTTPException(status_code=422, detail=_validation_detail("Upload payload is missing."))

    registration_config_files = _collect_uploads(form.getlist("registration_config_files"))

    artifact_files = _collect_uploads(form.getlist("artifact_files"))
    if artifact_files:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(
                "Artifact binaries should only be sent during the final import step."
            ),
        )

    try:
        payload = LocalUploadPreflightRequest.model_validate_json(raw_payload)
        return registry.preflight_local_upload(
            payload,
            registration_config_uploads=await _consume_uploads(registration_config_files),
        )
    except RegistryValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(str(exc), exc.field_errors),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(str(exc)),
        ) from exc


@router.post("/models/local/import", response_model=ModelRegistrationResponse)
async def import_local_model(
    request: Request,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> ModelRegistrationResponse:
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

    raw_payload = form.get("payload")
    if not isinstance(raw_payload, str):
        raise HTTPException(status_code=422, detail=_validation_detail("Upload payload is missing."))

    artifact_files = _collect_uploads(form.getlist("artifact_files"))
    dashboard_files = _collect_uploads(form.getlist("dashboard_files"))
    registration_config_files = _collect_uploads(form.getlist("registration_config_files"))

    try:
        payload = LocalUploadPreflightRequest.model_validate_json(raw_payload)
        outcome = registry.register_local_upload(
            payload,
            artifact_uploads=await _consume_uploads(artifact_files),
            dashboard_uploads=await _consume_uploads(dashboard_files),
            registration_config_uploads=await _consume_uploads(registration_config_files),
        )
    except RegistryValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(str(exc), exc.field_errors),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(str(exc)),
        ) from exc
    return _build_registration_response(outcome)


@router.post("/models/huggingface/preflight", response_model=HuggingFacePreflightResponse)
def preflight_huggingface_import(
    payload: HuggingFacePreflightRequest,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> HuggingFacePreflightResponse:
    try:
        return registry.preflight_huggingface_import(payload)
    except RegistryValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(str(exc), exc.field_errors),
        ) from exc


@router.post("/models/huggingface/import", response_model=ModelRegistrationResponse)
def import_huggingface_model(
    payload: HuggingFacePreflightRequest,
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> ModelRegistrationResponse:
    try:
        outcome = registry.import_huggingface_model(payload)
    except RegistryValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(str(exc), exc.field_errors),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=_validation_detail(str(exc)),
        ) from exc
    return _build_registration_response(outcome)


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


def _build_registration_response(outcome: RegistrationOutcome) -> ModelRegistrationResponse:
    return ModelRegistrationResponse(
        snapshot=_build_snapshot_response(outcome.snapshot),
        result=outcome.result,
    )


def _validation_detail(
    message: str,
    field_errors: dict[str, str] | None = None,
) -> dict[str, object]:
    detail: dict[str, object] = {"message": message}
    if field_errors:
        detail["field_errors"] = field_errors
    return detail

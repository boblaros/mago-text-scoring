from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_registry_dependency, get_settings_dependency
from app.core.settings import Settings
from app.registry.model_registry import ModelRegistry
from app.schemas.analysis import HealthResponse


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def healthcheck(
    settings: Settings = Depends(get_settings_dependency),
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> HealthResponse:
    return HealthResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        discovered_domains=registry.domains(),
    )


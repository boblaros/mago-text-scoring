from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    get_analysis_service_dependency,
    get_registry_dependency,
)
from app.registry.model_registry import ModelRegistry
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.schemas.models import DomainCatalogEntry, DomainCatalogModel, DomainCatalogResponse
from app.services.analysis_service import AnalysisService


router = APIRouter(tags=["analysis"])


@router.get("/domains", response_model=DomainCatalogResponse)
def get_domains(
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> DomainCatalogResponse:
    return DomainCatalogResponse(
        domains=[
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
            for entry in registry.catalog(active_only=True)
        ]
    )


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_text(
    payload: AnalysisRequest,
    analysis_service: AnalysisService = Depends(get_analysis_service_dependency),
) -> AnalysisResponse:
    return analysis_service.analyze(payload.text, payload.domains)

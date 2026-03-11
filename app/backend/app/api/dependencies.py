from __future__ import annotations

from fastapi import Request

from app.core.settings import Settings
from app.registry.model_registry import ModelRegistry
from app.services.analysis_service import AnalysisService


def get_settings_dependency(request: Request) -> Settings:
    return request.app.state.settings


def get_registry_dependency(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_analysis_service_dependency(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


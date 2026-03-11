from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router
from app.api.routes.models import router as models_router
from app.api.routes.system import router as system_router
from app.core.settings import get_settings
from app.inference.factory import InferencePluginRegistry
from app.registry.model_registry import ModelRegistry
from app.services.analysis_service import AnalysisService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mago-text-scoring")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    plugin_registry = InferencePluginRegistry()
    registry = ModelRegistry(settings=settings, plugin_registry=plugin_registry)
    registry.discover()
    if settings.model_preload:
        logger.info("Preloading active models.")
        registry.preload_active_models()

    app.state.settings = settings
    app.state.registry = registry
    app.state.analysis_service = AnalysisService(registry)
    yield


app = FastAPI(
    title="Mago Text Scoring",
    version=get_settings().app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router, prefix=get_settings().api_prefix)
app.include_router(analysis_router, prefix=get_settings().api_prefix)
app.include_router(models_router, prefix=get_settings().api_prefix)

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "Mago Text Scoring"
    app_version: str = "0.1.0-alpha"
    api_prefix: str = "/api/v1"
    debug: bool = False

    model_discovery_root: Path = Field(default=APP_ROOT / "app-models")
    model_preload: bool = False

    alpha_domains: list[str] = Field(
        default_factory=lambda: ["sentiment", "complexity", "age", "abuse"]
    )
    domain_aliases: dict[str, str] = Field(
        default_factory=lambda: {"abuse-detection": "abuse"}
    )

    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=str(APP_ROOT / ".env"),
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("model_discovery_root", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        return Path(value)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if value.startswith("["):
            return json.loads(value)
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("alpha_domains", mode="before")
    @classmethod
    def _parse_alpha_domains(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, list):
            return value
        if value.startswith("["):
            return json.loads(value)
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("domain_aliases", mode="before")
    @classmethod
    def _parse_aliases(cls, value: str | dict[str, str]) -> dict[str, str]:
        if isinstance(value, dict):
            return value
        return json.loads(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class YamlConfigSource(PydanticBaseSettingsSource):
    """Loads config from a YAML file. Lower priority than .env and env vars."""

    def __init__(self, settings_cls, yaml_path: Path):
        super().__init__(settings_cls)
        self._yaml_path = yaml_path

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not self._yaml_path.exists():
            return {}

        with self._yaml_path.open() as file_handle:
            data = yaml.safe_load(file_handle) or {}

        if not isinstance(data, dict):
            return {}

        remapped: dict[str, Any] = {}
        cache_block = data.pop("cache", {}) or {}

        for key, value in data.items():
            remapped[key.upper().replace("-", "_")] = value

        if isinstance(cache_block, dict):
            for key, value in cache_block.items():
                remapped[f"CACHE_{key.upper().replace('-', '_')}"] = value

        return remapped


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    WORKERS: int = 4
    LOG_LEVEL: str = "info"
    DEBUG: bool = False
    API_KEY_1: str = ""
    API_KEY_2: str = ""
    API_KEY_3: str = ""
    PERPLEXITY_COOKIES: dict[str, str] = Field(default_factory=dict)
    CACHE_ENABLED: bool = True
    CACHE_MAX_SIZE: int = 256
    CACHE_TTL_SECONDS: int = 300
    REFRESH_SECRET: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
        **kwargs,
    ):
        yaml_path = Path(os.environ.get("CONFIG_FILE", "config.yaml")).expanduser()
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSource(settings_cls, yaml_path),
            file_secret_settings,
        )

    @field_validator("PERPLEXITY_COOKIES", mode="before")
    @classmethod
    def parse_cookies(cls, value: object) -> object:
        if isinstance(value, str):
            return json.loads(value)
        if value is None:
            return {}
        return value


settings = Settings()

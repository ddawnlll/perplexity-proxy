from __future__ import annotations

import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    WORKERS: int = 4
    LOG_LEVEL: str = "info"
    PERPLEXITY_COOKIES: dict[str, str] = Field(default_factory=dict)
    CACHE_ENABLED: bool = True
    CACHE_MAX_SIZE: int = 256
    CACHE_TTL_SECONDS: int = 300

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("PERPLEXITY_COOKIES", mode="before")
    @classmethod
    def parse_cookies(cls, value: object) -> object:
        if isinstance(value, str):
            return json.loads(value)
        if value is None:
            return {}
        return value


settings = Settings()

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.client import close_client, init_client
from app.config import settings
from app.router import router

logger = logging.getLogger(__name__)
API_KEY_EXEMPT_PATHS = {"/health", "/v1/models/refresh", "/docs", "/openapi.json", "/redoc"}


def _configured_api_keys() -> set[str]:
    return {key for key in (settings.API_KEY_1, settings.API_KEY_2, settings.API_KEY_3) if key}


def _debug_settings_snapshot() -> dict[str, Any]:
    values = settings.model_dump()
    cookies = values.get("PERPLEXITY_COOKIES")
    if isinstance(cookies, dict):
        values["PERPLEXITY_COOKIES"] = {key: "***" for key in cookies}
    elif cookies is not None:
        values["PERPLEXITY_COOKIES"] = "***"

    for secret_key in ("API_KEY_1", "API_KEY_2", "API_KEY_3", "REFRESH_SECRET"):
        if values.get(secret_key):
            values[secret_key] = "***"
    return values


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.DEBUG:
        logger.info("Debug settings: %s", _debug_settings_snapshot())
    await init_client()
    yield
    await close_client()


async def api_key_middleware(request: Request, call_next):
    if request.url.path in API_KEY_EXEMPT_PATHS or not request.url.path.startswith("/v1/"):
        return await call_next(request)

    configured_keys = _configured_api_keys()
    if not configured_keys:
        return await call_next(request)

    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Missing API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_key = authorization.removeprefix("Bearer ").strip()
    if provided_key not in configured_keys:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Invalid API key"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


def create_app() -> FastAPI:
    app = FastAPI(
        title="perplexity-proxy",
        description="OpenAI-compatible proxy for Perplexity AI",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.middleware("http")(api_key_middleware)
    app.include_router(router)
    return app


app = create_app()

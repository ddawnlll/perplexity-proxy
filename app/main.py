from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import sys
import time
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.client import check_perplexity_session, close_client, init_client
from app.config import settings
from app.router import router
from perplexity.models import _STATIC_MAP, _registry

logger = logging.getLogger("perplexity_proxy")
access_logger = logging.getLogger("perplexity_proxy.access")
API_KEY_EXEMPT_PATHS = {"/health", "/v1/models/refresh", "/docs", "/openapi.json", "/redoc"}


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)

    for name in ("perplexity_proxy", "perplexity_proxy.access", "app"):
        logger_obj = logging.getLogger(name)
        logger_obj.handlers.clear()
        logger_obj.setLevel(level)
        logger_obj.addHandler(handler)
        logger_obj.propagate = False


def _configured_api_keys() -> set[str]:
    return {key for key in (settings.API_KEY_1, settings.API_KEY_2, settings.API_KEY_3) if key}


def _status_to_error_type(status_code: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "invalid_request_error",
        408: "invalid_request_error",
        422: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status_code, "server_error")


def _openai_error(message: str, status_code: int, code: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": _status_to_error_type(status_code),
            "code": code,
        }
    }


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


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start = time.monotonic()

        access_logger.info("→ %s %s", request.method, request.url.path)

        if settings.DEBUG:
            safe_headers = {
                key: ("***" if key.lower() in ("authorization", "cookie") else value)
                for key, value in request.headers.items()
            }
            access_logger.debug("  headers: %s", safe_headers)

        status_code = 500

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            access_logger.error(
                "← ERR  %s %s  %dms  %s: %s",
                request.method,
                request.url.path,
                elapsed_ms,
                type(exc).__name__,
                exc,
            )
            raise

        elapsed_ms = int((time.monotonic() - start) * 1000)
        level = logging.WARNING if status_code >= 400 else logging.INFO
        access_logger.log(
            level,
            "← %d  %s %s  %dms",
            status_code,
            request.method,
            request.url.path,
            elapsed_ms,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.DEBUG:
        logger.info("Debug settings: %s", _debug_settings_snapshot())
    await init_client()

    if settings.PERPLEXITY_COOKIES:
        health = await check_perplexity_session()
        if health.get("ok") and health.get("authenticated"):
            logger.info("Perplexity auth health check passed")
        else:
            logger.warning("Perplexity auth health check failed: %s", health)
    else:
        logger.info("Perplexity auth health check skipped (no cookies configured)")

    static_count = sum(len(v) for v in _STATIC_MAP.values())
    live_count = sum(len(v) for v in _registry.available().values())
    discovered_count = live_count - static_count

    logger.info(
        "perplexity-proxy ready  host=%s  port=%s  debug=%s",
        settings.HOST,
        settings.PORT,
        settings.DEBUG,
    )
    logger.info(
        "model registry: %d static models, %d discovered models (%d total)",
        static_count,
        discovered_count,
        live_count,
    )

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
        logger.info("%s %s | status=401 | auth=missing", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=_openai_error("Missing API key", status.HTTP_401_UNAUTHORIZED, "invalid_api_key"),
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_key = authorization.removeprefix("Bearer ").strip()
    if provided_key not in configured_keys:
        logger.info("%s %s | status=401 | auth=invalid", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=_openai_error("Invalid API key", status.HTTP_401_UNAUTHORIZED, "invalid_api_key"),
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        status_code = exc.status_code or 500
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            body = detail
        else:
            message = detail if isinstance(detail, str) and detail else str(detail or "HTTP error")
            code = None
            if status_code == 401:
                code = "invalid_api_key"
            elif status_code == 404:
                code = "model_not_found"
            elif status_code == 429:
                code = "rate_limit_exceeded"
            elif status_code >= 500:
                code = "internal_server_error"
            body = _openai_error(message, status_code, code)
        headers = getattr(exc, "headers", None)
        return JSONResponse(status_code=status_code, content=body, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_openai_error(f"Validation error: {exc.errors()}", 422, "validation_error"),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Internal proxy error: {type(exc).__name__}: {exc}",
                    "type": "server_error",
                    "code": "proxy_error",
                }
            },
        )


def create_app() -> FastAPI:
    configure_logging(settings.DEBUG)
    app = FastAPI(
        title="perplexity-proxy",
        description="OpenAI-compatible proxy for Perplexity AI",
        version="1.0.0",
        lifespan=lifespan,
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1, "displayRequestDuration": True},
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.middleware("http")(api_key_middleware)
    app.include_router(router)
    register_exception_handlers(app)
    return app


app = create_app()

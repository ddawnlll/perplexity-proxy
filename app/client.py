from __future__ import annotations

import asyncio

import perplexity_async
from fastapi import HTTPException
from perplexity.exceptions import (
    AccountCreationError,
    AuthenticationError,
    EmailnatorError,
    FileUploadError,
    InvalidModelError,
    InvalidModeError,
    InvalidSourceError,
    NetworkError,
    ParsingError as ResponseParseError,
    QueryLimitExceededError,
    RateLimitError,
    ValidationError,
)

from app.config import settings

_client: perplexity_async.Client | None = None

EXCEPTION_MAP = {
    AuthenticationError: 401,
    RateLimitError: 429,
    NetworkError: 503,
    ValidationError: 400,
    ResponseParseError: 502,
    InvalidModeError: 400,
    InvalidModelError: 400,
    InvalidSourceError: 400,
    QueryLimitExceededError: 429,
    FileUploadError: 502,
    EmailnatorError: 502,
    AccountCreationError: 502,
}


async def init_client():
    global _client
    _client = await perplexity_async.Client(dict(settings.PERPLEXITY_COOKIES or {}))


async def close_client():
    global _client
    _client = None


def get_client() -> perplexity_async.Client:
    if _client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    return _client


def _map_exception(error: Exception) -> HTTPException | None:
    for exception_type, status_code in EXCEPTION_MAP.items():
        if isinstance(error, exception_type):
            return HTTPException(status_code=status_code, detail=str(error))
    return None


async def search(query: str, mode: str, model, stream: bool = False):
    client = get_client()
    retries = 3
    for attempt in range(retries):
        try:
            return await client.search(query, mode=mode, model=model, stream=stream)
        except NetworkError as error:
            if attempt == retries - 1:
                raise HTTPException(status_code=503, detail="Upstream unavailable") from error
            await asyncio.sleep(2 ** attempt)
        except Exception as error:
            mapped_error = _map_exception(error)
            if mapped_error is not None:
                raise mapped_error from error
            raise HTTPException(status_code=500, detail=str(error)) from error


__all__ = ["EXCEPTION_MAP", "close_client", "get_client", "init_client", "search"]

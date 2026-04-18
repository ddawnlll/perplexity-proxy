from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.cache import cache
from app.client import get_client, search
from app.config import settings
from app import mapper
from app.mapper import get_model_list, resolve


def build_model_map():
    return mapper.build_model_map()
from app.models import (
    ChatChoice,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatUsage,
    HealthResponse,
    ModelList,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from app.streaming import chat_completions_stream, responses_stream

router = APIRouter()


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _extract_chat_query(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "role", None) == "user":
            return _content_to_text(getattr(message, "content", None))
    if messages:
        return _content_to_text(getattr(messages[-1], "content", None))
    return ""


def _extract_responses_query(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        for message in reversed(input_value):
            if getattr(message, "role", None) == "user":
                return _content_to_text(getattr(message, "content", None))
        if input_value:
            return _content_to_text(getattr(input_value[-1], "content", None))
    return _content_to_text(input_value)


def _extract_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("answer", "text", "content", "response"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(result, ensure_ascii=False)
    return _content_to_text(result)


def _chat_response(model_name: str, result: Any) -> ChatResponse:
    text = _extract_result_text(result)
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatResponseMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(),
    )


def _responses_response(model_name: str, result: Any) -> ResponsesResponse:
    text = _extract_result_text(result)
    return ResponsesResponse(
        id=f"resp-{uuid.uuid4().hex}",
        object="response",
        created_at=int(time.time()),
        model=model_name,
        output=[
            ResponsesOutputMessage(
                type="message",
                role="assistant",
                content=[ResponsesOutputText(type="text", text=text)],
            )
        ],
        usage=ResponsesUsage(),
    )


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=get_model_list())


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        cache_enabled=settings.CACHE_ENABLED,
        authenticated=bool(settings.PERPLEXITY_COOKIES),
        api_key_auth_enabled=bool(settings.API_KEY_1 or settings.API_KEY_2 or settings.API_KEY_3),
        model_count=len(mapper.MODEL_MAP),
    )


@router.post("/v1/models/refresh")
async def refresh_models_endpoint(authorization: str = Header(...)):
    expected = f"Bearer {settings.REFRESH_SECRET}"
    if not settings.REFRESH_SECRET or authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing refresh secret")

    client = get_client()
    success = await client.refresh_models()

    if not success:
        raise HTTPException(
            status_code=503,
            detail="Model refresh failed — static map still active",
        )

    mapper.MODEL_MAP = build_model_map()

    return {
        "status": "ok",
        "model_count": len(mapper.MODEL_MAP),
        "models": list(mapper.MODEL_MAP.keys()),
    }


@router.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    mode, model = resolve(req.model)
    query = _extract_chat_query(req.messages)
    cache_key = cache.make_key(query, req.model)

    if req.stream:
        generator = await search(query, mode, model, stream=True)
        return StreamingResponse(
            chat_completions_stream(generator, req.model, f"chatcmpl-{uuid.uuid4().hex}"),
            media_type="text/event-stream",
        )

    cached = await cache.get(cache_key)
    if cached is not None:
        return ChatResponse.model_validate_json(cached)

    result = await search(query, mode, model, stream=False)
    response = _chat_response(req.model, result)
    await cache.set(cache_key, response.model_dump_json())
    return response


@router.post("/v1/responses", response_model=ResponsesResponse)
async def responses(req: ResponsesRequest):
    mode, model = resolve(req.model)
    query = _extract_responses_query(req.input)
    cache_key = cache.make_key(query, req.model)

    if req.stream:
        generator = await search(query, mode, model, stream=True)
        return StreamingResponse(
            responses_stream(generator, req.model, f"resp-{uuid.uuid4().hex}"),
            media_type="text/event-stream",
        )

    cached = await cache.get(cache_key)
    if cached is not None:
        return ResponsesResponse.model_validate_json(cached)

    result = await search(query, mode, model, stream=False)
    response = _responses_response(req.model, result)
    await cache.set(cache_key, response.model_dump_json())
    return response


__all__ = ["router"]

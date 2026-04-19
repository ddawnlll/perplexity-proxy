from __future__ import annotations

import json
import logging
import re as _re
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import ValidationError as PydanticValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from perplexity.exceptions import PerplexityError

from app import mapper
from app.cache import cache
from app.client import get_client, search
from app.config import settings
from app.mapper import get_model_list, resolve
from app.state import follow_up_store
from app.models import (
    ChatChoice,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatResponseMessage,
    ChatUsage,
    CompletionsChoice,
    CompletionsRequest,
    CompletionsResponse,
    HealthResponse,
    ModelList,
    RefreshResponse,
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
)
from app.streaming import _extract_text, _snapshot_to_delta, chat_completions_stream, completions_stream, responses_stream
from app.tools.shim import (
    _last_tool_call,
    build_perplexity_instruction,
    decide_tool,
    is_roo_request,
    wrap_as_tool_response,
)

router = APIRouter()
logger = logging.getLogger("app.requests")

def build_model_map():
    return mapper.build_model_map()


def _extract_system_message(messages: list[ChatMessage]) -> str | None:
    for msg in messages:
        role = ""
        content: Any = ""
        if isinstance(msg, dict):
            role = str(msg.get("role", "")).strip()
            content = msg.get("content", "")
        else:
            role = str(getattr(msg, "role", "")).strip()
            content = getattr(msg, "content", "")
        if role != "system":
            continue
        if isinstance(content, str):
            return content
    return None


def _extract_last_read_path(messages: list[ChatMessage]) -> str | None:
    for msg in reversed(messages):
        role = ""
        content: Any = None
        tool_calls: list[Any] = []
        if isinstance(msg, dict):
            role = str(msg.get("role", "")).strip()
            content = msg.get("content", None)
            raw_tool_calls = msg.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                tool_calls.extend(item for item in raw_tool_calls if isinstance(item, dict))
        else:
            role = str(getattr(msg, "role", "")).strip()
            content = getattr(msg, "content", None)
        if role != "assistant":
            continue
        if isinstance(content, list):
            tool_calls.extend(item for item in content if isinstance(item, dict))
        elif isinstance(content, dict):
            tool_calls.append(content)
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            if not isinstance(function, dict) or function.get("name") != "read_file":
                continue
            try:
                args = json.loads(function.get("arguments", "{}"))
            except Exception:
                continue
            path = args.get("path")
            if isinstance(path, str) and path:
                return path
    return None


def _message_to_dict(message: ChatMessage) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "role": str(message.get("role", "")).strip(),
            "content": message.get("content", None),
            "tool_calls": message.get("tool_calls", None),
        }
    return {
        "role": str(getattr(message, "role", "")).strip(),
        "content": getattr(message, "content", None),
    }


def _clean_user_content(text: str) -> str:
    """Remove Roo wrappers and environment_details noise from user text."""
    user_message_match = _re.search(r"<user_message>\s*(.*?)\s*</user_message>", text, _re.DOTALL)
    if user_message_match:
        return user_message_match.group(1).strip()
    text = _re.sub(r"<environment_details>.*?</environment_details>", "", text, flags=_re.DOTALL)
    text = _re.sub(r"</?user_message>", "", text)
    return text.strip()

def _extract_query(messages: list) -> str:
    """
    Return the last user message content as the primary query.
    Iterates in reverse so the first match is the most recent.
    """
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "")

        if role != "user":
            continue

        if isinstance(content, str):
            cleaned = _clean_user_content(content)
            if cleaned:
                return cleaned
            # content existed but was all noise (env_details etc) — keep looking
            continue

        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    continue
                raw_text = part.get("text", "")
                if not isinstance(raw_text, str) or not raw_text:
                    continue
                cleaned = _clean_user_content(raw_text)
                if cleaned:
                    return cleaned
                texts.append(raw_text)

            joined = _clean_user_content("\n".join(texts))
            if joined:
                return joined
            # all blocks were noise — keep looking for an earlier real message
            continue

    return ""
    

def _strip_citations(text: str) -> str:
    """Remove Perplexity citation markers from plain-text output."""
    text = text.strip()
    text = _re.sub(r"^\*\*[^*\n]+\*\*\s*\n?", "", text)
    text = _re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n?", "", text)
    text = _re.sub(r"\n?```$", "", text)
    text = _re.sub(r"\[\d+\]", "", text)
    text = _re.sub(r"\b\w+\+\d+\b", "", text)
    text = _re.sub(r"\[\d+(,\s*\d+)*\]", "", text)
    return text.strip()


def _wrap_as_tool_call(text: str) -> dict[str, list[dict[str, Any]]]:
    """
    Wrap a plain-text Perplexity answer as an attempt_completion tool call.
    Roo Code mode accepts this and marks the task as complete.
    """
    return {
        "tool_calls": [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": "attempt_completion",
                    "arguments": json.dumps({"result": text}),
                },
            }
        ]
    }


async def _tool_call_stream(
    *,
    model: str,
    response_id: str,
    decision: dict[str, Any],
    prose: str | None,
) -> AsyncGenerator[str, None]:
    payload = wrap_as_tool_response(prose, model, response_id, decision)
    tool_calls = payload["choices"][0]["message"]["tool_calls"]
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': None, 'tool_calls': tool_calls}, 'finish_reason': 'tool_calls'}]}, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _extract_content_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part for part in parts if part)
    return str(content)


def _extract_prompt_text(prompt: str | list[str] | None) -> str:
    if prompt is None:
        return "Complete this:"
    if isinstance(prompt, str):
        text = prompt.strip()
        return text or "Complete this:"
    parts = [str(item).strip() for item in prompt if str(item).strip()]
    return "\n".join(parts) if parts else "Complete this:"


def _one_line(value: str) -> str:
    return " ".join(value.replace("\n", " ").replace("\r", " ").split())


def _truncate(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _summary(value: Any, limit: int = 120) -> str:
    return _truncate(_one_line(_content_to_text(value)), limit)


def _log_summary(route: str, status_code: int, **fields: Any) -> None:
    parts = [route, f"status={status_code}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    logger.info(" | ".join(parts))


def _upstream_error(message: str) -> HTTPException:
    return HTTPException(
        status_code=502,
        detail={
            "error": {
                "message": message,
                "type": "upstream_error",
                "code": "perplexity_error",
            }
        },
    )


def _internal_error(context: str, exc: Exception) -> HTTPException:
    logger.exception("Unexpected error in %s: %s", context, exc)
    return HTTPException(
        status_code=500,
        detail={
            "error": {
                "message": "Internal proxy error",
                "type": "internal_error",
                "code": "proxy_error",
            }
        },
    )


def _invalid_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "invalid_request_error",
            }
        },
    )


def _build_query_from_messages(messages: list[ChatMessage]) -> tuple[str, str | None]:
    system_prompt: str | None = None
    turns: list[str] = []

    for message in messages:
        role = str(getattr(message, "role", "")).strip()
        content = _extract_content_str(getattr(message, "content", None))
        if role == "system" and system_prompt is None:
            system_prompt = content or None
            continue
        if role == "assistant":
            turns.append(f"[assistant]: {content}")
            continue
        if role == "user":
            turns.append(content)
            continue
        if content:
            turns.append(f"[{role}]: {content}" if role else content)

    if not turns:
        query = _extract_content_str(messages[-1].content) if messages else ""
    elif len(turns) == 1:
        query = turns[0]
    else:
        query = "\n".join(turns[:-1]) + "\n\n" + turns[-1]

    return query, system_prompt


def _serialize_messages(
    messages: list[ChatMessage],
    *,
    include_tool_messages: bool = True,
) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for message in messages:
        role = str(getattr(message, "role", "")).strip()
        if not include_tool_messages and role == "tool":
            continue
        content = _extract_content_str(getattr(message, "content", None))
        if role == "user":
            content = _clean_user_content(content)
        serialized.append({"role": role, "content": content})
    return serialized


def _continuation_transcript_key(messages: list[ChatMessage]) -> str | None:
    if not messages:
        return None
    continuation = messages
    last_role = str(getattr(messages[-1], "role", "")).strip()
    if last_role == "user":
        continuation = messages[:-1]
    if not continuation:
        return None
    if not any(str(getattr(message, "role", "")).strip() == "assistant" for message in continuation):
        return None
    return follow_up_store.make_transcript_key(
        _serialize_messages(continuation, include_tool_messages=False)
    )


def _assistant_transcript_key(messages: list[ChatMessage], assistant_text: str) -> str:
    transcript = _serialize_messages(messages) + [{"role": "assistant", "content": assistant_text}]
    return follow_up_store.make_transcript_key(transcript)


def _roo_assistant_transcript_key(messages: list[ChatMessage]) -> str:
    transcript = _serialize_messages(messages, include_tool_messages=False) + [
        {"role": "assistant", "content": ""}
    ]
    return follow_up_store.make_transcript_key(transcript)


def _extract_follow_up_state(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    backend_uuid = payload.get("backend_uuid")
    if not isinstance(backend_uuid, str) or not backend_uuid:
        return None
    attachments = payload.get("attachments")
    if not isinstance(attachments, list):
        attachments = []
    return {"backend_uuid": backend_uuid, "attachments": list(attachments)}


async def _resolve_follow_up(
    *,
    response_id: str | None = None,
    transcript_key: str | None = None,
) -> dict[str, Any] | None:
    follow_up = await follow_up_store.get_by_response_id(response_id)
    if follow_up is not None:
        return follow_up
    return await follow_up_store.get_by_transcript(transcript_key)


async def _store_follow_up(
    *,
    response_id: str,
    transcript_key: str | None,
    payload: Any,
) -> None:
    follow_up = _extract_follow_up_state(payload)
    if follow_up is None:
        return
    await follow_up_store.put(response_id=response_id, transcript_key=transcript_key, follow_up=follow_up)


def _wrap_stream_with_follow_up(
    generator: Any,
    *,
    response_id: str,
    transcript_key: str | None,
    transcript_messages: list[ChatMessage] | None = None,
) -> AsyncGenerator[Any, None]:
    async def wrapped() -> AsyncGenerator[Any, None]:
        latest_follow_up: dict[str, Any] | None = None
        rendered_text = ""
        async for chunk in generator:
            follow_up = _extract_follow_up_state(chunk)
            if follow_up is not None:
                latest_follow_up = follow_up
            snapshot = _extract_text(chunk)
            if snapshot:
                _, rendered_text = _snapshot_to_delta(snapshot, rendered_text)
            yield chunk
        if latest_follow_up is not None:
            final_transcript_key = transcript_key
            if final_transcript_key is None and transcript_messages and rendered_text:
                final_transcript_key = _assistant_transcript_key(transcript_messages, rendered_text)
            await follow_up_store.put(
                response_id=response_id,
                transcript_key=final_transcript_key,
                follow_up=latest_follow_up,
            )

    return wrapped()


def _normalize_chat_payload(raw: Any) -> ChatRequest:
    if not isinstance(raw, dict):
        raise _invalid_request("Invalid request body")

    normalized: dict[str, Any] = {
        "model": raw.get("model"),
        "stream": raw.get("stream", False),
        "temperature": raw.get("temperature"),
        "max_tokens": raw.get("max_tokens"),
        "top_p": raw.get("top_p"),
        "tools": raw.get("tools"),
        "tool_choice": raw.get("tool_choice"),
        "parallel_tool_calls": raw.get("parallel_tool_calls"),
    }

    messages = raw.get("messages")
    if isinstance(messages, list) and messages:
        normalized["messages"] = messages
        try:
            return ChatRequest.model_validate(normalized)
        except PydanticValidationError as exc:
            raise _invalid_request(str(exc)) from exc

    if "input" in raw or "instructions" in raw:
        normalized_messages: list[dict[str, Any]] = []
        instructions = raw.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            normalized_messages.append({"role": "system", "content": instructions})

        input_value = raw.get("input")
        if isinstance(input_value, str):
            normalized_messages.append({"role": "user", "content": input_value})
        elif isinstance(input_value, list):
            for item in input_value:
                if isinstance(item, dict):
                    normalized_messages.append(item)
                else:
                    normalized_messages.append({"role": "user", "content": _extract_content_str(item)})
        elif input_value is not None:
            normalized_messages.append({"role": "user", "content": _extract_content_str(input_value)})

        normalized["messages"] = normalized_messages
        try:
            return ChatRequest.model_validate(normalized)
        except PydanticValidationError as exc:
            raise _invalid_request(str(exc)) from exc

    raise _invalid_request("Missing messages or input")


def _normalize_completions_payload(raw: Any) -> CompletionsRequest:
    if not isinstance(raw, dict):
        raise _invalid_request("Invalid request body")
    try:
        return CompletionsRequest.model_validate(raw)
    except PydanticValidationError as exc:
        raise _invalid_request(str(exc)) from exc


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


def _chat_response(model_name: str, result: Any, *, roo_mode: bool = False) -> ChatResponse:
    text = _extract_result_text(result)
    message = ChatResponseMessage(role="assistant", content=text)
    finish_reason = "stop"
    if roo_mode:
        message = ChatResponseMessage(role="assistant", content=None, tool_calls=_wrap_as_tool_call(text)["tool_calls"])
        finish_reason = "tool_calls"
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        usage=ChatUsage(),
    )


def _completions_response(model_name: str, result: Any) -> CompletionsResponse:
    text = _extract_result_text(result)
    return CompletionsResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model_name,
        choices=[CompletionsChoice(index=0, text=text, finish_reason="stop")],
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


def _extract_responses_query(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        query, system_prompt = _build_query_from_messages(input_value)
        if system_prompt:
            query = f"{system_prompt}\n\n{query}"
        return query
    return _content_to_text(input_value)


async def _prefetch_first_chunk(stream: Any):
    first_chunk = await anext(stream)

    async def replay():
        yield first_chunk
        async for item in stream:
            yield item

    return replay()


@router.get(
    "/v1/models",
    response_model=ModelList,
    tags=["Models"],
    summary="List available models",
    description="Returns the currently available proxy model IDs exposed by the upstream Perplexity model map.",
)
async def list_models() -> ModelList:
    model_list = ModelList(data=get_model_list())
    _log_summary("GET /v1/models", 200, models=len(model_list.data))
    return model_list


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Reports proxy status, cache status, cookie auth status, and whether API-key auth is enabled.",
)
async def health() -> HealthResponse:
    response = HealthResponse(
        status="ok",
        cache_enabled=settings.CACHE_ENABLED,
        authenticated=bool(settings.PERPLEXITY_COOKIES),
        api_key_auth_enabled=bool(settings.API_KEY_1 or settings.API_KEY_2 or settings.API_KEY_3),
        model_count=len(mapper.MODEL_MAP),
    )
    _log_summary(
        "GET /health",
        200,
        cache_enabled=response.cache_enabled,
        authenticated=response.authenticated,
        api_key_auth_enabled=response.api_key_auth_enabled,
        model_count=response.model_count,
    )
    return response


@router.post(
    "/v1/models/refresh",
    response_model=RefreshResponse,
    tags=["Models"],
    summary="Refresh the upstream model map",
    description="Refreshes the dynamic model map from Perplexity using the active session cookies.",
    responses={401: {"description": "Invalid or missing refresh secret"}, 503: {"description": "Refresh failed"}},
)
async def refresh_models_endpoint(authorization: str = Header(...)) -> RefreshResponse:
    try:
        expected = f"Bearer {settings.REFRESH_SECRET}"
        if not settings.REFRESH_SECRET or authorization != expected:
            _log_summary("POST /v1/models/refresh", 401, result="invalid refresh secret")
            raise HTTPException(status_code=401, detail="Invalid or missing refresh secret")

        client = get_client()
        success = await client.refresh_models()

        if not success:
            _log_summary("POST /v1/models/refresh", 503, result="refresh failed")
            raise HTTPException(
                status_code=503,
                detail="Model refresh failed — static map still active",
            )

        mapper.MODEL_MAP = build_model_map()

        response = RefreshResponse(
            status="ok",
            model_count=len(mapper.MODEL_MAP),
            models=list(mapper.MODEL_MAP.keys()),
        )
        _log_summary("POST /v1/models/refresh", 200, model_count=response.model_count)
        return response
    except HTTPException:
        raise
    except PerplexityError as exc:
        raise _upstream_error(str(exc)) from exc
    except Exception as exc:
        raise _internal_error("refresh_models_endpoint", exc) from exc


@router.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Create a chat completion",
    description="OpenAI-compatible chat completion endpoint that forwards the conversation to Perplexity.",
    responses={401: {"description": "Invalid API key"}, 400: {"description": "Invalid model"}},
)
async def chat_completions(request: Request):
    try:
        raw_body = await request.json()
        raw_messages = raw_body.get("messages") if isinstance(raw_body.get("messages"), list) else []
        req = _normalize_chat_payload(raw_body)
        roo_mode = is_roo_request(req.tools)
        mode, model = resolve(req.model)
        continuation_key = _continuation_transcript_key(req.messages)
        follow_up = await _resolve_follow_up(transcript_key=continuation_key)
        query, system_prompt = _build_query_from_messages(req.messages)
        if roo_mode:
            if not raw_messages:
                logger.error("raw_messages empty! roo extraction will fail.")
            roo_messages = raw_messages
            user_query = _extract_query(roo_messages)
            logger.info("roo extracted query=%s", _summary(user_query))
            decision = decide_tool(roo_messages, user_query)
            perplexity_q = build_perplexity_instruction(decision, user_query, messages=roo_messages)
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            prose = decision.get("static_result")
            raw_result: Any = None
            if perplexity_q is not None:
                raw_result = await search(perplexity_q, mode, model, stream=False, follow_up=follow_up)
                prose = _strip_citations(_extract_result_text(raw_result))
                await _store_follow_up(
                    response_id=response_id,
                    transcript_key=_roo_assistant_transcript_key(req.messages),
                    payload=raw_result,
                )
            logger.warning(
                "ROO TURN AUDIT | tool=%s | path=%s | prose_len=%d | last_tool=%s",
                decision["tool"],
                decision.get("args_hint", {}).get("path", "-"),
                len(prose or ""),
                (_last_tool_call(roo_messages) or [None])[0],
            )
            _log_summary(
                "POST /v1/chat/completions",
                200,
                model=req.model,
                mode=mode,
                stream=req.stream,
                query=_summary(perplexity_q or user_query),
                response=_summary(prose or ""),
                roo_tool=decision["tool"],
            )
            if req.stream:
                return StreamingResponse(
                    _tool_call_stream(
                        model=req.model,
                        response_id=response_id,
                        decision=decision,
                        prose=prose,
                    ),
                    media_type="text/event-stream",
                )
            return JSONResponse(content=wrap_as_tool_response(prose, req.model, response_id, decision))
        elif system_prompt:
            query = f"{system_prompt}\n\n{query}"

        cache_key = None
        if not roo_mode:
            cache_key = cache.make_key(
                query,
                req.model,
                request_type="chat",
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                top_p=req.top_p,
                stream=req.stream,
                tools=req.tools,
                tool_choice=req.tool_choice,
                parallel_tool_calls=req.parallel_tool_calls,
            )

        if req.stream:
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            generator = await search(query, mode, model, stream=True, follow_up=follow_up)
            stream = chat_completions_stream(
                _wrap_stream_with_follow_up(
                    generator,
                    response_id=response_id,
                    transcript_key=None,
                    transcript_messages=req.messages,
                ),
                req.model,
                response_id,
                roo_mode=roo_mode,
            )
            _log_summary(
                "POST /v1/chat/completions",
                200,
                model=req.model,
                mode=mode,
                stream=True,
                query=_summary(query),
                response="stream",
            )
            return StreamingResponse(await _prefetch_first_chunk(stream), media_type="text/event-stream")

        if cache_key is not None:
            cached = await cache.get(cache_key)
            if cached is not None:
                response = ChatResponse.model_validate_json(cached)
                cached_text = ""
                if response.choices:
                    cached_text = _extract_result_text(response.choices[0].message.content)
                _log_summary(
                    "POST /v1/chat/completions",
                    200,
                    model=req.model,
                    mode=mode,
                    stream=False,
                    query=_summary(query),
                    response=_summary(cached_text),
                    cache="hit",
                )
                return response

        result = await search(query, mode, model, stream=False, follow_up=follow_up)
        response = _chat_response(req.model, result, roo_mode=False)
        await _store_follow_up(
            response_id=response.id,
            transcript_key=_assistant_transcript_key(req.messages, _extract_result_text(result)),
            payload=result,
        )
        if cache_key is not None:
            await cache.set(cache_key, response.model_dump_json())
        _log_summary(
            "POST /v1/chat/completions",
            200,
            model=req.model,
            mode=mode,
            stream=False,
            query=_summary(query),
            response=_summary(_extract_result_text(result)),
            cache="miss",
        )
        return response
    except HTTPException:
        raise
    except PerplexityError as exc:
        raise _upstream_error(str(exc)) from exc
    except Exception as exc:
        raise _internal_error("chat_completions", exc) from exc


@router.post(
    "/v1/completions",
    response_model=CompletionsResponse,
    tags=["Completions"],
    summary="Create a legacy text completion",
    description="OpenAI-compatible completions endpoint implemented as a shim over chat completions.",
    responses={401: {"description": "Invalid API key"}, 400: {"description": "Invalid model"}},
)
async def completions(request: Request):
    try:
        try:
            raw_body = await request.json()
        except Exception as exc:
            raise _invalid_request(f"Invalid request body: {exc}") from exc
        req = _normalize_completions_payload(raw_body)
        mode, model = resolve(req.model)
        query = _extract_prompt_text(req.prompt)
        cache_key = cache.make_key(
            query,
            req.model,
            request_type="completions",
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            stream=req.stream,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            stop=req.stop,
            logprobs=req.logprobs,
            top_logprobs=req.top_logprobs,
            echo=req.echo,
        )

        if req.stream:
            generator = await search(query, mode, model, stream=True)
            stream = completions_stream(generator, req.model, f"cmpl-{uuid.uuid4().hex}")
            _log_summary(
                "POST /v1/completions",
                200,
                model=req.model,
                mode=mode,
                stream=True,
                query=_summary(query),
                response="stream",
            )
            return StreamingResponse(await _prefetch_first_chunk(stream), media_type="text/event-stream")

        cached = await cache.get(cache_key)
        if cached is not None:
            response = CompletionsResponse.model_validate_json(cached)
            cached_text = response.choices[0].text if response.choices else ""
            _log_summary(
                "POST /v1/completions",
                200,
                model=req.model,
                mode=mode,
                stream=False,
                query=_summary(query),
                response=_summary(cached_text),
                cache="hit",
            )
            return response

        result = await search(query, mode, model, stream=False)
        response = _completions_response(req.model, result)
        await cache.set(cache_key, response.model_dump_json())
        _log_summary(
            "POST /v1/completions",
            200,
            model=req.model,
            mode=mode,
            stream=False,
            query=_summary(query),
            response=_summary(_extract_result_text(result)),
            cache="miss",
        )
        return response
    except HTTPException:
        raise
    except PerplexityError as exc:
        raise _upstream_error(str(exc)) from exc
    except Exception as exc:
        raise _internal_error("completions", exc) from exc


@router.post(
    "/v1/responses",
    response_model=ResponsesResponse,
    tags=["Responses"],
    summary="Create a response",
    description="OpenAI-compatible Responses API endpoint for text and streaming output.",
    responses={401: {"description": "Invalid API key"}, 400: {"description": "Invalid model"}},
)
async def responses(req: ResponsesRequest):
    try:
        mode, model = resolve(req.model)
        transcript_key = _continuation_transcript_key(req.input) if isinstance(req.input, list) else None
        follow_up = await _resolve_follow_up(
            response_id=req.previous_response_id,
            transcript_key=transcript_key,
        )
        query = _extract_responses_query(req.input)
        if req.instructions:
            query = f"{req.instructions}\n\n{query}"
        cache_key = cache.make_key(
            query,
            req.model,
            request_type="responses",
            temperature=req.temperature,
            stream=req.stream,
            instructions=req.instructions,
            previous_response_id=req.previous_response_id,
            tools=req.tools,
            tool_choice=req.tool_choice,
            parallel_tool_calls=req.parallel_tool_calls,
        )

        if req.stream:
            response_id = f"resp-{uuid.uuid4().hex}"
            generator = await search(query, mode, model, stream=True, follow_up=follow_up)
            stream = responses_stream(
                _wrap_stream_with_follow_up(
                    generator,
                    response_id=response_id,
                    transcript_key=None,
                    transcript_messages=req.input if isinstance(req.input, list) else None,
                ),
                req.model,
                response_id,
            )
            _log_summary(
                "POST /v1/responses",
                200,
                model=req.model,
                mode=mode,
                stream=True,
                query=_summary(query),
                response="stream",
            )
            return StreamingResponse(await _prefetch_first_chunk(stream), media_type="text/event-stream")

        cached = await cache.get(cache_key)
        if cached is not None:
            response = ResponsesResponse.model_validate_json(cached)
            cached_text = ""
            if response.output and response.output[0].content:
                cached_text = response.output[0].content[0].text
            _log_summary(
                "POST /v1/responses",
                200,
                model=req.model,
                mode=mode,
                stream=False,
                query=_summary(query),
                response=_summary(cached_text),
                cache="hit",
            )
            return response

        result = await search(query, mode, model, stream=False, follow_up=follow_up)
        response = _responses_response(req.model, result)
        await _store_follow_up(
            response_id=response.id,
            transcript_key=(
                _assistant_transcript_key(req.input, _extract_result_text(result))
                if isinstance(req.input, list)
                else None
            ),
            payload=result,
        )
        await cache.set(cache_key, response.model_dump_json())
        _log_summary(
            "POST /v1/responses",
            200,
            model=req.model,
            mode=mode,
            stream=False,
            query=_summary(query),
            response=_summary(_extract_result_text(result)),
            cache="miss",
        )
        return response
    except HTTPException:
        raise
    except PerplexityError as exc:
        raise _upstream_error(str(exc)) from exc
    except Exception as exc:
        raise _internal_error("responses", exc) from exc


__all__ = ["router"]

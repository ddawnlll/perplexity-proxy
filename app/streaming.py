from __future__ import annotations

import json
import time
from typing import AsyncGenerator


def _extract_text(chunk: object) -> str | None:
    if chunk is None:
        return None
    if isinstance(chunk, str):
        return chunk or None
    if isinstance(chunk, dict):
        for key in ("delta", "content", "text", "answer"):
            value = chunk.get(key)
            if isinstance(value, str) and value:
                return value
    return str(chunk) or None


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


async def chat_completions_stream(
    generator: AsyncGenerator,
    model: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """
    Yields OpenAI chat.completion.chunk SSE events from a Perplexity stream.
    Format: data: {json}\n\n
    Terminates with: data: [DONE]\n\n
    """
    created = int(time.time())
    emitted_any = False
    async for chunk in generator:
        text = _extract_text(chunk)
        if not text:
            continue
        delta = {"content": text}
        if not emitted_any:
            delta = {"role": "assistant", "content": text}
            emitted_any = True
        yield _sse(
            {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
    yield _sse(
        {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


async def responses_stream(
    generator: AsyncGenerator,
    model: str,
    resp_id: str,
) -> AsyncGenerator[str, None]:
    """
    Yields Responses API SSE events from a Perplexity stream.
    event types: response.output_text.delta, response.completed
    Format: data: {json}\n\n
    Terminates with: data: [DONE]\n\n
    """
    full_text = ""
    created_at = int(time.time())
    async for chunk in generator:
        text = _extract_text(chunk)
        if not text:
            continue
        full_text += text
        yield _sse(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }
        )
    yield _sse(
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": full_text,
        }
    )
    completed = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": full_text}],
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    yield _sse({"type": "response.completed", "response": completed})
    yield "data: [DONE]\n\n"


__all__ = ["chat_completions_stream", "responses_stream"]

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator


def _extract_text(chunk: object) -> str | None:
    """
    Extract displayable text from a Perplexity upstream chunk.

    Priority order:
    1. blocks[].markdown_block.chunks[] for ask_text_0_markdown or ask_text blocks
    2. Top-level delta/content/text/answer keys (legacy/fallback)
    3. None if no text found (never return the raw blob)
    """
    if chunk is None:
        return None

    if isinstance(chunk, dict):
        blocks = chunk.get("blocks")
        if isinstance(blocks, list):
            for preferred_usage in ("ask_text_0_markdown", "ask_text"):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("intended_usage") != preferred_usage:
                        continue
                    markdown_block = block.get("markdown_block")
                    if not isinstance(markdown_block, dict):
                        continue
                    chunks = markdown_block.get("chunks")
                    if isinstance(chunks, list) and chunks:
                        text = "".join(str(part) for part in chunks if part)
                        if text:
                            return text
            return None

        for key in ("delta", "content", "text", "answer"):
            value = chunk.get(key)
            if isinstance(value, str) and value:
                return value

        return None

    if isinstance(chunk, str):
        text = chunk.strip()
        if text.startswith("{'") or text.startswith('{"'):
            return None
        return chunk if chunk else None

    return None


def _snapshot_to_delta(snapshot: str | None, rendered_text: str) -> tuple[str | None, str]:
    """
    Convert upstream text into an incremental delta and next rendered state.

    Perplexity stream chunks frequently contain the full accumulated answer so far
    rather than only the newly appended token fragment. OpenAI-style SSE clients
    expect deltas, so we emit only the newly appended suffix when possible.
    """
    if not snapshot:
        return None, rendered_text

    if not rendered_text:
        return snapshot, snapshot

    if snapshot == rendered_text:
        return None, rendered_text

    if snapshot.startswith(rendered_text):
        return snapshot[len(rendered_text) :], snapshot

    return snapshot, rendered_text + snapshot


_INTERNAL_KEYS = frozenset(
    {
        "backend_uuid",
        "frontend_context_uuid",
        "classifier_results",
        "context_uuid",
        "read_write_token",
        "search_implementation_mode",
        "final_sse_message",
        "message_mode",
    }
)


def _is_internal_chunk(chunk) -> bool:
    """
    Returns True for Perplexity internal state blobs that must never
    reach the client. Handles both dict and string forms.
    """
    if isinstance(chunk, dict):
        blocks = chunk.get("blocks")
        if isinstance(blocks, list):
            for preferred_usage in ("ask_text_0_markdown", "ask_text"):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("intended_usage") != preferred_usage:
                        continue
                    markdown_block = block.get("markdown_block")
                    if not isinstance(markdown_block, dict):
                        continue
                    chunks = markdown_block.get("chunks")
                    if isinstance(chunks, list) and chunks:
                        text = "".join(str(part) for part in chunks if part)
                        if text:
                            return False
        # Most common form from perplexity_async — check keys directly.
        return bool(_INTERNAL_KEYS & chunk.keys()) or isinstance(blocks, list)
    if isinstance(chunk, str):
        s = chunk.strip()
        return (
            s.startswith("{'backend_uuid'")
            or s.startswith('{"backend_uuid"')
            or "'backend_uuid'" in s[:120]
            or '"backend_uuid"' in s[:120]
        )
    return False


_is_state_blob = _is_internal_chunk


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _roo_tool_call(text: str) -> list[dict]:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": "attempt_completion",
                "arguments": json.dumps({"result": text}),
            },
        }
    ]


async def chat_completions_stream(
    generator: AsyncGenerator,
    model: str,
    req_id: str,
    roo_mode: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Yields OpenAI chat.completion.chunk SSE events from a Perplexity stream.
    Format: data: {json}\n\n
    Terminates with: data: [DONE]\n\n
    """
    created = int(time.time())
    emitted_any = False
    rendered_text = ""
    async for chunk in generator:
        if _is_internal_chunk(chunk):
            continue
        snapshot = _extract_text(chunk)
        if not snapshot:
            continue
        if snapshot.strip().startswith("{'backend_uuid'") or snapshot.strip().startswith('{"backend_uuid"'):
            continue  # secondary guard — should never fire if the raw chunk filter is correct
        text, rendered_text = _snapshot_to_delta(snapshot, rendered_text)
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
    if roo_mode:
        yield _sse(
            {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": _roo_tool_call(rendered_text),
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )
    else:
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


async def completions_stream(
    generator: AsyncGenerator,
    model: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """Wrap chat completion SSE chunks in legacy completions format."""
    async for chunk in chat_completions_stream(generator, model, req_id):
        if chunk == "data: [DONE]\n\n":
            yield chunk
            continue
        if not chunk.startswith("data: "):
            yield chunk
            continue
        try:
            payload = json.loads(chunk.removeprefix("data: ").removesuffix("\n\n"))
        except Exception:
            yield chunk
            continue
        payload["object"] = "text_completion"
        for choice in payload.get("choices", []):
            delta = choice.pop("delta", None)
            text = ""
            if isinstance(delta, dict):
                text = str(delta.get("content") or "")
            choice["text"] = text
        yield _sse(payload)


async def responses_stream(
    generator: AsyncGenerator,
    model: str,
    resp_id: str,
) -> AsyncGenerator[str, None]:
    """
    Yields Responses API SSE events from a Perplexity stream.
    event types: response.created, response.output_text.delta, response.completed
    Format: data: {json}\n\n
    Terminates with: data: [DONE]\n\n
    """
    full_text = ""
    created_at = int(time.time())
    rendered_text = ""
    yield _sse(
        {
            "type": "response.created",
            "response": {
                "id": resp_id,
                "object": "response",
                "created_at": created_at,
                "status": "in_progress",
                "background": False,
                "error": None,
                "model": model,
                "output": [],
            },
        }
    )
    async for chunk in generator:
        if _is_internal_chunk(chunk):
            continue
        snapshot = _extract_text(chunk)
        if not snapshot:
            continue
        if snapshot.strip().startswith("{'backend_uuid'") or snapshot.strip().startswith('{"backend_uuid"'):
            continue  # secondary guard — should never fire if the raw chunk filter is correct
        text, rendered_text = _snapshot_to_delta(snapshot, rendered_text)
        full_text = rendered_text
        if not text:
            continue
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


__all__ = ["chat_completions_stream", "completions_stream", "responses_stream", "_is_internal_chunk", "_is_state_blob"]

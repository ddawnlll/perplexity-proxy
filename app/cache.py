from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict

from app.config import settings


class LRUCache:
    def __init__(self, max_size: int, ttl: int, enabled: bool):
        self._store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl = ttl
        self._enabled = enabled

    def make_key(
        self,
        query: str,
        model_name: str,
        *,
        request_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stream: bool = False,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        logprobs: int | None = None,
        top_logprobs: int | None = None,
        echo: bool | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: object | None = None,
        tool_choice: object | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> str:
        payload = {
            "echo": echo,
            "frequency_penalty": frequency_penalty,
            "instructions": instructions,
            "logprobs": logprobs,
            "max_tokens": max_tokens,
            "model_name": model_name,
            "presence_penalty": presence_penalty,
            "parallel_tool_calls": parallel_tool_calls,
            "previous_response_id": previous_response_id,
            "query": query,
            "request_type": request_type,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "top_logprobs": top_logprobs,
            "top_p": top_p,
            "tool_choice": tool_choice,
            "tools": tools,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, key: str) -> str | None:
        if not self._enabled:
            return None
        async with self._lock:
            if key not in self._store:
                return None
            value, ts = self._store[key]
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, key: str, value: str):
        if not self._enabled:
            return
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time())
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    async def clear(self):
        async with self._lock:
            self._store.clear()


cache = LRUCache(
    max_size=settings.CACHE_MAX_SIZE,
    ttl=settings.CACHE_TTL_SECONDS,
    enabled=settings.CACHE_ENABLED,
)


__all__ = ["LRUCache", "cache"]

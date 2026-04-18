from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


class FollowUpStateStore:
    def __init__(self, max_size: int = 1024, ttl_seconds: int = 3600):
        self._by_response_id: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()
        self._by_transcript: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def make_transcript_key(messages: list[dict[str, str]]) -> str:
        payload = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _prune_locked(self, store: OrderedDict[str, tuple[dict[str, Any], float]]) -> None:
        now = time.time()
        expired = [key for key, (_, ts) in store.items() if now - ts > self._ttl_seconds]
        for key in expired:
            store.pop(key, None)
        while len(store) > self._max_size:
            store.popitem(last=False)

    async def put(
        self,
        *,
        response_id: str | None,
        transcript_key: str | None,
        follow_up: dict[str, Any],
    ) -> None:
        async with self._lock:
            ts = time.time()
            if response_id:
                self._by_response_id[response_id] = (dict(follow_up), ts)
                self._by_response_id.move_to_end(response_id)
                self._prune_locked(self._by_response_id)
            if transcript_key:
                self._by_transcript[transcript_key] = (dict(follow_up), ts)
                self._by_transcript.move_to_end(transcript_key)
                self._prune_locked(self._by_transcript)

    async def get_by_response_id(self, response_id: str | None) -> dict[str, Any] | None:
        if not response_id:
            return None
        async with self._lock:
            entry = self._by_response_id.get(response_id)
            if entry is None:
                return None
            value, ts = entry
            if time.time() - ts > self._ttl_seconds:
                self._by_response_id.pop(response_id, None)
                return None
            self._by_response_id.move_to_end(response_id)
            return dict(value)

    async def get_by_transcript(self, transcript_key: str | None) -> dict[str, Any] | None:
        if not transcript_key:
            return None
        async with self._lock:
            entry = self._by_transcript.get(transcript_key)
            if entry is None:
                return None
            value, ts = entry
            if time.time() - ts > self._ttl_seconds:
                self._by_transcript.pop(transcript_key, None)
                return None
            self._by_transcript.move_to_end(transcript_key)
            return dict(value)

    async def clear(self) -> None:
        async with self._lock:
            self._by_response_id.clear()
            self._by_transcript.clear()


follow_up_store = FollowUpStateStore()


__all__ = ["FollowUpStateStore", "follow_up_store"]

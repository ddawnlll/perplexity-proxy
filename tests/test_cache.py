from __future__ import annotations

import pytest

import app.cache as cache_module
from app.cache import LRUCache


@pytest.mark.asyncio
async def test_cache_hit_returns_stored_value():
    cache = LRUCache(max_size=2, ttl=60, enabled=True)
    key = cache.make_key("Paris", "gpt-5.2")

    await cache.set(key, "France")

    assert await cache.get(key) == "France"


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    cache = LRUCache(max_size=2, ttl=60, enabled=True)

    assert await cache.get("missing") is None


@pytest.mark.asyncio
async def test_cache_ttl_expiry_returns_none(monkeypatch):
    cache = LRUCache(max_size=2, ttl=1, enabled=True)
    key = cache.make_key("Paris", "gpt-5.2")
    now = 1000.0

    monkeypatch.setattr(cache_module.time, "time", lambda: now)
    await cache.set(key, "France")

    monkeypatch.setattr(cache_module.time, "time", lambda: now + 2)
    assert await cache.get(key) is None


@pytest.mark.asyncio
async def test_cache_lru_eviction_removes_oldest():
    cache = LRUCache(max_size=2, ttl=60, enabled=True)
    key1 = cache.make_key("q1", "m1")
    key2 = cache.make_key("q2", "m2")
    key3 = cache.make_key("q3", "m3")

    await cache.set(key1, "v1")
    await cache.set(key2, "v2")
    await cache.set(key3, "v3")

    assert await cache.get(key1) is None
    assert await cache.get(key2) == "v2"
    assert await cache.get(key3) == "v3"


@pytest.mark.asyncio
async def test_cache_disabled_never_stores_or_returns_values():
    cache = LRUCache(max_size=2, ttl=60, enabled=False)
    key = cache.make_key("Paris", "gpt-5.2")

    await cache.set(key, "France")

    assert await cache.get(key) is None


@pytest.mark.asyncio
async def test_make_key_is_deterministic():
    cache = LRUCache(max_size=2, ttl=60, enabled=True)

    key1 = cache.make_key("query", "model")
    key2 = cache.make_key("query", "model")

    assert key1 == key2


@pytest.mark.asyncio
async def test_clear_empties_cache():
    cache = LRUCache(max_size=2, ttl=60, enabled=True)
    key = cache.make_key("Paris", "gpt-5.2")

    await cache.set(key, "France")
    await cache.clear()

    assert await cache.get(key) is None

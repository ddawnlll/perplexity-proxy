from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatResponse, ResponsesResponse
from perplexity.exceptions import AuthenticationError, RateLimitError


@pytest.fixture
def client(mocker):
    mocker.patch("app.main.init_client", new=AsyncMock(return_value=None))
    mocker.patch("app.main.close_client", new=AsyncMock(return_value=None))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def cache_mocks(mocker):
    mocker.patch("app.router.cache.get", new=AsyncMock(return_value=None))
    mocker.patch("app.router.cache.set", new=AsyncMock(return_value=None))


@pytest.fixture
def search_mock(mocker):
    return mocker.patch("app.router.search", new=AsyncMock())


async def _stream_gen():
    yield "Hello"
    yield " world"


def test_get_v1_models_returns_list(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)


def test_get_health_returns_expected_fields(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"status", "cache_enabled", "authenticated", "model_count"}
    assert payload["status"] == "ok"


def test_chat_completions_success_returns_message_content(client, cache_mocks, search_mock):
    search_mock.return_value = "Paris is the capital of France."

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Paris is the capital of France."


def test_responses_success_string_input_returns_text_output(client, cache_mocks, search_mock):
    search_mock.return_value = "Paris is the capital of France."

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.2", "input": "What is the capital of France?", "stream": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "Paris is the capital of France."


def test_responses_success_list_input_returns_text_output(client, cache_mocks, search_mock):
    search_mock.return_value = "Paris is the capital of France."

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.2",
            "input": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "Paris is the capital of France."


def test_chat_unknown_model_returns_400(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "unknown-model", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 400


def test_responses_unknown_model_returns_400(client):
    response = client.post(
        "/v1/responses",
        json={"model": "unknown-model", "input": "Hi"},
    )

    assert response.status_code == 400


def test_chat_stream_returns_event_stream(client, cache_mocks, search_mock):
    search_mock.return_value = _stream_gen()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_responses_stream_returns_event_stream(client, cache_mocks, search_mock):
    search_mock.return_value = _stream_gen()

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.2", "input": "Hello", "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_cache_hit_skips_search(client, mocker):
    cached_chat = ChatResponse(
        id="chatcmpl-cached",
        object="chat.completion",
        created=1,
        model="gpt-5.2",
        choices=[],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    ).model_dump_json()
    get_mock = mocker.patch("app.router.cache.get", new=AsyncMock(return_value=cached_chat))
    set_mock = mocker.patch("app.router.cache.set", new=AsyncMock(return_value=None))
    search_mock = mocker.patch("app.router.search", new=AsyncMock())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    search_mock.assert_not_called()
    get_mock.assert_awaited()
    set_mock.assert_not_awaited()


def test_authentication_error_maps_to_401(client, mocker):
    class FakeClient:
        async def search(self, *args, **kwargs):
            raise AuthenticationError("bad auth")

    mocker.patch("app.client.get_client", return_value=FakeClient())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.2", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 401


def test_rate_limit_error_maps_to_429(client, mocker):
    class FakeClient:
        async def search(self, *args, **kwargs):
            raise RateLimitError("slow down")

    mocker.patch("app.client.get_client", return_value=FakeClient())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.2", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 429


def test_router_has_no_double_prefixed_routes():
    paths = [route.path for route in app.router.routes]
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/responses" in paths
    assert "/health" in paths
    assert not any(path.startswith("/v1/v1/") for path in paths)

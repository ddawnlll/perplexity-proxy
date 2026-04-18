from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import app.client as client_module
from perplexity.exceptions import NetworkError


@pytest.mark.asyncio
async def test_search_retries_network_error_then_succeeds(monkeypatch, mocker):
    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def search(self, *args, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise NetworkError("temporary network failure")
            return "ok"

    fake_client = FakeClient()
    monkeypatch.setattr(client_module, "_client", fake_client)
    sleep_mock = mocker.patch("app.client.asyncio.sleep", new=AsyncMock(return_value=None))

    result = await client_module.search("query", "auto", None, stream=False)

    assert result == "ok"
    assert fake_client.calls == 3
    assert sleep_mock.await_count == 2


@pytest.mark.asyncio
async def test_search_network_error_exhausts_retries_and_raises_503(monkeypatch, mocker):
    class FakeClient:
        async def search(self, *args, **kwargs):
            raise NetworkError("temporary network failure")

    monkeypatch.setattr(client_module, "_client", FakeClient())
    mocker.patch("app.client.asyncio.sleep", new=AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as exc_info:
        await client_module.search("query", "auto", None, stream=False)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Upstream unavailable"

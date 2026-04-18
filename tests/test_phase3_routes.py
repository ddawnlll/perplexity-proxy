from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app import mapper
from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def patch_lifespan(monkeypatch, mocker):
    original_map = dict(mapper.MODEL_MAP)
    mocker.patch("app.main.init_client", new=AsyncMock(return_value=None))
    mocker.patch("app.main.close_client", new=AsyncMock(return_value=None))
    yield
    mapper.MODEL_MAP = original_map


def _test_client() -> TestClient:
    return TestClient(app)


def test_refresh_returns_401_with_no_secret_configured(monkeypatch, mocker):
    monkeypatch.setattr(settings, "REFRESH_SECRET", "")
    client_mock = mocker.Mock()
    client_mock.refresh_models = AsyncMock(return_value=True)
    mocker.patch("app.router.get_client", return_value=client_mock)

    with _test_client() as client:
        response = client.post("/v1/models/refresh", headers={"Authorization": "Bearer anything"})

    assert response.status_code == 401


def test_refresh_returns_401_with_wrong_token(monkeypatch, mocker):
    monkeypatch.setattr(settings, "REFRESH_SECRET", "abc")
    client_mock = mocker.Mock()
    client_mock.refresh_models = AsyncMock(return_value=True)
    mocker.patch("app.router.get_client", return_value=client_mock)

    with _test_client() as client:
        response = client.post("/v1/models/refresh", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_refresh_returns_401_with_no_authorization_header(monkeypatch, mocker):
    monkeypatch.setattr(settings, "REFRESH_SECRET", "abc")
    client_mock = mocker.Mock()
    client_mock.refresh_models = AsyncMock(return_value=True)
    mocker.patch("app.router.get_client", return_value=client_mock)

    with _test_client() as client:
        response = client.post("/v1/models/refresh")

    assert response.status_code in {401, 422}


def test_refresh_returns_503_when_refresh_models_fails(monkeypatch, mocker):
    monkeypatch.setattr(settings, "REFRESH_SECRET", "abc")
    client_mock = mocker.Mock()
    client_mock.refresh_models = AsyncMock(return_value=False)
    mocker.patch("app.router.get_client", return_value=client_mock)

    with _test_client() as client:
        response = client.post("/v1/models/refresh", headers={"Authorization": "Bearer abc"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model refresh failed — static map still active"


def test_refresh_returns_200_on_success(monkeypatch, mocker):
    monkeypatch.setattr(settings, "REFRESH_SECRET", "abc")
    client_mock = mocker.Mock()
    client_mock.refresh_models = AsyncMock(return_value=True)
    mocker.patch("app.router.get_client", return_value=client_mock)

    updated_map = {"auto": {None: "turbo"}, "pro": {None: "pplx_pro", "gpt-5.4": "gpt54"}}
    build_mock = mocker.patch("app.router.mapper.build_model_map", return_value=updated_map)

    with _test_client() as client:
        response = client.post("/v1/models/refresh", headers={"Authorization": "Bearer abc"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_count"] == len(updated_map)
    assert payload["models"] == list(updated_map.keys())
    build_mock.assert_called_once()


def test_get_models_after_refresh_returns_updated_list(monkeypatch, mocker):
    monkeypatch.setattr(settings, "REFRESH_SECRET", "abc")
    client_mock = mocker.Mock()
    client_mock.refresh_models = AsyncMock(return_value=True)
    mocker.patch("app.router.get_client", return_value=client_mock)

    original_map = dict(mapper.MODEL_MAP)
    updated_map = dict(original_map)
    updated_map["gpt-5.4"] = {"mode": "pro", "model": "gpt-5.4"}
    mocker.patch("app.router.mapper.build_model_map", return_value=updated_map)

    with _test_client() as client:
        refresh_response = client.post("/v1/models/refresh", headers={"Authorization": "Bearer abc"})
        assert refresh_response.status_code == 200

        models_response = client.get("/v1/models")

    assert models_response.status_code == 200
    payload = models_response.json()
    assert "gpt-5.4" in [entry["id"] for entry in payload["data"]]

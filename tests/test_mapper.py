from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.mapper import MODEL_MAP, get_model_list, resolve
from perplexity.models import MODEL_PREFERENCE_MAP


def test_model_map_entry_count_matches_source_map():
    expected = sum(len(models) for models in MODEL_PREFERENCE_MAP.values())
    assert len(MODEL_MAP) == expected


def test_resolve_auto_default():
    assert resolve("auto") == ("auto", None)


def test_resolve_sonar():
    assert resolve("sonar") == ("pro", "sonar")


def test_resolve_gpt_52():
    assert resolve("gpt-5.2") == ("pro", "gpt-5.2")


def test_resolve_deep_research_default():
    assert resolve("deep-research") == ("deep research", None)


def test_resolve_reasoning_default():
    assert resolve("reasoning") == ("reasoning", None)


def test_resolve_unknown_model_raises_http_400():
    with pytest.raises(HTTPException) as exc_info:
        resolve("unknown-model")
    assert exc_info.value.status_code == 400


def test_get_model_list_shape_and_length():
    model_list = get_model_list()
    assert len(model_list) == len(MODEL_MAP)
    for entry in model_list:
        assert set(entry.keys()) == {"id", "object", "created", "owned_by"}
        assert entry["object"] == "model"
        assert entry["owned_by"] == "perplexity"
        assert isinstance(entry["created"], int)


def test_get_model_list_ids_match_model_map_keys():
    model_list = get_model_list()
    assert [entry["id"] for entry in model_list] == list(MODEL_MAP.keys())

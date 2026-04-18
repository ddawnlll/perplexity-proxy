from perplexity.models import MODEL_PREFERENCE_MAP
from fastapi import HTTPException
import time


def build_model_map() -> dict:
    result = {}
    for mode, models in MODEL_PREFERENCE_MAP.items():
        for model in models:
            proxy_name = model if model is not None else mode.replace(" ", "-")
            result[proxy_name] = {"mode": mode, "model": model}
    return result


MODEL_MAP = build_model_map()


def resolve(model_name: str) -> tuple[str, str | None]:
    entry = MODEL_MAP.get(model_name)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model_name!r}. Available: {list(MODEL_MAP.keys())}"
        )
    return entry["mode"], entry["model"]


def get_model_list() -> list[dict]:
    return [
        {"id": name, "object": "model", "created": int(time.time()), "owned_by": "perplexity"}
        for name in MODEL_MAP.keys()
    ]

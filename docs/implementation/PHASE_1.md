# Phase 1 — perplexity-ai Library Refactor (Status: Planned)

**Status:** Planned
**Owner:** perplexity-ai core
**Last updated:** 2026-04-17
**Delivery status:** Not started

---

## 1. Purpose

The `perplexity-ai` library currently hardcodes all model definitions and internal API identifiers directly inside `client.py` in two separate locations. This phase extracts those definitions into a dedicated `perplexity/models.py` module and exposes a public API for dynamic model discovery — unblocking downstream consumers such as `perplexity-proxy` from having to maintain their own static model lists.

This phase is NOT about fetching models from the Perplexity network at runtime. That is a separate concern tracked in the GitHub issue titled *"Dynamic model fetching from Perplexity instead of hardcoded model list"* and is deferred to a future phase once the endpoint is identified and verified.

---

## 2. What Carried Over / What Must Stay Stable

The following are already implemented and must remain stable:

- [x] `Client.__init__()` session initialization and cookie injection
- [x] `Client.search()` public API signature — all parameters unchanged
- [x] `Client.create_account()` Emailnator flow
- [x] SSE streaming response parsing and `stream_response()` generator
- [x] File upload flow via `CurlMime`
- [x] `perplexity_async.Client` — mirrors sync client; must receive the same changes

This phase builds on top of these. Do not regress them.

---

## 3. Background & Motivation

Two locations inside `client.py` today contain all model knowledge:

**Location 1 — validation assert (~line 60):**
```python
assert model in {
    "auto": [None],
    "pro": [None, "sonar", "gpt-5.2", "claude-4.5-sonnet", "grok-4.1"],
    "reasoning": [None, "gpt-5.2-thinking", ...],
    "deep research": [None],
}[mode]
```

**Location 2 — internal API ID mapping (~line 120):**
```python
"model_preference": {
    "auto":          {None: "turbo"},
    "pro":           {None: "pplx_pro", "sonar": "experimental", "gpt-5.2": "gpt52", ...},
    "reasoning":     {None: "pplx_reasoning", "gpt-5.2-thinking": "gpt52_thinking", ...},
    "deep research": {None: "pplx_alpha"},
}[mode][model]
```

These two structures are tightly coupled but live separately — creating a silent drift risk when one is updated and the other is not. Extracting them into a single source of truth eliminates the coupling and enables any downstream tool to import model metadata without instantiating a `Client`.

---

## 4. Current Failure State / Known Blockers

- `MODEL_PREFERENCE_MAP` = not exported — downstream tools cannot access internal API IDs without importing `client.py` and instantiating a full session
- `AVAILABLE_MODELS` = not exported — no programmatic way to list valid models without reading source code
- `client.py` = two independent hardcoded structures that must be kept manually in sync
- `perplexity_async/client.py` = same duplication — changes to sync client must be manually mirrored

---

## 5. Workstream A — Create `perplexity/models.py`

**Status:** New

### Problem / Goal

All model definitions and internal Perplexity API identifiers live inside `client.py` with no public access point. A dedicated `models.py` module must own this data and expose it cleanly.

### Implementation Tasks

- [ ] Create `perplexity/models.py`
- [ ] Define `MODEL_PREFERENCE_MAP` — full `mode → {model → internal_api_id}` dict
- [ ] Derive `AVAILABLE_MODELS` from `MODEL_PREFERENCE_MAP` keys (do not duplicate by hand)
- [ ] Implement `get_available_models() -> dict` — returns `AVAILABLE_MODELS`
- [ ] Implement `list_flat_models() -> list[str]` — flattens all models; `None` entries become hyphenated mode names (e.g. `"deep research"` → `"deep-research"`)
- [ ] Implement `resolve_preference(mode: str, model) -> str` — returns the internal Perplexity API ID; raises `ValueError` for unknown combinations
- [ ] Export all public symbols from `perplexity/__init__.py`

### Configuration / Code Reference

```python
# perplexity/models.py

MODEL_PREFERENCE_MAP = {
    "auto": {
        None: "turbo",
    },
    "pro": {
        None: "pplx_pro",
        "sonar": "experimental",
        "gpt-5.2": "gpt52",
        "claude-4.5-sonnet": "claude45sonnet",
        "grok-4.1": "grok41nonreasoning",
    },
    "reasoning": {
        None: "pplx_reasoning",
        "gpt-5.2-thinking": "gpt52_thinking",
        "claude-4.5-sonnet-thinking": "claude45sonnetthinking",
        "gemini-3.0-pro": "gemini30pro",
        "kimi-k2-thinking": "kimik2thinking",
        "grok-4.1-reasoning": "grok41reasoning",
    },
    "deep research": {
        None: "pplx_alpha",
    },
}

# Derived automatically — never edit this manually
AVAILABLE_MODELS = {
    mode: list(models.keys())
    for mode, models in MODEL_PREFERENCE_MAP.items()
}

def get_available_models() -> dict:
    """Returns the full mode → valid model list mapping."""
    return AVAILABLE_MODELS

def list_flat_models() -> list[str]:
    """
    Returns a flat list of all proxy-friendly model name strings.
    None entries (mode defaults) are represented by their mode name
    with spaces replaced by hyphens.
    """
    result = []
    for mode, models in AVAILABLE_MODELS.items():
        for model in models:
            result.append(model if model is not None else mode.replace(" ", "-"))
    return result

def resolve_preference(mode: str, model) -> str:
    """
    Returns the Perplexity internal API model ID for a given
    mode + model combination. Raises ValueError for unknown pairs.
    """
    try:
        return MODEL_PREFERENCE_MAP[mode][model]
    except KeyError:
        raise ValueError(f"Invalid mode/model combination: mode={mode!r}, model={model!r}")
```

### Acceptance Criteria

- [ ] `perplexity/models.py` exists and is importable as `from perplexity.models import ...`
- [ ] `AVAILABLE_MODELS` is derived from `MODEL_PREFERENCE_MAP` — no manual duplication
- [ ] `list_flat_models()` returns a flat `list[str]` with no `None` values
- [ ] `resolve_preference("pro", "gpt-5.2")` returns `"gpt52"`
- [ ] `resolve_preference("auto", None)` returns `"turbo"`
- [ ] `resolve_preference("pro", "nonexistent")` raises `ValueError`
- [ ] All symbols exported from `perplexity/__init__.py`

---

## 6. Workstream B — Refactor `perplexity/client.py`

**Status:** New

### Problem / Goal

Replace the two hardcoded model structures in `client.py` with imports from `perplexity.models`. The public `search()` API must not change.

### Implementation Tasks

- [ ] Add `from .models import AVAILABLE_MODELS, resolve_preference` import
- [ ] Replace the validation `assert` dict with `AVAILABLE_MODELS[mode]` lookup
- [ ] Replace the `model_preference` inline dict with a call to `resolve_preference(mode, model)`
- [ ] Add `available_models(self) -> dict` method to `Client` — returns `get_available_models()`
- [ ] Remove all now-redundant inline model dicts from the file

### Configuration / Code Reference

```python
# perplexity/client.py — diff summary

# ADD at top of file
from .models import AVAILABLE_MODELS, resolve_preference, get_available_models

# REPLACE validation assert
assert (
    model in AVAILABLE_MODELS[mode] if self.own else True
), "Invalid model for the selected mode."

# REPLACE model_preference inline dict (inside json_data)
"model_preference": resolve_preference(mode, model),

# ADD new method to Client class
def available_models(self) -> dict:
    """Returns available modes and their supported models."""
    return get_available_models()
```

### Acceptance Criteria

- [ ] `client.py` contains no inline model dicts
- [ ] `Client("...").search("query", mode="pro", model="gpt-5.2")` still works correctly
- [ ] `Client().available_models()` returns the full `AVAILABLE_MODELS` dict
- [ ] `Client().search("query", mode="pro", model="invalid")` raises `AssertionError`
- [ ] No change to any other `search()` parameter behavior

---

## 7. Workstream C — Mirror Changes in `perplexity_async/client.py`

**Status:** New

### Problem / Goal

The async client is a parallel implementation of the sync client. All model-related changes from Workstream B must be applied identically to `perplexity_async/client.py`.

### Implementation Tasks

- [ ] Add `from perplexity.models import AVAILABLE_MODELS, resolve_preference, get_available_models` import
- [ ] Replace validation assert with `AVAILABLE_MODELS[mode]` lookup
- [ ] Replace `model_preference` inline dict with `resolve_preference(mode, model)`
- [ ] Add `async def available_models(self) -> dict` method (or sync — match existing client style)
- [ ] Remove all redundant inline model dicts

### Acceptance Criteria

- [ ] `perplexity_async/client.py` contains no inline model dicts
- [ ] Async client behavior is identical to sync client for all model/mode combinations
- [ ] `await client.available_models()` (or `client.available_models()`) returns the correct dict

---

## 8. Workstream D — Test Coverage

**Status:** New
**Required before:** merging Phase 1 PR

### 8.1 Unit tests — `perplexity/models.py`

- [ ] `AVAILABLE_MODELS` keys match `MODEL_PREFERENCE_MAP` keys exactly
- [ ] `AVAILABLE_MODELS` values are lists derived from dict keys (not hardcoded)
- [ ] `list_flat_models()` contains no `None` values
- [ ] `list_flat_models()` maps `None` entries to hyphenated mode name strings
- [ ] `resolve_preference("auto", None)` returns `"turbo"`
- [ ] `resolve_preference("pro", "gpt-5.2")` returns `"gpt52"`
- [ ] `resolve_preference("reasoning", None)` returns `"pplx_reasoning"`
- [ ] `resolve_preference("deep research", None)` returns `"pplx_alpha"`
- [ ] `resolve_preference("pro", "nonexistent")` raises `ValueError`
- [ ] `resolve_preference("invalid_mode", None)` raises `ValueError`

### 8.2 Regression tests — `client.py` behavior

- [ ] `search()` with valid mode/model combination produces same `model_preference` value as before refactor
- [ ] `search()` with invalid model raises `AssertionError` (same as before)
- [ ] `available_models()` return value matches `AVAILABLE_MODELS`

### 8.3 Regression tests — `perplexity_async/client.py`

- [ ] Same regression tests as 8.2 applied to async client

---

## 9. Workstream E — Pre-Merge Audit Checklist

**Status:** New
**Must complete before:** merging Phase 1 PR into main

### 9.1 API surface audit

- [ ] `search()` signature is unchanged — no new required parameters
- [ ] `create_account()` signature is unchanged
- [ ] All previously passing tests still pass after refactor

### 9.2 Import hygiene

- [ ] No circular imports introduced between `models.py` and `client.py`
- [ ] `perplexity/__init__.py` exports are additive only — no removals

### 9.3 Sync/async parity

- [ ] Every change applied to `client.py` is also applied to `perplexity_async/client.py`
- [ ] Both clients return identical `available_models()` output

---

## 10. Combined Implementation Order

1. Complete Workstream A — create and test `perplexity/models.py`
2. Complete Workstream B — refactor `perplexity/client.py`
3. Complete Workstream C — mirror changes in `perplexity_async/client.py`
4. Run Workstream E — pre-merge audit checklist
5. Complete Workstream D — full test coverage
6. Open PR and verify CI passes
7. Evaluate all acceptance criteria before merge

### Acceptance Criteria for First Combined Run

- All existing tests pass without modification
- `from perplexity.models import AVAILABLE_MODELS, list_flat_models` works in isolation
- `resolve_preference` is the single source of truth for all internal API IDs
- No inline model dicts remain in `client.py` or `perplexity_async/client.py`

---

## 11. Definition of Done

Phase 1 is complete when **all** of the following are true simultaneously.

### 11.1 Library layer

- [x] `client.py` sync implementation exists and is stable
- [x] `perplexity_async/client.py` async implementation exists and is stable
- [ ] `perplexity/models.py` created with `MODEL_PREFERENCE_MAP`, `AVAILABLE_MODELS`, `get_available_models()`, `list_flat_models()`, `resolve_preference()`
- [ ] No inline model dicts remain in `client.py`
- [ ] No inline model dicts remain in `perplexity_async/client.py`

### 11.2 Public API layer

- [ ] `Client.available_models()` exposed on sync client
- [ ] `Client.available_models()` exposed on async client
- [ ] All new symbols exported from `perplexity/__init__.py`

### 11.3 Test layer

- [ ] All Workstream D tests implemented and passing
- [ ] No regressions in existing test suite
- [ ] `resolve_preference` tested for all valid combinations and known invalid inputs

### 11.4 Quality layer

- [ ] No circular imports
- [ ] `mypy` passes on `perplexity/models.py`
- [ ] Pre-merge audit checklist (Workstream E) fully completed

---

## 12. What Phase 2 Inherits

### 12.1 Capabilities unlocked by Phase 1

- `perplexity-proxy` can import `AVAILABLE_MODELS` and `list_flat_models()` directly — no proxy-side model list to maintain
- `resolve_preference()` is available as a stable import for the proxy's `mapper.py`
- Adding a new Perplexity model in the future requires editing only `MODEL_PREFERENCE_MAP` in one file
- Foundation is in place for Phase 3's runtime model fetching feature (the GitHub issue)

### 12.2 Phase Boundary

- Phase 2 is infrastructure work — building the `perplexity-proxy` FastAPI server on top of the refactored library.
- Phase 1 is the prerequisite.
- Do not start Phase 2 work until Phase 1 definition of done is fully satisfied.

---

## 13. Compact Mental Model

### 13.1 Phase Relationships

- Phase 1: Extract hardcoded model definitions from `client.py` into `perplexity/models.py` — single source of truth, public API for model discovery
- Phase 2: Build `perplexity-proxy` — OpenAI-compatible FastAPI server using the refactored library
- Phase 3: Runtime model fetching — replace static `MODEL_PREFERENCE_MAP` with live data from Perplexity

### 13.2 Key Takeaway

Phase 1 is a pure refactor — zero behavior change, zero new network calls, zero new dependencies. Its sole purpose is to make the model data importable and testable in isolation so that Phase 2 can build a self-configuring proxy without maintaining a duplicate model registry. Skipping Phase 1 and going straight to Phase 2 would mean the proxy maintains its own hardcoded model list that diverges from the library over time.

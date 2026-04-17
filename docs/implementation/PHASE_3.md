# Phase 3 — Dynamic Model Fetching (Status: Planned)

**Status:** Planned
**Owner:** perplexity-ai core + perplexity-proxy
**Last updated:** 2026-04-17
**Delivery status:** Not started

---

## 1. Purpose

Phase 2 ships a fully working proxy with a static `MODEL_PREFERENCE_MAP` hardcoded in `perplexity/models.py`. This works correctly today but creates a maintenance burden: every time Perplexity silently adds, renames, or removes a model (e.g. GPT-5.4, Claude Sonnet 4.6), a human must notice, verify the internal API ID, and push a code change.

Phase 3 eliminates that burden by reverse-engineering the endpoint Perplexity's web app uses to discover available models at runtime, fetching that data during `perplexity_async.Client` startup, and merging it with the static map as a fallback. The static map is never removed — it is the safety net when the live endpoint is unreachable.

This phase was deferred from Phase 1 because the live endpoint had not yet been identified. Phase 3 begins with that discovery step.

---

## 2. What Carried Over / What Must Stay Stable

The following are already implemented and must remain stable:

- [x] `perplexity/models.py` — `MODEL_PREFERENCE_MAP`, `AVAILABLE_MODELS`, `resolve_preference()`, `list_flat_models()`
- [x] `perplexity_async.Client.search()` — public API signature unchanged
- [x] `perplexity-proxy` — `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/responses`
- [x] Phase 1 and Phase 2 Definitions of Done fully satisfied

This phase adds on top of these. The static map must continue to work as fallback. No proxy endpoint signatures change.

---

## 3. Background & Motivation

### The Drift Problem

Perplexity regularly ships new models without announcement. The internal API IDs (e.g. `gpt52`, `claude45sonnet`) are not documented. When a new model appears in the Perplexity UI, the proxy does not know about it until someone manually:

1. Opens Perplexity in a browser
2. Inspects the network request for the new model
3. Extracts the internal API ID from the SSE payload
4. Updates `MODEL_PREFERENCE_MAP` in `models.py`
5. Deploys a new version of both libraries

This is fragile and slow. Dynamic fetching closes the loop.

### How Perplexity Loads Its Own Model List

Perplexity's React frontend fetches available models from its own API before rendering the model selector. Based on network inspection, the candidate endpoints are:

```
GET https://www.perplexity.ai/api/list-models
GET https://www.perplexity.ai/api/auth/session  (contains model flags in user object)
GET https://www.perplexity.ai/p/api/v1/models   (speculative)
```

The exact endpoint and response shape must be confirmed during Workstream A (discovery). Until confirmed, the static map remains the source of truth.

### Merge Strategy

Dynamic data augments the static map — it does not replace it. The merge rule is:

- Known model (exists in static map): static internal ID wins — static map has been human-verified
- New model (not in static map): dynamic data is accepted if it contains a valid internal ID
- Dynamic fetch fails: static map used entirely, no error surfaced to callers

---

## 4. Current Failure State / Known Blockers

- Live model endpoint = not yet identified — Workstream A is a discovery/research task
- Internal ID format for new models = unknown until endpoint is found
- `perplexity_async.Client` = no model refresh capability
- `perplexity/models.py` = no mechanism to accept runtime-sourced model data
- `perplexity-proxy` `/v1/models` = returns static list only

---

## 5. Workstream A — Endpoint Discovery (Research Task)

**Status:** New — must complete before any code is written

### Problem / Goal

Identify the exact URL, headers, authentication requirements, and response shape of the endpoint Perplexity's web app uses to populate its model selector.

### Discovery Steps

1. Open `https://www.perplexity.ai` in a browser with DevTools Network tab open
2. Filter requests to `perplexity.ai` domain, type: `Fetch/XHR`
3. Reload the page and look for requests that return a list of model objects
4. Inspect the request (headers, cookies required?) and response (JSON shape)
5. Test the endpoint with `curl` using session cookies to confirm reproducibility
6. Test the endpoint without cookies (anonymous) to determine if auth is required
7. Identify whether the response contains internal API IDs (like `gpt52`) or only display names

### Candidate Endpoint Shapes to Look For

```json
// Hypothetical response shape — confirm actual shape during discovery
{
  "models": [
    {
      "id": "gpt52",               // internal API ID — this is what we need
      "name": "GPT-5.2",           // display name
      "mode": "pro",               // or "reasoning", "auto", "deep research"
      "requires_pro": true
    }
  ]
}
```

### Decision Tree After Discovery

```
Endpoint found AND contains internal IDs?
  → Proceed with full Workstream B–F implementation

Endpoint found BUT only contains display names (no internal IDs)?
  → Implement partial fetching: update display names only, keep static internal IDs
  → Mark internal ID discovery as a separate future task

Endpoint NOT found after exhaustive inspection?
  → Phase 3 is deferred
  → Document findings in a GitHub issue
  → Proceed directly to Phase 4 with static map
```

### Acceptance Criteria

- [ ] Candidate endpoint URL documented
- [ ] Response JSON shape documented with example payload
- [ ] Authentication requirement documented (cookies required / optional / not needed)
- [ ] Internal API ID presence confirmed (yes/no)
- [ ] Decision tree outcome selected
- [ ] Discovery findings committed to `docs/model-discovery.md` in the repo

---

## 6. Workstream B — `perplexity/models.py` Runtime Extension

**Status:** Blocked on Workstream A

### Problem / Goal

Extend `perplexity/models.py` with a `ModelRegistry` class that holds the merged (static + dynamic) model map and exposes a refresh mechanism. All existing exports (`MODEL_PREFERENCE_MAP`, `AVAILABLE_MODELS`, `resolve_preference`, `list_flat_models`) must continue to work identically — they delegate to the registry.

### Implementation Tasks

- [ ] Define `ModelRegistry` class with internal `_map: dict` initialized from static `_STATIC_MAP`
- [ ] Implement `registry.resolve(mode, model) -> str` — same contract as current `resolve_preference()`
- [ ] Implement `registry.list_flat() -> list[str]` — same contract as current `list_flat_models()`
- [ ] Implement `registry.available() -> dict` — same contract as current `get_available_models()`
- [ ] Implement `registry.merge(dynamic: dict)` — merges dynamic data into `_map` using the merge strategy defined in Section 3
- [ ] Implement `registry.reset()` — resets `_map` to static baseline (for testing)
- [ ] Keep `MODEL_PREFERENCE_MAP`, `AVAILABLE_MODELS`, `resolve_preference`, `list_flat_models`, `get_available_models` as module-level aliases delegating to the registry — no callers break
- [ ] All registry operations must be thread-safe via `threading.RLock`

### Configuration / Code Reference

```python
# perplexity/models.py — ModelRegistry addition

import threading

_STATIC_MAP = {
    "auto": {None: "turbo"},
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
    "deep research": {None: "pplx_alpha"},
}

class ModelRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._map: dict = self._deep_copy(_STATIC_MAP)

    def _deep_copy(self, d: dict) -> dict:
        return {k: dict(v) for k, v in d.items()}

    def resolve(self, mode: str, model) -> str:
        with self._lock:
            try:
                return self._map[mode][model]
            except KeyError:
                raise ValueError(
                    f"Invalid mode/model combination: mode={mode!r}, model={model!r}"
                )

    def list_flat(self) -> list[str]:
        with self._lock:
            result = []
            for mode, models in self._map.items():
                for model in models:
                    result.append(model if model is not None else mode.replace(" ", "-"))
            return result

    def available(self) -> dict:
        with self._lock:
            return {mode: list(models.keys()) for mode, models in self._map.items()}

    def merge(self, dynamic: dict):
        """
        Merge dynamic model data into the registry.
        dynamic shape: {mode: {model_name: internal_api_id}}
        Static entries always win on conflict.
        """
        with self._lock:
            for mode, models in dynamic.items():
                if mode not in self._map:
                    self._map[mode] = {}
                for model_name, internal_id in models.items():
                    if model_name not in self._map[mode]:
                        self._map[mode][model_name] = internal_id

    def reset(self):
        with self._lock:
            self._map = self._deep_copy(_STATIC_MAP)

_registry = ModelRegistry()

# Backward-compatible module-level aliases
MODEL_PREFERENCE_MAP = _STATIC_MAP  # static reference, unchanged
AVAILABLE_MODELS = _registry.available()

def get_available_models() -> dict:
    return _registry.available()

def list_flat_models() -> list[str]:
    return _registry.list_flat()

def resolve_preference(mode: str, model) -> str:
    return _registry.resolve(mode, model)
```

### Acceptance Criteria

- [ ] All existing Phase 1 tests pass without modification
- [ ] `_registry.merge({"pro": {"gpt-5.4": "gpt54"}})` adds the new model
- [ ] `resolve_preference("pro", "gpt-5.4")` returns `"gpt54"` after merge
- [ ] `resolve_preference("pro", "gpt-5.2")` still returns `"gpt52"` after merge (static wins)
- [ ] `_registry.reset()` restores static baseline
- [ ] Concurrent `resolve()` and `merge()` calls do not deadlock or corrupt state
- [ ] `MODEL_PREFERENCE_MAP` module-level reference unchanged for any code that imports it directly

---

## 7. Workstream C — Fetcher (`perplexity/model_fetcher.py`)

**Status:** Blocked on Workstream A (endpoint shape required)

### Problem / Goal

Implement the async function that calls Perplexity's live model endpoint, parses the response, and returns a dict in the `ModelRegistry.merge()` expected shape.

### Implementation Tasks

- [ ] Create `perplexity/model_fetcher.py`
- [ ] Implement `async fetch_models(session) -> dict | None` — uses the existing `curl_cffi` session from the caller; returns parsed model dict or `None` on failure
- [ ] Parse the response JSON into `{mode: {model_name: internal_api_id}}` shape
- [ ] Handle all failure modes gracefully — network error, non-200 status, unexpected JSON shape, missing fields — all return `None`, never raise
- [ ] Log a warning (not an error) when fetch fails
- [ ] Add 5-second timeout to the request
- [ ] Do NOT instantiate a new session — reuse the caller's session to avoid extra TLS handshakes

### Configuration / Code Reference

```python
# perplexity/model_fetcher.py

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Populated after Workstream A discovery
_MODEL_ENDPOINT = "https://www.perplexity.ai/api/list-models"  # placeholder

async def fetch_models(session) -> Optional[dict]:
    """
    Fetch live model list from Perplexity.
    Returns merged-format dict on success, None on any failure.
    Never raises.
    """
    try:
        resp = await session.get(
            _MODEL_ENDPOINT,
            timeout=5,
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            logger.warning(
                "model_fetcher: non-200 response %d from %s",
                resp.status_code, _MODEL_ENDPOINT
            )
            return None

        data = resp.json()
        return _parse(data)

    except Exception as e:
        logger.warning("model_fetcher: fetch failed — %s", e)
        return None


def _parse(data: dict) -> Optional[dict]:
    """
    Parse raw API response into {mode: {model_name: internal_id}} shape.
    Returns None if shape is unrecognizable.
    Shape depends on Workstream A discovery outcome — implement after.
    """
    # TODO: implement after Workstream A confirms response shape
    raise NotImplementedError("Implement after endpoint discovery in Workstream A")
```

### Acceptance Criteria

- [ ] `fetch_models()` returns `None` on network failure (no exception propagates)
- [ ] `fetch_models()` returns `None` on non-200 response
- [ ] `fetch_models()` returns `None` on unexpected JSON shape
- [ ] `fetch_models()` returns valid dict on well-formed response
- [ ] Request timeout respected (5 seconds)
- [ ] No new session created — existing session reused
- [ ] Warning logged on every failure

---

## 8. Workstream D — Client Integration (`perplexity_async/client.py`)

**Status:** Blocked on Workstream C

### Problem / Goal

Wire the fetcher into `perplexity_async.Client.__init__()` so model data is refreshed once at startup. Also expose a `refresh_models()` method for on-demand refresh (used by the proxy's `/v1/models` hot-reload path in Phase 4).

### Implementation Tasks

- [ ] After session initialization in `Client.__init__()`, call `fetch_models(self._session)` 
- [ ] If result is not `None`, call `_registry.merge(result)`
- [ ] If result is `None`, log a warning and continue with static map — do not fail init
- [ ] Add `async def refresh_models(self) -> bool` method — re-fetches and merges; returns `True` on success, `False` on failure
- [ ] Startup model fetch must not add more than 5 seconds to init time (enforced by fetcher timeout)
- [ ] Apply identical changes to `perplexity/client.py` (sync client) using the sync `curl_cffi` session

### Configuration / Code Reference

```python
# perplexity_async/client.py — additions to __init__ and new method

async def __init__(self, cookies: dict = {}):
    # ... existing session init ...

    # Phase 3 addition — dynamic model refresh at startup
    from perplexity.model_fetcher import fetch_models
    from perplexity.models import _registry

    dynamic = await fetch_models(self._session)
    if dynamic:
        _registry.merge(dynamic)
    else:
        logger.warning("perplexity_async: model refresh failed, using static map")

async def refresh_models(self) -> bool:
    """Re-fetches live model list and merges into registry. Returns True on success."""
    from perplexity.model_fetcher import fetch_models
    from perplexity.models import _registry
    dynamic = await fetch_models(self._session)
    if dynamic:
        _registry.merge(dynamic)
        return True
    return False
```

### Acceptance Criteria

- [ ] `Client.__init__()` completes within 10 seconds regardless of model fetch outcome
- [ ] Failed model fetch does not raise — client initializes with static map
- [ ] After successful init, `resolve_preference()` returns IDs for any new models fetched
- [ ] `await client.refresh_models()` returns `True` and updates registry on success
- [ ] `await client.refresh_models()` returns `False` and does not corrupt registry on failure
- [ ] Sync client receives identical changes

---

## 9. Workstream E — Proxy Hot-Reload (`perplexity-proxy/app/router.py`)

**Status:** Blocked on Workstream D

### Problem / Goal

Expose a `POST /v1/models/refresh` endpoint in the proxy that triggers `client.refresh_models()` and invalidates the model mapper cache. After a refresh, `GET /v1/models` returns the updated model list without a server restart.

### Implementation Tasks

- [ ] Add `POST /v1/models/refresh` route to `app/router.py`
- [ ] Route calls `await get_client().refresh_models()`
- [ ] On success: rebuild `MODEL_MAP` in `app/mapper.py` by calling `build_model_map()` again
- [ ] On failure: return `503` with detail `"Model refresh failed — static map still active"`
- [ ] Protect endpoint with a secret token from config (`REFRESH_SECRET` env var) — `Authorization: Bearer <secret>` header required
- [ ] Add `REFRESH_SECRET` to `app/config.py` and `.env.example`
- [ ] `GET /v1/models` after a successful refresh returns the updated list

### Configuration / Code Reference

```python
# app/router.py — new endpoint

from fastapi import Header, HTTPException
from app.config import settings
from app import mapper

@router.post("/v1/models/refresh")
async def refresh_models(authorization: str = Header(...)):
    expected = f"Bearer {settings.REFRESH_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid refresh secret")

    client = get_client()
    success = await client.refresh_models()

    if not success:
        raise HTTPException(
            status_code=503,
            detail="Model refresh failed — static map still active"
        )

    # Rebuild proxy model map from updated registry
    mapper.MODEL_MAP = mapper.build_model_map()

    return {
        "status": "ok",
        "model_count": len(mapper.MODEL_MAP),
        "models": list(mapper.MODEL_MAP.keys())
    }
```

```env
# .env.example addition
REFRESH_SECRET=change-me-in-production
```

### Acceptance Criteria

- [ ] `POST /v1/models/refresh` with valid token triggers `refresh_models()`
- [ ] `GET /v1/models` after refresh returns updated list
- [ ] `POST /v1/models/refresh` without token returns `401`
- [ ] `POST /v1/models/refresh` when fetch fails returns `503`
- [ ] `GET /v1/models` after failed refresh still returns previous (static) list

---

## 10. Workstream F — Test Coverage

**Status:** New
**Required before:** merging Phase 3 PR

### 10.1 Unit tests — `perplexity/models.py` (ModelRegistry)

- [ ] `_registry.merge()` adds new models not in static map
- [ ] `_registry.merge()` does NOT overwrite existing static entries
- [ ] `_registry.reset()` restores static baseline
- [ ] Concurrent `merge()` + `resolve()` calls do not corrupt state
- [ ] `list_flat()` after merge includes newly added models
- [ ] `available()` after merge includes newly added modes/models
- [ ] All Phase 1 unit tests still pass

### 10.2 Unit tests — `perplexity/model_fetcher.py`

- [ ] Network error returns `None`
- [ ] Non-200 status returns `None`
- [ ] Malformed JSON returns `None`
- [ ] Missing required fields in response returns `None`
- [ ] Well-formed response returns correctly shaped dict
- [ ] Timeout (> 5s) returns `None`

### 10.3 Integration tests — `perplexity_async/client.py`

- [ ] Client init with successful fetch → `resolve_preference()` returns new model IDs
- [ ] Client init with failed fetch → initializes successfully with static map
- [ ] `refresh_models()` returns `True` on success
- [ ] `refresh_models()` returns `False` on failure without corrupting registry

### 10.4 Integration tests — proxy `POST /v1/models/refresh`

- [ ] Valid token + successful refresh → 200, updated model list
- [ ] Valid token + failed refresh → 503
- [ ] Invalid token → 401
- [ ] `GET /v1/models` after refresh returns updated list

---

## 11. Workstream G — Pre-Merge Audit Checklist

**Status:** New

### 11.1 Backward compatibility

- [ ] All Phase 1 tests pass without modification
- [ ] All Phase 2 tests pass without modification
- [ ] `resolve_preference()`, `list_flat_models()`, `get_available_models()` signatures unchanged
- [ ] `Client.search()` signature unchanged
- [ ] `GET /v1/models` still returns 200 (now dynamic)
- [ ] `POST /v1/chat/completions` still returns 200
- [ ] `POST /v1/responses` still returns 200

### 11.2 Failure safety

- [ ] Proxy starts correctly when Perplexity model endpoint is unreachable
- [ ] Proxy starts correctly with no cookies (anonymous mode)
- [ ] Model fetch timeout does not block client initialization beyond 10s
- [ ] Failed refresh does not affect in-flight requests

### 11.3 Security

- [ ] `REFRESH_SECRET` required and non-empty in production config
- [ ] `POST /v1/models/refresh` returns 401 for missing or wrong token
- [ ] Fetched model data is validated before merge — no arbitrary key injection

---

## 12. Combined Implementation Order

1. Complete Workstream A — endpoint discovery and documentation
2. Complete Workstream B — `ModelRegistry` in `perplexity/models.py`
3. Complete Workstream C — `perplexity/model_fetcher.py` (implement `_parse()` after A)
4. Complete Workstream D — wire fetcher into `perplexity_async.Client` and sync client
5. Complete Workstream E — `POST /v1/models/refresh` proxy endpoint
6. Run Workstream G — pre-merge audit
7. Complete Workstream F — full test coverage
8. Open PR, verify CI passes
9. Deploy and verify `GET /v1/models` returns live Perplexity model list

### Acceptance Criteria for First End-to-End Run

- `GET /v1/models` returns models beyond the static list (confirms live fetch working)
- `POST /v1/models/refresh` with valid token triggers live re-fetch and updates list
- Proxy starts correctly when network is unavailable (static fallback confirmed)
- All existing proxy endpoints (chat/completions, responses) work with newly fetched models

---

## 13. Definition of Done

Phase 3 is complete when **all** of the following are true simultaneously.

### 13.1 Library layer

- [x] Phase 1 and Phase 2 Definitions of Done fully satisfied
- [ ] `ModelRegistry` implemented in `perplexity/models.py`
- [ ] `perplexity/model_fetcher.py` implemented with confirmed endpoint from Workstream A
- [ ] `perplexity_async.Client` fetches models at startup and exposes `refresh_models()`
- [ ] Sync `perplexity.Client` receives identical changes

### 13.2 Proxy layer

- [ ] `POST /v1/models/refresh` implemented with token auth
- [ ] `GET /v1/models` returns live model list after refresh
- [ ] `REFRESH_SECRET` added to config and `.env.example`

### 13.3 Test layer

- [ ] All Workstream F tests implemented and passing
- [ ] No regressions in Phase 1 or Phase 2 tests
- [ ] Failure paths tested (no network, bad response, wrong token)

### 13.4 Discovery layer

- [ ] `docs/model-discovery.md` committed with endpoint details
- [ ] `_parse()` in `model_fetcher.py` implemented (not `NotImplementedError`)

---

## 14. What Phase 4 Inherits

### 14.1 Capabilities unlocked by Phase 3

- Proxy model list is live and self-updating — no deploys needed when Perplexity adds models
- `refresh_models()` available for CLIProxyAPI to call via a scheduled task or webhook
- New models (e.g. GPT-5.4, Claude 4.6) automatically appear in `GET /v1/models` and are routable

### 14.2 Phase Boundary

- Phase 4 is CLIProxyAPI integration — the final wiring step
- Phase 3 is the prerequisite
- Do not start Phase 4 until Phase 3 Definition of Done is fully satisfied, OR skip Phase 3 if Workstream A (endpoint discovery) fails and proceed to Phase 4 with static map

---

## 15. Compact Mental Model

### 15.1 Phase Relationships

- Phase 1: Single source of truth for model data (`models.py`)
- Phase 2: OpenAI-compatible proxy with static model list
- Phase 3: Static model list becomes live + self-updating
- Phase 4: Wire everything into CLIProxyAPI

### 15.2 Key Takeaway

Phase 3 has one hard dependency that cannot be coded around: **Workstream A must confirm that Perplexity's model endpoint exists and returns internal API IDs**. If the endpoint returns only display names (not internal IDs like `gpt52`), Phase 3 can only update display names but not add new models to the routing table. If the endpoint does not exist at all, Phase 3 is skipped entirely and Phase 4 proceeds with the static map from Phase 2.

The static map is always the fallback. Phase 3 is an enhancement, not a requirement for the proxy to function.

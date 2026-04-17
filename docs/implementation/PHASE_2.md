# Phase 2 — perplexity-proxy FastAPI Server (Status: Planned)

**Status:** Planned
**Owner:** perplexity-proxy
**Last updated:** 2026-04-17
**Delivery status:** Not started

---

## 1. Purpose

Phase 2 builds the `perplexity-proxy` FastAPI server — an OpenAI-compatible HTTP API that translates incoming requests into Perplexity queries using the refactored `perplexity-ai` library from Phase 1. The server must implement every endpoint that real-world OpenAI-compatible clients call, including both the legacy `POST /v1/chat/completions` and the newer `POST /v1/responses` (Responses API), which is what tools like opencode actively use.

This phase is NOT about dynamic model fetching from Perplexity at runtime (deferred to Phase 3), nor about CLIProxyAPI integration config (covered in Phase 4).

---

## 2. What Carried Over / What Must Stay Stable

The following are already implemented and must remain stable:

- [x] `perplexity/models.py` — `MODEL_PREFERENCE_MAP`, `AVAILABLE_MODELS`, `resolve_preference()`, `list_flat_models()`
- [x] `perplexity_async.Client` — async search interface used as the proxy backend
- [x] `perplexity_async.Client.search()` signature — `query, mode, model, sources, stream, language, follow_up, incognito`
- [x] Phase 1 Definition of Done fully satisfied

This phase builds on top of these. Do not regress them.

---

## 3. Background & Motivation

### Why two generation endpoints?

CLIProxyAPI logs revealed that opencode sends requests to **both** `/v1/chat/completions` and `/v1/responses` depending on the model and task. The Responses API is OpenAI's next-generation interface — newer models and agentic clients prefer or require it. A proxy that only implements `chat/completions` will return 404s for Responses API callers.

The double-prefix 404s observed in the logs (`/v1/v1/responses`, `/v1/v1/chat/completions`) are a CLIProxyAPI config issue — the proxy's `base-url` must be set to `http://127.0.0.1:8080` (no `/v1` suffix), as CLIProxyAPI automatically prepends `/v1` to all upstream calls.

### Endpoint inventory from logs

```
GET  /v1/models              ✅ needed — model discovery
POST /v1/chat/completions    ✅ needed — legacy generation (Cursor, LangChain, etc.)
POST /v1/responses           ✅ needed — Responses API (opencode, newer clients)
GET  /health                 ✅ needed — uptime monitoring
```

---

## 4. Current Failure State / Known Blockers

- `POST /v1/responses` = not implemented — opencode receives 404
- `POST /v1/v1/responses` = 404 — caused by `/v1` suffix in CLIProxyAPI `base-url` config (fix: remove `/v1` from base-url)
- `perplexity_async.Client` = not yet wrapped in a lifecycle-managed singleton
- Response caching = not implemented — identical queries hit Perplexity every time
- Streaming SSE format for `/v1/responses` differs from `/v1/chat/completions` — needs separate formatter

---

## 5. Workstream A — Project Scaffold & Configuration

**Status:** New

### Problem / Goal

Create the full project directory structure, configuration system, and base FastAPI app before any endpoints are written.

### Implementation Tasks

- [ ] Create `perplexity-proxy/` directory with full structure as defined in README file map
- [ ] Create `app/__init__.py`, `tests/__init__.py`
- [ ] Implement `app/config.py` using `pydantic-settings` `BaseSettings`
- [ ] Implement `app/main.py` — FastAPI app factory with lifespan handler
- [ ] Create `pyproject.toml` with all dependencies
- [ ] Create `requirements.txt` (pinned) and `requirements-dev.txt`
- [ ] Create `.env.example` documenting all env vars
- [ ] Create `.gitignore`

### Configuration / Code Reference

```python
# app/config.py
from pydantic_settings import BaseSettings
from pydantic import field_validator
import json

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8080
    WORKERS: int = 4
    LOG_LEVEL: str = "info"
    PERPLEXITY_COOKIES: dict = {}
    CACHE_ENABLED: bool = True
    CACHE_MAX_SIZE: int = 256
    CACHE_TTL_SECONDS: int = 300

    @field_validator("PERPLEXITY_COOKIES", mode="before")
    @classmethod
    def parse_cookies(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    class Config:
        env_file = ".env"

settings = Settings()
```

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.client import init_client, close_client
from app.router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_client()
    yield
    await close_client()

app = FastAPI(
    title="perplexity-proxy",
    description="OpenAI-compatible proxy for Perplexity AI",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)
```

```env
# .env.example
HOST=0.0.0.0
PORT=8080
WORKERS=4
LOG_LEVEL=info
# Without cookies: only 'auto' mode available
# PERPLEXITY_COOKIES={"next-auth.session-token": "your-token"}
CACHE_ENABLED=true
CACHE_MAX_SIZE=256
CACHE_TTL_SECONDS=300
```

### Acceptance Criteria

- [ ] `uvicorn app.main:app` starts without errors
- [ ] All config fields readable from `.env` file
- [ ] `PERPLEXITY_COOKIES` parsed correctly from JSON string env var
- [ ] Lifespan handler runs init/close without errors

---

## 6. Workstream B — Pydantic Schemas (`app/models.py`)

**Status:** New

### Problem / Goal

Define all request and response Pydantic schemas for both `chat/completions` and `responses` endpoints. Schemas must accept real OpenAI client payloads without validation errors, even for fields the proxy ignores.

### Implementation Tasks

- [ ] Define `ChatMessage` — `role`, `content`
- [ ] Define `ChatRequest` — `model`, `messages`, `stream`, `temperature`, `max_tokens`, `top_p` (accepted, ignored)
- [ ] Define `ChatResponseMessage` — `role`, `content`
- [ ] Define `ChatChoice` — `index`, `message`, `finish_reason`
- [ ] Define `ChatResponse` — `id`, `object`, `created`, `model`, `choices`, `usage`
- [ ] Define `StreamDelta` — `role` (optional), `content` (optional)
- [ ] Define `StreamChoice` — `index`, `delta`, `finish_reason`
- [ ] Define `StreamChunk` — `id`, `object`, `created`, `model`, `choices`
- [ ] Define `ResponsesInput` — `str` or `list[ChatMessage]` (union)
- [ ] Define `ResponsesRequest` — `model`, `input`, `instructions` (optional system prompt), `stream`, `temperature`, `max_output_tokens` (accepted, ignored)
- [ ] Define `ResponsesOutputText` — `type: "text"`, `text: str`
- [ ] Define `ResponsesOutputMessage` — `type: "message"`, `role: "assistant"`, `content: list[ResponsesOutputText]`
- [ ] Define `ResponsesResponse` — `id`, `object: "response"`, `created_at`, `model`, `output: list[ResponsesOutputMessage]`, `usage`
- [ ] Define `ResponsesStreamEvent` — SSE event format for Responses API streaming
- [ ] Define `ModelObject` — `id`, `object: "model"`, `created`, `owned_by`
- [ ] Define `ModelList` — `object: "list"`, `data: list[ModelObject]`
- [ ] Define `HealthResponse` — `status`, `cache_enabled`, `authenticated`, `model_count`

### Configuration / Code Reference

```python
# Responses API request shape (POST /v1/responses)
{
    "model": "sonar-reasoning",
    "input": "How does Python's GIL work?",       # string form
    # OR
    "input": [                                      # array form
        {"role": "user", "content": "..."}
    ],
    "instructions": "You are a coding assistant.", # optional system prompt (ignored by proxy)
    "stream": false
}

# Responses API response shape
{
    "id": "resp_abc123",
    "object": "response",
    "created_at": 1713000000,
    "model": "sonar-reasoning",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The GIL is..."}
            ]
        }
    ],
    "usage": {"input_tokens": 12, "output_tokens": 150, "total_tokens": 162}
}
```

### Acceptance Criteria

- [ ] `ChatRequest` validates real openai-python SDK payloads without errors
- [ ] `ResponsesRequest` accepts both string and array `input`
- [ ] All extra fields from real clients are ignored gracefully (`model_config = ConfigDict(extra="ignore")`)
- [ ] `ChatResponse` shape matches OpenAI SDK expectations exactly
- [ ] `ResponsesResponse` shape matches OpenAI Responses API spec

---

## 7. Workstream C — Client Wrapper (`app/client.py`)

**Status:** New

### Problem / Goal

Wrap `perplexity_async.Client` in a singleton lifecycle manager. All endpoints share one client instance per worker. Handle exceptions from the library and map them to appropriate HTTP errors.

### Implementation Tasks

- [ ] Define module-level `_client: perplexity_async.Client | None = None`
- [ ] Implement `async init_client()` — instantiates client with cookies from config, called during lifespan startup
- [ ] Implement `async close_client()` — called during lifespan shutdown
- [ ] Implement `get_client() -> perplexity_async.Client` — returns singleton, raises `503` if not initialized
- [ ] Implement `async search(query, mode, model, stream) -> dict | AsyncGenerator` — thin wrapper with retry logic
- [ ] Add exponential backoff retry — 3 attempts, delays: 1s, 2s, 4s — only on `NetworkError`
- [ ] Map library exceptions to `HTTPException`:
  - `AuthenticationError` → `401`
  - `RateLimitError` → `429`
  - `NetworkError` → `503` (after retries exhausted)
  - `ValidationError` → `400`
  - `ResponseParseError` → `502`
  - All others → `500`

### Configuration / Code Reference

```python
# app/client.py
import asyncio
import perplexity_async
from perplexity.exceptions import (
    AuthenticationError, RateLimitError, NetworkError,
    ValidationError, ResponseParseError
)
from fastapi import HTTPException
from app.config import settings

_client: perplexity_async.Client | None = None

EXCEPTION_MAP = {
    AuthenticationError: 401,
    RateLimitError: 429,
    NetworkError: 503,
    ValidationError: 400,
    ResponseParseError: 502,
}

async def init_client():
    global _client
    _client = await perplexity_async.Client(settings.PERPLEXITY_COOKIES or {})

async def close_client():
    global _client
    _client = None

def get_client() -> perplexity_async.Client:
    if _client is None:
        raise HTTPException(status_code=503, detail="Client not initialized")
    return _client

async def search(query: str, mode: str, model, stream: bool = False):
    client = get_client()
    retries = 3
    for attempt in range(retries):
        try:
            return await client.search(query, mode=mode, model=model, stream=stream)
        except NetworkError:
            if attempt == retries - 1:
                raise HTTPException(status_code=503, detail="Upstream unavailable")
            await asyncio.sleep(2 ** attempt)
        except tuple(EXCEPTION_MAP.keys()) as e:
            raise HTTPException(status_code=EXCEPTION_MAP[type(e)], detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
```

### Acceptance Criteria

- [ ] Single `perplexity_async.Client` instance shared across all requests within a worker
- [ ] `init_client()` called exactly once during lifespan startup
- [ ] `close_client()` called on shutdown without errors
- [ ] `NetworkError` retried 3 times with backoff before returning 503
- [ ] All library exceptions map to correct HTTP status codes
- [ ] Proxy with empty cookies starts without errors (anonymous mode)

---

## 8. Workstream D — Dynamic Model Mapper (`app/mapper.py`)

**Status:** New

### Problem / Goal

Build the proxy's model name → Perplexity mode/model resolution layer dynamically from `perplexity.models.MODEL_PREFERENCE_MAP`. No hardcoded model names in the proxy.

### Implementation Tasks

- [ ] Import `MODEL_PREFERENCE_MAP` from `perplexity.models`
- [ ] Implement `build_model_map() -> dict[str, dict]` — generates `{proxy_name: {mode, model}}` for all entries
- [ ] Handle `None` model keys → hyphenated mode name (e.g. `"deep research"` → `"deep-research"`)
- [ ] Export `MODEL_MAP` as module-level constant (built once at import time)
- [ ] Implement `resolve(model_name: str) -> tuple[str, str | None]` — returns `(mode, model)`, raises `HTTPException(400)` for unknown names
- [ ] Implement `get_model_list() -> list[dict]` — returns list of `ModelObject`-shaped dicts for `/v1/models`

### Configuration / Code Reference

```python
# app/mapper.py
from perplexity.models import MODEL_PREFERENCE_MAP
from fastapi import HTTPException

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
    import time
    return [
        {"id": name, "object": "model", "created": int(time.time()), "owned_by": "perplexity"}
        for name in MODEL_MAP.keys()
    ]
```

### Acceptance Criteria

- [ ] `MODEL_MAP` contains an entry for every mode/model combination in `MODEL_PREFERENCE_MAP`
- [ ] `resolve("sonar-reasoning")` returns `("reasoning", None)`
- [ ] `resolve("gpt-5.2")` returns `("pro", "gpt-5.2")`
- [ ] `resolve("deep-research")` returns `("deep research", None)`
- [ ] `resolve("unknown-model")` raises `HTTPException(400)`
- [ ] `get_model_list()` length equals `len(MODEL_MAP)`
- [ ] Adding a new entry to `MODEL_PREFERENCE_MAP` automatically appears in `MODEL_MAP` — no proxy changes needed

---

## 9. Workstream E — LRU Cache (`app/cache.py`)

**Status:** New

### Problem / Goal

Avoid redundant upstream requests for identical queries. Cache responses in memory with TTL expiry and LRU eviction. Respect `CACHE_ENABLED` config flag.

### Implementation Tasks

- [ ] Implement `LRUCache` class with `asyncio.Lock`
- [ ] `make_key(query: str, model_name: str) -> str` — SHA256 hash of `f"{model_name}:{query}"`
- [ ] `async get(key: str) -> str | None` — returns cached value or `None` on miss/expiry
- [ ] `async set(key: str, value: str)` — stores value with timestamp; evicts LRU when at max size
- [ ] `async clear()` — wipes all entries (for testing)
- [ ] Instantiate a module-level `cache = LRUCache(...)` using config values
- [ ] Cache is bypassed entirely (always returns `None`) when `CACHE_ENABLED=false`
- [ ] Streaming responses are NOT cached (cache applies to non-streaming only)

### Configuration / Code Reference

```python
# app/cache.py
import asyncio
import hashlib
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

    def make_key(self, query: str, model_name: str) -> str:
        raw = f"{model_name}:{query}"
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

cache = LRUCache(
    max_size=settings.CACHE_MAX_SIZE,
    ttl=settings.CACHE_TTL_SECONDS,
    enabled=settings.CACHE_ENABLED,
)
```

### Acceptance Criteria

- [ ] Cache hit returns stored value without calling `search()`
- [ ] Cache miss returns `None`
- [ ] Entry expired after `CACHE_TTL_SECONDS` seconds
- [ ] LRU eviction removes oldest entry when `CACHE_MAX_SIZE` is reached
- [ ] `CACHE_ENABLED=false` always returns `None`
- [ ] Streaming requests bypass cache entirely

---

## 10. Workstream F — Streaming Formatters (`app/streaming.py`)

**Status:** New

### Problem / Goal

`/v1/chat/completions` and `/v1/responses` use different SSE event formats. Both must be implemented. Each takes a `perplexity_async` stream generator and yields properly formatted SSE strings.

### Implementation Tasks

- [ ] Implement `chat_completions_stream(generator, model, req_id)` — yields OpenAI `chat.completion.chunk` SSE events
- [ ] Implement `responses_stream(generator, model, resp_id)` — yields Responses API `response.output_text.delta` SSE events
- [ ] Both functions yield `f"data: {json.dumps(event)}

"` per chunk
- [ ] Both functions yield `"data: [DONE]

"` as the final event
- [ ] Empty / `None` content chunks are skipped silently
- [ ] Both functions handle generator exhaustion gracefully

### Configuration / Code Reference

```python
# app/streaming.py — SSE format reference

# chat/completions chunk format:
{
    "id": "chatcmpl-abc",
    "object": "chat.completion.chunk",
    "created": 1713000000,
    "model": "sonar-reasoning",
    "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
}
# final chunk:
{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

# /v1/responses stream event format:
{
    "type": "response.output_text.delta",
    "output_index": 0,
    "content_index": 0,
    "delta": "Hello"
}
# completion event:
{"type": "response.completed", "response": { ...full response object... }}
```

### Acceptance Criteria

- [ ] `chat_completions_stream` output is parseable by `openai.ChatCompletionChunk`
- [ ] `responses_stream` output is parseable by opencode's Responses API client
- [ ] Both end with `data: [DONE]

`
- [ ] No `None` values in emitted `content` fields
- [ ] Generators that yield zero chunks produce only the `[DONE]` terminator

---

## 11. Workstream G — API Routes (`app/router.py`)

**Status:** New

### Problem / Goal

Implement all four required endpoints. Each route is a thin HTTP layer — validation, cache check, client call, format, return. No business logic in routes.

### Implementation Tasks

#### `GET /v1/models`
- [ ] Return `ModelList` built from `get_model_list()`
- [ ] No authentication required

#### `GET /health`
- [ ] Return `HealthResponse` with `status`, `cache_enabled`, `authenticated` (bool: cookies set), `model_count`

#### `POST /v1/chat/completions`
- [ ] Validate `ChatRequest`
- [ ] Extract last `user` role message as `query`
- [ ] Call `resolve(req.model)` to get `(mode, model)`
- [ ] Check cache with `cache.get(key)`
- [ ] If cache hit: return cached `ChatResponse`
- [ ] If `stream=True`: return `StreamingResponse` from `chat_completions_stream()`
- [ ] If `stream=False`: call `search()`, format as `ChatResponse`, cache result, return

#### `POST /v1/responses`
- [ ] Validate `ResponsesRequest`
- [ ] Extract query: if `input` is `str` use directly; if list, take last `user` message content
- [ ] Call `resolve(req.model)` to get `(mode, model)`
- [ ] Check cache (same key scheme as chat/completions)
- [ ] If cache hit: return cached `ResponsesResponse`
- [ ] If `stream=True`: return `StreamingResponse` from `responses_stream()`
- [ ] If `stream=False`: call `search()`, format as `ResponsesResponse`, cache result, return

### Configuration / Code Reference

```python
# app/router.py — route signatures

@router.get("/v1/models", response_model=ModelList)
async def list_models(): ...

@router.get("/health", response_model=HealthResponse)
async def health(): ...

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest): ...

@router.post("/v1/responses")
async def responses(req: ResponsesRequest): ...
```

### Acceptance Criteria

- [ ] `GET /v1/models` returns 200 with all models from `MODEL_MAP`
- [ ] `POST /v1/chat/completions` returns valid OpenAI ChatCompletion shape
- [ ] `POST /v1/responses` returns valid Responses API shape
- [ ] `POST /v1/chat/completions` with `stream=true` returns `text/event-stream`
- [ ] `POST /v1/responses` with `stream=true` returns `text/event-stream`
- [ ] Unknown model returns `400` with list of valid models
- [ ] Cache hit skips `search()` call (verifiable via mock)
- [ ] `/v1/v1/responses` is NOT a registered route (no double-prefix routes)

---

## 12. Workstream H — Infrastructure

**Status:** New

### Problem / Goal

Create production deployment files: Gunicorn config, Dockerfile, and docker-compose.

### Implementation Tasks

- [ ] Create `gunicorn.conf.py` — uvicorn worker class, `2×CPU+1` workers, 120s timeout, 5s keepalive
- [ ] Create `Dockerfile` — `python:3.12-slim`, install local `perplexity-ai`, install proxy deps, uvicorn entrypoint
- [ ] Create `docker-compose.yml` — proxy service with gunicorn, `.env` file mount, port 8080, `unless-stopped`

### Configuration / Code Reference

```python
# gunicorn.conf.py
import multiprocessing

worker_class = "uvicorn.workers.UvicornWorker"
workers = multiprocessing.cpu_count() * 2 + 1
timeout = 120      # deep research queries are slow
keepalive = 5
bind = "0.0.0.0:8080"
```

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY ../perplexity-ai /deps/perplexity-ai
RUN pip install --no-cache-dir /deps/perplexity-ai
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Acceptance Criteria

- [ ] `docker-compose up -d` starts the proxy successfully
- [ ] `curl http://localhost:8080/health` returns 200
- [ ] `gunicorn` starts with correct worker count
- [ ] Container restarts automatically after crash

---

## 13. Workstream I — Test Coverage

**Status:** New
**Required before:** merging Phase 2 PR

### 13.1 Unit tests — `app/mapper.py`

- [ ] `MODEL_MAP` contains correct entry count
- [ ] `resolve()` returns correct `(mode, model)` pairs for known models
- [ ] `resolve("unknown")` raises `HTTPException(400)`
- [ ] `get_model_list()` returns list with correct `id` fields

### 13.2 Unit tests — `app/cache.py`

- [ ] Cache hit returns stored value
- [ ] Cache miss returns `None`
- [ ] TTL expiry returns `None`
- [ ] LRU eviction removes oldest on overflow
- [ ] `CACHE_ENABLED=false` always returns `None`

### 13.3 Unit tests — `app/streaming.py`

- [ ] `chat_completions_stream` yields valid delta events
- [ ] `responses_stream` yields valid `response.output_text.delta` events
- [ ] Both end with `data: [DONE]

`
- [ ] Empty chunks are skipped

### 13.4 Integration tests — `app/router.py`

- [ ] `GET /v1/models` → 200, correct shape
- [ ] `POST /v1/chat/completions` → 200, `ChatResponse` shape (mock client)
- [ ] `POST /v1/responses` → 200, `ResponsesResponse` shape (mock client)
- [ ] `POST /v1/chat/completions` with `stream=true` → `text/event-stream`
- [ ] `POST /v1/responses` with `stream=true` → `text/event-stream`
- [ ] Unknown model → 400
- [ ] `AuthenticationError` from client → 401
- [ ] `RateLimitError` from client → 429
- [ ] Cache hit → `search()` not called (assert mock not called)
- [ ] `GET /health` → 200, correct fields

---

## 14. Workstream J — Pre-Merge Audit Checklist

**Status:** New
**Must complete before:** merging Phase 2 PR

### 14.1 Endpoint audit

- [ ] `GET /v1/models` registered and returns 200
- [ ] `POST /v1/chat/completions` registered and returns 200
- [ ] `POST /v1/responses` registered and returns 200
- [ ] `GET /health` registered and returns 200
- [ ] No `/v1/v1/...` routes registered

### 14.2 CLIProxyAPI config validation

- [ ] `base-url` in CLIProxyAPI config set to `http://127.0.0.1:8080` (no `/v1` suffix)
- [ ] CLIProxyAPI logs show no more `404 /v1/v1/...` errors after fix

### 14.3 Schema validation

- [ ] Real `openai` Python SDK client can call proxy without validation errors
- [ ] `openai.ChatCompletion.parse()` succeeds on proxy response
- [ ] Streaming responses parseable by `openai` SDK stream iterator

### 14.4 Performance baseline

- [ ] Non-streaming response time < 10s for `auto` mode
- [ ] Streaming first-token latency < 3s
- [ ] Cache hit response time < 5ms

---

## 15. Combined Implementation Order

1. Complete Workstream A — scaffold, config, main.py
2. Complete Workstream B — Pydantic schemas
3. Complete Workstream D — mapper (depends on Phase 1)
4. Complete Workstream C — client wrapper
5. Complete Workstream E — LRU cache
6. Complete Workstream F — streaming formatters
7. Complete Workstream G — API routes (depends on B, C, D, E, F)
8. Complete Workstream H — infrastructure files
9. Run Workstream J — pre-merge audit
10. Complete Workstream I — full test coverage
11. Start proxy, run end-to-end test with opencode
12. Fix CLIProxyAPI `base-url` config, verify no 404s in logs

### Acceptance Criteria for First End-to-End Run

- `curl http://localhost:8080/v1/models` returns all Perplexity models
- opencode successfully calls `POST /v1/chat/completions` and receives a response
- opencode successfully calls `POST /v1/responses` and receives a response
- CLIProxyAPI logs show 200 for both endpoints with no double-prefix 404s
- Streaming request produces visible token-by-token output in opencode

---

## 16. Definition of Done

Phase 2 is complete when **all** of the following are true simultaneously.

### 16.1 Endpoint layer

- [x] Phase 1 Definition of Done fully satisfied
- [ ] `GET /v1/models` implemented and tested
- [ ] `POST /v1/chat/completions` implemented and tested (streaming + non-streaming)
- [ ] `POST /v1/responses` implemented and tested (streaming + non-streaming)
- [ ] `GET /health` implemented and tested

### 16.2 Core layer

- [ ] `perplexity_async.Client` singleton managed by lifespan
- [ ] `MODEL_MAP` built dynamically from `perplexity.models.MODEL_PREFERENCE_MAP`
- [ ] LRU cache operational with TTL and eviction
- [ ] All library exceptions mapped to HTTP status codes
- [ ] Retry logic operational for `NetworkError`

### 16.3 Schema layer

- [ ] `ChatResponse` passes `openai` SDK parsing
- [ ] `ResponsesResponse` passes Responses API shape validation
- [ ] Extra fields from real clients ignored without validation errors

### 16.4 Test layer

- [ ] All Workstream I tests implemented and passing
- [ ] No regressions in Phase 1 tests
- [ ] Pre-merge audit (Workstream J) fully completed

### 16.5 Infrastructure layer

- [ ] `docker-compose up` starts proxy successfully
- [ ] `gunicorn.conf.py` configured correctly

---

## 17. What Phase 3 Inherits

### 17.1 Capabilities unlocked by Phase 2

- Fully working OpenAI-compatible proxy serving both `chat/completions` and `responses` endpoints
- CLIProxyAPI can route to proxy with correct `base-url` config
- All current Perplexity models available to opencode and Cursor
- Static `MODEL_MAP` ready to be replaced by dynamic fetching in Phase 3

### 17.2 Phase Boundary

- Phase 3 is runtime model fetching — replacing static `MODEL_PREFERENCE_MAP` with live data scraped from Perplexity's web app at client startup.
- Phase 2 is the prerequisite.
- Do not start Phase 3 work until Phase 2 definition of done is fully satisfied.

---

## 18. Compact Mental Model

### 18.1 Phase Relationships

- Phase 1: Extract model definitions from `client.py` → `perplexity/models.py`
- Phase 2: Build FastAPI proxy with all OpenAI-compatible endpoints
- Phase 3: Dynamic model fetching from Perplexity at runtime
- Phase 4: CLIProxyAPI native integration (optional)

### 18.2 Key Takeaway

The most important decision in Phase 2 is implementing **both** `/v1/chat/completions` and `/v1/responses`. Skipping the Responses API endpoint would cause all opencode requests to 404. The second most important decision is setting `base-url` in CLIProxyAPI **without** the `/v1` suffix — the double-prefix bug is a config error, not a code error, but it blocks all routing until fixed.

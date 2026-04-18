# perplexity-proxy

> A high-performance OpenAI-compatible API proxy that routes requests to Perplexity AI using a reverse-engineered web client — no official API key required.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

`perplexity-proxy` is an OpenAI-compatible REST API server built with FastAPI that sits between any OpenAI-compatible client (opencode, Cursor, VS Code Copilot, LangChain, etc.) and Perplexity AI. It translates standard `/v1/chat/completions`, `/v1/completions`, and `/v1/responses` requests into Perplexity queries using the `perplexity-ai` Python library, which interacts with Perplexity's web interface directly — bypassing the need for a paid API key.

Model availability is resolved **dynamically** at startup by reading `AVAILABLE_MODELS` from the `perplexity` library itself. When the upstream library adds or removes models, the proxy reflects those changes automatically — no manual mapping maintenance required.

> **CLIProxyAPI note:** set `base-url` to `http://127.0.0.1:8080` (no `/v1` suffix). CLIProxyAPI prepends `/v1` automatically.

---

## How It Works

```
Client (opencode / Cursor / any OpenAI-compatible tool)
        │
        │  POST /v1/chat/completions  (OpenAI format)
        ▼
┌─────────────────────────────────────────────────────────┐
│                    perplexity-proxy                     │
│                                                         │
│  ┌─────────────┐   ┌───────────────┐   ┌─────────────┐ │
│  │  FastAPI    │──▶│   Request     │──▶│   Dynamic   │ │
│  │  Router     │   │   Validator   │   │  Model Map  │ │
│  └─────────────┘   └───────────────┘   └─────────────┘ │
│         │                                     │         │
│  ┌──────▼──────┐   ┌───────────────┐          │         │
│  │  Response   │◀──│ perplexity    │◀──────────┘         │
│  │  Formatter  │   │ _async client │                     │
│  └─────────────┘   └───────────────┘                     │
│         │                │                               │
│  ┌──────▼──────┐   ┌─────▼──────────┐                    │
│  │  Streaming  │   │   LRU Cache    │                    │
│  │  SSE writer │   │  (TTL-based)   │                    │
│  └─────────────┘   └────────────────┘                    │
└─────────────────────────────────────────────────────────┘
        │
        │  Reverse-engineered web requests
        ▼
  Perplexity.ai  (sonar, reasoning, deep research, ...)
```

### Request Lifecycle

1. Client sends a standard OpenAI `POST /v1/chat/completions`, `POST /v1/completions`, or `POST /v1/responses` request
2. FastAPI validates the request body against Pydantic schemas
3. The **Dynamic Model Map** (built at startup from `perplexity.models.AVAILABLE_MODELS`) resolves the requested model name to a Perplexity `mode` and `model` pair
4. The **LRU Cache** is checked — if an identical normalized request was recently seen, the cached response is returned immediately
5. `perplexity_async.Client` sends the query to Perplexity's web interface asynchronously
6. The response is formatted into the appropriate OpenAI-compatible schema
7. The result is returned as a full JSON response **or** streamed token-by-token via SSE

---

## Features

- **OpenAI-compatible API** — drop-in replacement, works with any OpenAI SDK client
- **Multiple compatibility surfaces** — supports `/v1/chat/completions`, `/v1/completions`, and `/v1/responses`
- **Dynamic model discovery** — model list is auto-generated from `perplexity.models.AVAILABLE_MODELS` at startup; no hardcoded mappings
- **Async throughout** — built on `perplexity_async` and FastAPI's full async stack
- **Streaming support** — Server-Sent Events (SSE) for real-time token streaming
- **Streaming blob filtering** — internal Perplexity state blobs are dropped before formatting, and real text is extracted from upstream blocks
- **LRU response cache** — in-memory cache with configurable TTL and max size
- **Multi-worker** — runs with `gunicorn` + `uvicorn` workers for horizontal concurrency
- **Cookie-based auth** — injects Perplexity session cookies for Pro/Reasoning/Deep Research access
- **Inbound API-key auth** — optional bearer-token protection for all `/v1/*` endpoints
- **Anonymous fallback** — works without cookies in `auto` mode
- **`/v1/models` endpoint** — dynamically exposes all available models for client discovery
- **`/v1/models/refresh` endpoint** — refreshes the model map without restarting the server
- **Health check endpoint** — `/health` for uptime and auth status monitoring
- **OpenAPI / Swagger / ReDoc** — interactive API docs at `/openapi.json`, `/docs`, and `/redoc`
- **Startup Perplexity session check** — verifies cookies against Perplexity when configured
- **Structured error handling** — maps `perplexity` library exceptions to proper HTTP status codes

---

## File Structure

```
perplexity-proxy/
│
├── app/                            # Main application package
│   ├── __init__.py
│   │
│   ├── main.py                     # FastAPI app factory and lifespan handler
│   │                               # - Creates the FastAPI instance
│   │                               # - Initializes perplexity_async.Client on startup
│   │                               # - Builds the dynamic MODEL_MAP from perplexity.models
│   │                               # - Registers all routers
│   │                               # - Gracefully closes the client on shutdown
│   │
│   ├── config.py                   # Settings — loaded from environment variables or .env
│   │                               # Fields: HOST, PORT, WORKERS, LOG_LEVEL,
│   │                               #         PERPLEXITY_COOKIES, CACHE_ENABLED,
│   │                               #         CACHE_MAX_SIZE, CACHE_TTL_SECONDS
│   │
│   ├── models.py                   # Pydantic request/response schemas
│   │                               # - ChatRequest: messages, model, stream, temperature, tools
│   │                               # - CompletionsRequest: prompt-based legacy completion shim
│   │                               # - ResponsesRequest: input/instructions/stream-compatible schema
│   │                               # - ChatResponse: OpenAI-shaped completion response
│   │                               # - StreamChunk: SSE delta event schema
│   │                               # - ModelList: /v1/models response schema
│   │                               # - HealthResponse: /health response schema
│   │
│   ├── router.py                   # API route definitions (thin HTTP layer)
│   │                               # - POST /v1/chat/completions
│   │                               # - POST /v1/completions
│   │                               # - POST /v1/responses
│   │                               # - GET  /v1/models
│   │                               # - GET  /health
│   │                               # - POST /v1/models/refresh
│   │
│   ├── client.py                   # perplexity_async.Client lifecycle wrapper
│   │                               # - Holds a single shared async client instance per worker
│   │                               # - Injects cookies from config at initialization
│   │                               # - Exposes async search() with exponential backoff retry
│   │                               # - Maps perplexity exceptions → HTTPException status codes
│   │                               #   (AuthenticationError → 401, RateLimitError → 429, etc.)
│   │
│   ├── mapper.py                   # Dynamic model map builder
│   │                               # - Imports AVAILABLE_MODELS from perplexity.models
│   │                               # - build_model_map() generates proxy model names:
│   │                               #     None model  → mode name (e.g. "auto", "reasoning")
│   │                               #     Named model → model name (e.g. "gpt-5.2-thinking")
│   │                               #     "deep research" → "deep-research" (spaces → hyphens)
│   │                               # - MODEL_MAP: Dict[str, Dict] used by router and /v1/models
│   │
│   ├── cache.py                    # LRU response cache
│   │                               # - Keyed by normalized request payload hash
│   │                               # - Includes query, model_name, request_type, and request-shaping fields
│   │                               # - Max size and TTL configurable via config.py
│   │                               # - Thread-safe for multi-worker use via asyncio.Lock
│   │                               # - Returns None on cache miss
│   │
│   └── streaming.py                # SSE stream formatter
│                                   # - Wraps perplexity_async stream generator
│                                   # - Filters internal Perplexity state blobs before formatting
│                                   # - Extracts real answer text from upstream blocks/legacy fields
│                                   # - Converts chunks into OpenAI delta format
│                                   # - Yields "data: {json}\n\n" strings
│                                   # - Terminates with "data: [DONE]\n\n"
│
├── tests/
│   ├── __init__.py
│   ├── test_router.py              # Integration tests — full HTTP request/response cycle
│   ├── test_mapper.py              # Unit tests — model map generation from AVAILABLE_MODELS
│   ├── test_cache.py               # Unit tests — LRU eviction, TTL expiry, cache hits/misses
│   └── test_streaming.py           # Unit tests — SSE chunk formatting and [DONE] termination
│
├── .env.example                    # Template for environment configuration
├── .gitignore
├── Dockerfile                      # Single-worker container (uvicorn)
├── docker-compose.yml              # Multi-worker production setup (gunicorn + uvicorn workers)
├── gunicorn.conf.py                # Worker count (2×CPU+1), timeout, keepalive, worker class
├── pyproject.toml                  # Project metadata, dependencies, tool config (black, mypy)
├── requirements.txt                # Pinned production dependencies
├── requirements-dev.txt            # Dev dependencies: pytest, httpx, black, mypy, flake8
└── README.md
```

> **Note:** This proxy depends on a fork of the `perplexity-ai` library that exposes `perplexity.models.AVAILABLE_MODELS` and `perplexity.models.list_flat_models()`. See the [perplexity-ai fork changes](#perplexity-ai-library-changes) section below.

---

## perplexity-ai Library Changes

The upstream `perplexity-ai` library does not expose its model list programmatically. The following additions are required in the library before running this proxy.

### New file: `perplexity/models.py`

```python
# perplexity/models.py

AVAILABLE_MODELS = {
    'auto': [None],
    'pro': [None, 'sonar', 'gpt-5.2', 'claude-4.5-sonnet', 'grok-4-1'],
    'reasoning': [None, 'gpt-5.2-thinking', 'claude-4.5-sonnet-thinking',
                  'gemini-3.0-pro', 'kimi-k2-thinking', 'grok-4.1-reasoning'],
    'deep research': [None]
}

def get_available_models() -> dict:
    """Returns the full mode → model list mapping."""
    return AVAILABLE_MODELS

def list_flat_models() -> list[str]:
    """
    Returns a flat list of all proxy-friendly model name strings.
    None entries (mode defaults) are represented by their mode name,
    with spaces replaced by hyphens.
    """
    result = []
    for mode, models in AVAILABLE_MODELS.items():
        for model in models:
            if model is None:
                result.append(mode.replace(" ", "-"))
            else:
                result.append(model)
    return result
```

### Addition to `perplexity/client.py`

```python
from perplexity.models import get_available_models

class Client:
    # ... existing code ...

    def available_models(self) -> dict:
        """Returns available modes and their supported models."""
        return get_available_models()
```

Same addition applies to `perplexity_async/client.py`.

---

## Dynamic Model Map

At proxy startup, `mapper.py` calls `build_model_map()` which produces:

| Proxy Model Name | Perplexity Mode | Perplexity Model |
|-----------------|-----------------|-----------------|
| `auto` | `auto` | `None` (default) |
| `pro` | `pro` | `None` (default) |
| `sonar` | `pro` | `sonar` |
| `gpt-5.2` | `pro` | `gpt-5.2` |
| `claude-4.5-sonnet` | `pro` | `claude-4.5-sonnet` |
| `grok-4-1` | `pro` | `grok-4-1` |
| `reasoning` | `reasoning` | `None` (default) |
| `gpt-5.2-thinking` | `reasoning` | `gpt-5.2-thinking` |
| `claude-4.5-sonnet-thinking` | `reasoning` | `claude-4.5-sonnet-thinking` |
| `gemini-3.0-pro` | `reasoning` | `gemini-3.0-pro` |
| `kimi-k2-thinking` | `reasoning` | `kimi-k2-thinking` |
| `grok-4.1-reasoning` | `reasoning` | `grok-4.1-reasoning` |
| `deep-research` | `deep research` | `None` (default) |

When `perplexity/models.py` is updated with new models, this table regenerates automatically on next startup.

---

## Installation

### Prerequisites

- Python 3.10+
- The modified `perplexity-ai` library (with `perplexity/models.py` added — see above)

### 1. Clone and install

```bash
git clone https://github.com/yourname/perplexity-proxy.git
cd perplexity-proxy
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

`.env.example` contents:

```env
# Server
HOST=0.0.0.0
API_KEY_1=
API_KEY_2=
API_KEY_3=
PORT=8080
LOG_LEVEL=info

# Workers (gunicorn only — ignored by uvicorn single-worker mode)
# Default: 2 × CPU cores + 1
WORKERS=4

# Auth — optional
# Without this, only 'auto' mode works (no Pro/Reasoning/Deep Research)
# Value must be a JSON string of your Perplexity cookies
PERPLEXITY_COOKIES={"next-auth.session-token": "your-session-token-here"}

# Cache
CACHE_ENABLED=true
CACHE_MAX_SIZE=256
CACHE_TTL_SECONDS=300
```

### 3. Run

**Development:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

**Production (multi-worker):**
```bash
gunicorn app.main:app -c gunicorn.conf.py
```

**Docker:**
```bash
docker-compose up -d
```

---

## Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8080/v1"
)

# Chat Completions
response = client.chat.completions.create(
    model="sonar-reasoning",
    messages=[{"role": "user", "content": "How does Python's GIL affect async I/O?"}]
)
print(response.choices[0].message.content)

# Legacy Completions
completion = client.completions.create(
    model="sonar",
    prompt="Write a one-line summary of asyncio"
)
print(completion.choices[0].text)

# Streaming
for chunk in client.chat.completions.create(
    model="gpt-5.2-thinking",
    messages=[{"role": "user", "content": "Design a rate limiter in Python"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# Responses API
response = client.responses.create(
    model="sonar",
    input="Explain what a mutex is in one sentence"
)
print(response.output[0].content[0].text)
```

### opencode

```json
{
  "providers": {
    "perplexity": {
      "base_url": "http://localhost:8080/v1",
      "api_key": "dummy",
      "models": ["sonar", "sonar-reasoning", "deep-research"]
    }
  }
}
```

### Cursor

Settings → Models → Add custom model:
- **Base URL:** `http://localhost:8080/v1`
- **API Key:** `dummy`
- **Model:** `sonar-reasoning`

### Discover available models

```bash
curl http://localhost:8080/v1/models
```

### Refresh the model map

```bash
curl -X POST http://localhost:8080/v1/models/refresh \
  -H "Authorization: Bearer $REFRESH_SECRET"
```

### Health check

```bash
curl http://localhost:8080/health
# {"status": "ok", "cache_enabled": true, "authenticated": true, "api_key_auth_enabled": true, "model_count": 13}
```

### API docs

- Swagger UI: `http://localhost:8080/docs`
- OpenAPI JSON: `http://localhost:8080/openapi.json`
- ReDoc: `http://localhost:8080/redoc`

---

## Performance

| Optimization | Detail |
|---|---|
| **Async I/O** | All Perplexity requests use `perplexity_async` — zero blocking calls in the request path |
| **Multi-worker** | `gunicorn` spawns `2 × CPU + 1` `uvicorn` workers by default |
| **Persistent client** | `perplexity_async.Client` is initialized once per worker at startup and reused across all requests |
| **LRU Cache** | Identical (query + model) pairs are served from memory in < 1ms — no upstream request made |
| **Streaming** | Responses stream token-by-token via SSE — time-to-first-token is not delayed by full response buffering |
| **Retry with backoff** | Transient failures are retried with exponential backoff before surfacing an error |

---

## Error Handling

| Library Exception | HTTP Status | Meaning |
|---|---|---|
| `AuthenticationError` | `401 Unauthorized` | Cookies missing or expired |
| `RateLimitError` | `429 Too Many Requests` | Perplexity rate limit hit |
| `NetworkError` | `503 Service Unavailable` | Upstream connection failed |
| `ValidationError` | `400 Bad Request` | Invalid mode/model combination |
| `ResponseParseError` | `502 Bad Gateway` | Unexpected upstream response format |
| Unknown | `500 Internal Server Error` | Unexpected error |

---

## Known Limitations

- **No function/tool execution** — tool fields are accepted for compatibility, but Perplexity's web interface does not expose tool use
- **Responses API is simplified** — the proxy returns a clean OpenAI-shaped subset, not the full OpenAI event lifecycle
- **Single upstream query synthesis** — conversation history is reduced into one text query instead of being replayed as a full multi-turn conversation
- **No search filters** — `search_recency_filter` and `search_domain_filter` are only available in the official paid API
- **Cookie expiry** — session tokens expire periodically and must be refreshed manually
- **ToS risk** — this project uses a reverse-engineered interface; use responsibly and at your own risk

---

## Development

```bash
pip install -r requirements-dev.txt

pytest tests/ -v --cov=app
mypy app/
black app/ tests/
flake8 app/ tests/
```

---

## License

MIT — see [LICENSE](LICENSE)

---

## Disclaimer

This is an unofficial proxy. It is not affiliated with or endorsed by Perplexity AI. Use responsibly and in accordance with Perplexity's terms of service. Intended for educational and personal use only.

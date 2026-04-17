
# Phase 4 — CLIProxyAPI Integration & Deployment (Status: Planned)

**Status:** Planned
**Owner:** perplexity-proxy + CLIProxyAPI config
**Last updated:** 2026-04-17
**Delivery status:** Not started

***

## 1. Purpose

Phase 4 is the final wiring step. All previous phases produced working software in isolation — Phase 4 connects them into a single deployable system. The goal is: opencode (or any OpenAI-compatible coding agent) sends a request to CLIProxyAPI, CLIProxyAPI routes it to `perplexity-proxy`, the proxy queries Perplexity, and the response flows back — with zero manual intervention after initial setup.

This phase is primarily configuration, Docker orchestration, and end-to-end validation. No new library code is introduced.

***

## 2. What Carried Over / What Must Stay Stable

- [x] Phase 1: `perplexity/models.py` — single source of truth
- [x] Phase 2: FastAPI proxy — `/v1/models`, `/v1/chat/completions`, `/v1/responses`, `/health`
- [x] Phase 3: Dynamic model fetching, `POST /v1/models/refresh`
- [x] CLIProxyAPI binary running at `:8317`

***

## 3. Background & Motivation

### The Double-Prefix Bug (Already Known)

From the CLIProxyAPI logs:
```
404 | POST "/v1/v1/responses"        ← base-url had /v1 suffix
404 | POST "/v1/v1/chat/completions" ← same cause
200 | POST "/v1/responses"           ← after fix
200 | POST "/v1/chat/completions"    ← after fix
```

CLIProxyAPI automatically prepends `/v1` to all upstream calls. Fix: set `base-url` to `http://127.0.0.1:8080` with no suffix.

### Architecture

```
opencode / Cursor / Claude Code
          ↓  :8317
   CLIProxyAPI (Go)
     ├── Gemini CLI     (OAuth, built-in)
     ├── Claude Code    (OAuth, built-in)
     └── perplexity     (openai-compat → :8080)
                ↓
   perplexity-proxy (FastAPI)
                ↓
   Perplexity web interface
```

***

## 4. Workstream A — CLIProxyAPI Config Fix

**Status:** New

### Implementation Tasks

- [ ] Set `base-url` to `http://127.0.0.1:8080` (no `/v1` suffix)
- [ ] Add full `openai-compatibility` block for perplexity
- [ ] Map all proxy model names to human-friendly aliases
- [ ] Set `api-key` to `"dummy"` — proxy does not validate keys
- [ ] Verify hot-reload picks up config changes without restart

### Configuration / Code Reference

```yaml
# CLIProxyAPI config.yaml

openai-compatibility:
  - name: "perplexity"
    base-url: "http://127.0.0.1:8080"
    api-key-entries:
      - api-key: "dummy"
    models:
      # Auto mode
      - name: "auto"
        alias: "perplexity-auto"

      # Pro mode — web search
      - name: "sonar"
        alias: "perplexity-sonar"
      - name: "gpt-5.2"
        alias: "perplexity-gpt52"
      - name: "claude-4.5-sonnet"
        alias: "perplexity-claude45"
      - name: "grok-4.1"
        alias: "perplexity-grok41"

      # Reasoning mode
      - name: "reasoning"
        alias: "perplexity-reasoning"
      - name: "gpt-5.2-thinking"
        alias: "perplexity-gpt52-think"
      - name: "claude-4.5-sonnet-thinking"
        alias: "perplexity-claude45-think"
      - name: "gemini-3.0-pro"
        alias: "perplexity-gemini30"
      - name: "kimi-k2-thinking"
        alias: "perplexity-kimi"
      - name: "grok-4.1-reasoning"
        alias: "perplexity-grok41-reason"

      # Deep research mode
      - name: "deep-research"
        alias: "perplexity-deep-research"
```

### Acceptance Criteria

- [ ] CLIProxyAPI logs show zero `404 /v1/v1/...` errors after config update
- [ ] `GET http://localhost:8317/v1/models` includes all `perplexity-*` aliases
- [ ] CLIProxyAPI hot-reloads without restart

***

## 5. Workstream B — Docker Compose Orchestration

**Status:** New

### Implementation Tasks

- [ ] Create `docker-compose.yml` at repo root
- [ ] Define `perplexity-proxy` service — builds from `perplexity-proxy/Dockerfile`, port 8080
- [ ] Define `cli-proxy-api` service — mounts binary + config, port 8317
- [ ] Both services on shared `proxy-net` bridge network
- [ ] `cli-proxy-api` depends on `perplexity-proxy` healthcheck
- [ ] `perplexity-proxy` healthcheck: `GET /health` every 10s
- [ ] Mount `.env` into proxy for cookie injection
- [ ] Mount CLIProxyAPI `config.yaml` as read-only volume for hot-reload
- [ ] `restart: unless-stopped` on both services
- [ ] Create `docker-compose.override.yml.example` for local dev overrides

### Configuration / Code Reference

```yaml
# docker-compose.yml

services:
  perplexity-proxy:
    build:
      context: ./perplexity-proxy
      dockerfile: Dockerfile
    container_name: perplexity-proxy
    restart: unless-stopped
    ports:
      - "8080:8080"
    env_file:
      - ./perplexity-proxy/.env
    networks:
      - proxy-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s

  cli-proxy-api:
    image: ghcr.io/router-for-me/cli-proxy-api:latest
    container_name: cli-proxy-api
    restart: unless-stopped
    ports:
      - "8317:8317"
    volumes:
      - ./cli-proxy-api/config.yaml:/app/config.yaml:ro
      - ./cli-proxy-api/auth:/app/auth:ro
    depends_on:
      perplexity-proxy:
        condition: service_healthy
    networks:
      - proxy-net

networks:
  proxy-net:
    driver: bridge
```

```yaml
# docker-compose.override.yml.example (local dev)
services:
  perplexity-proxy:
    build:
      context: ./perplexity-proxy
    volumes:
      - ./perplexity-proxy:/app   # live reload for local dev
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

### Acceptance Criteria

- [ ] `docker-compose up -d` starts both services without errors
- [ ] `cli-proxy-api` waits for `perplexity-proxy` healthcheck before starting
- [ ] `curl http://localhost:8080/health` returns `200`
- [ ] `curl http://localhost:8317/v1/models` returns all models including `perplexity-*`
- [ ] Both services restart automatically after `docker kill`

***

## 6. Workstream C — Client Configuration

**Status:** New

### opencode

```json
// opencode.json
{
  "provider": {
    "perplexity": {
      "name": "Perplexity via CLIProxyAPI",
      "api": "openai",
      "models": [
        "perplexity-reasoning",
        "perplexity-deep-research",
        "perplexity-gpt52-think",
        "perplexity-claude45-think",
        "perplexity-auto"
      ],
      "options": {
        "baseURL": "http://localhost:8317/v1",
        "apiKey": "dummy"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add --transport http perplexity http://localhost:8317/v1
```

### Cursor

```json
{
  "cursor.general.customApiEndpoints": [
    {
      "name": "Perplexity",
      "apiBase": "http://localhost:8317/v1",
      "apiKey": "dummy",
      "models": ["perplexity-reasoning", "perplexity-deep-research"]
    }
  ]
}
```

### Plain OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8317/v1"
)

response = client.chat.completions.create(
    model="perplexity-reasoning",
    messages=[{"role": "user", "content": "Explain Python's GIL."}]
)
print(response.choices[0].message.content)
```

### Acceptance Criteria

- [ ] opencode completes a code generation task using `perplexity-reasoning`
- [ ] Cursor autocomplete works with `perplexity-auto`
- [ ] Streaming response visible token-by-token in opencode
- [ ] `POST /v1/responses` used by opencode returns `200` (not `404`)

***

## 7. Workstream D — End-to-End Validation

**Status:** New

### Validation Script

```bash
#!/bin/bash
# e2e-test.sh

BASE="http://localhost:8317/v1"
PROXY="http://localhost:8080"

echo "===  [modelcontextprotocol](https://modelcontextprotocol.io/docs/learn/architecture) Health Check ==="
curl -sf $PROXY/health | jq .

echo "===  [k2view](https://www.k2view.com/blog/mcp-server/) Model List ==="
curl -sf -H "Authorization: Bearer dummy" $BASE/models \
  | jq '.data[].id' | grep perplexity

echo "=== [3] Chat Completions (non-streaming) ==="
curl -sf $BASE/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{"model":"perplexity-auto","messages":[{"role":"user","content":"Reply with: OK"}]}' \
  | jq '.choices[0].message.content'

echo "=== [4] Responses API (non-streaming) ==="
curl -sf $BASE/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{"model":"perplexity-auto","input":"Reply with: OK"}' \
  | jq '.output[0].content[0].text'

echo "=== [5] Streaming (chat/completions) ==="
curl -sf $BASE/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{"model":"perplexity-auto","messages":[{"role":"user","content":"Count to 3."}],"stream":true}'

echo "=== [6] Cache Verification ==="
QUERY='{"model":"perplexity-auto","messages":[{"role":"user","content":"What is 1+1?"}]}'
curl -sf $BASE/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" -d "$QUERY" > /dev/null
time curl -sf $BASE/chat/completions -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" -d "$QUERY" | jq '.choices[0].message.content'

echo "=== All checks done ==="
```

### Acceptance Criteria

- [ ] All 6 checks pass without errors
- [ ] Zero `404` in CLIProxyAPI logs during validation run
- [ ] Zero `500` in perplexity-proxy logs during validation run
- [ ] Check  produces `data: {...}` SSE lines followed by `data: [DONE]` [modelcontextprotocol](https://modelcontextprotocol.io/docs/learn/architecture)
- [ ] Check  second request completes in < 5ms (cache hit confirmed) [k2view](https://www.k2view.com/blog/mcp-server/)

***

## 8. Workstream E — Operational Runbook

**Status:** New

### Start / Stop

```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# Follow logs
docker-compose logs -f perplexity-proxy
docker-compose logs -f cli-proxy-api

# Restart proxy only (e.g. after cookie update)
docker-compose restart perplexity-proxy
```

### Update Cookies

```bash
# 1. Edit perplexity-proxy/.env — update PERPLEXITY_COOKIES value
# 2. Restart proxy
docker-compose restart perplexity-proxy
# 3. Verify
curl http://localhost:8080/health | jq .authenticated
```

### Trigger Model Refresh (Phase 3)

```bash
curl -X POST http://localhost:8080/v1/models/refresh \
  -H "Authorization: Bearer $REFRESH_SECRET"
```

### Add a New Model (Static Fallback)

```
1. Edit MODEL_PREFERENCE_MAP in perplexity/models.py
2. Rebuild proxy image:
   docker-compose build perplexity-proxy
3. Restart:
   docker-compose restart perplexity-proxy
4. Add alias to CLIProxyAPI config.yaml
   (hot-reloads automatically — no restart needed)
5. Verify:
   curl http://localhost:8317/v1/models | jq '.data[].id' | grep perplexity
```

### Check Cache Stats

```bash
curl http://localhost:8080/health | jq '{cache_enabled, model_count}'
```

***

## 9. Workstream F — Pre-Merge Audit Checklist

**Status:** New

### 9.1 Config audit

- [ ] `base-url` in CLIProxyAPI config has no `/v1` suffix
- [ ] All 12 proxy models have aliases in config
- [ ] `api-key: "dummy"` set (proxy accepts any key)
- [ ] `REFRESH_SECRET` set to non-default value in production `.env`
- [ ] `PERPLEXITY_COOKIES` set correctly in production `.env`

### 9.2 Network audit

- [ ] `perplexity-proxy` reachable from `cli-proxy-api` container via `proxy-net`
- [ ] Port 8080 not publicly exposed in production (internal only)
- [ ] Port 8317 exposed only to localhost unless remote access intended

### 9.3 Regression audit

- [ ] All Phase 1 tests pass
- [ ] All Phase 2 tests pass
- [ ] All Phase 3 tests pass
- [ ] End-to-end validation script (Workstream D) passes fully

### 9.4 Failure mode audit

- [ ] `perplexity-proxy` crash → CLIProxyAPI returns `503` to client (not hang)
- [ ] Perplexity network unreachable → proxy returns `503` within 15s (retry timeout)
- [ ] Invalid model name → proxy returns `400` with valid model list
- [ ] Expired cookies → proxy returns `401`

***

## 10. Combined Implementation Order

```
1. Fix CLIProxyAPI config.yaml (Workstream A) — immediate, no code change
2. Verify fix by checking logs — zero /v1/v1/ 404s
3. Create docker-compose.yml (Workstream B)
4. Test docker-compose up — both services healthy
5. Configure opencode / Cursor (Workstream C)
6. Run e2e-test.sh (Workstream D) — all 6 checks pass
7. Write runbook (Workstream E)
8. Complete pre-merge audit (Workstream F)
9. Tag release: v1.0.0
```

***

## 11. Definition of Done

Phase 4 is complete when **all** of the following are true simultaneously.

### 11.1 Integration layer

- [x] Phases 1–3 Definitions of Done satisfied
- [ ] CLIProxyAPI config has no double-prefix bug
- [ ] All 12 proxy models routable via CLIProxyAPI aliases
- [ ] `docker-compose up -d` starts both services cleanly

### 11.2 Client layer

- [ ] opencode routes to `perplexity-reasoning` successfully
- [ ] `POST /v1/responses` returns `200` from opencode (not `404`)
- [ ] Streaming works token-by-token

### 11.3 Validation layer

- [ ] All 6 e2e-test.sh checks pass
- [ ] Zero `404`/`500` errors in combined logs during validation
- [ ] Cache hit verified (< 5ms second response)

### 11.4 Operations layer

- [ ] Runbook documented
- [ ] Cookie rotation procedure documented
- [ ] Model refresh procedure documented

***

## 12. Complete System File Map

```
.
├── docker-compose.yml
├── docker-compose.override.yml.example
├── e2e-test.sh
│
├── perplexity-ai/                    ← Phase 1 (library)
│   ├── perplexity/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── models.py                 ← Phase 1: MODEL_PREFERENCE_MAP
│   │   ├── model_fetcher.py          ← Phase 3: live fetch
│   │   └── exceptions.py
│   └── perplexity_async/
│       ├── __init__.py
│       └── client.py
│
├── perplexity-proxy/                 ← Phase 2 (FastAPI)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   ← lifespan, app factory
│   │   ├── config.py                 ← pydantic-settings
│   │   ├── router.py                 ← all endpoints
│   │   ├── models.py                 ← Pydantic schemas
│   │   ├── client.py                 ← perplexity_async wrapper
│   │   ├── mapper.py                 ← proxy model name → mode/model
│   │   ├── cache.py                  ← LRU cache
│   │   └── streaming.py              ← SSE formatters
│   ├── tests/
│   ├── Dockerfile
│   ├── gunicorn.conf.py
│   ├── requirements.txt
│   └── .env.example
│
└── cli-proxy-api/                    ← Phase 4 (config only)
    ├── config.yaml                   ← openai-compatibility block
    └── auth/                         ← CLIProxyAPI OAuth tokens
```

***

## 13. Compact Mental Model

### 13.1 All Four Phases

| Phase | What | Output |
|---|---|---|
| **1** | Extract model definitions into `models.py` | Single source of truth |
| **2** | Build FastAPI proxy with all OpenAI endpoints | Working proxy at `:8080` |
| **3** | Live model fetching + `/v1/models/refresh` | Self-updating model list |
| **4** | CLIProxyAPI config + Docker Compose + e2e test | Production-ready system |

### 13.2 Key Takeaway

Phase 4 has zero new code. Its value is integration correctness. The most common failure point is the `base-url` double-prefix bug — fix that first, everything else follows. The e2e validation script is the single source of truth for "done": if all 6 checks pass with zero 404s in the logs, the system is working correctly end to end.

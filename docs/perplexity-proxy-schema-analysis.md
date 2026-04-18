# perplexity-proxy Schema Analysis

This document describes the current API schema and runtime behavior of **perplexity-proxy** as implemented today.
It is meant as a practical reference for compatibility, not as a theoretical OpenAI spec.

## 1) Overview

`perplexity-proxy` is a **FastAPI** application that exposes an OpenAI-shaped surface and forwards requests to Perplexity through the reverse-engineered `perplexity_async` client.

The current request flow is:

1. FastAPI receives an OpenAI-style request.
2. Pydantic validates and normalizes the payload.
3. The requested proxy model is resolved into a Perplexity mode/model pair.
4. The request is reduced to a text query.
5. A cache lookup may satisfy the request.
6. Otherwise the query is sent upstream.
7. The upstream result is reshaped into an OpenAI-compatible response.
8. Streaming requests are converted into SSE output.

The proxy is therefore a **single-provider compatibility layer**, not a multi-backend router.

## 2) Exposed endpoints

### Public routes

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/models` | List available proxy model IDs |
| POST | `/v1/chat/completions` | OpenAI Chat Completions compatibility endpoint |
| POST | `/v1/completions` | Legacy text completion shim |
| POST | `/v1/responses` | OpenAI Responses compatibility endpoint |
| GET | `/health` | Health and feature status |
| POST | `/v1/models/refresh` | Refresh the dynamic model map |

### FastAPI docs routes

- `/openapi.json`
- `/docs`
- `/redoc`

## 3) Schema layer

The request and response models live in `app/models.py` and are intentionally permissive.
Unknown fields are generally ignored, which keeps the proxy tolerant of OpenAI client drift.

### 3.1 Chat Completions request

```python
class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
```

Notes:

- `model` and `messages` are required.
- `stream` controls SSE vs non-streaming output.
- Tool-related fields are accepted and carried through cache-key generation, but the proxy does not execute tool calls.
- Extra fields are ignored.

#### Message schema

```python
class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None
```

Supported content shapes:

- plain string
- list of arbitrary JSON values
- null

The proxy converts these into text when building the upstream query.

### 3.2 Chat Completions response

```python
class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage | dict[str, Any] | None = None
```

Current behavior:

- one assistant choice
- `finish_reason = "stop"`
- zeroed token usage counters

### 3.3 Chat streaming chunk

```python
class StreamChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
```

The streaming helper renders these as SSE `data:` events.

### 3.4 Legacy completions request/response

```python
class CompletionsRequest(BaseModel):
    model: str
    prompt: str | list[str] | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    logprobs: int | None = None
    top_logprobs: int | None = None
    echo: bool | None = None
```

```python
class CompletionsResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionsChoice]
    usage: ChatUsage | None = None
```

This route is implemented as a shim over the same upstream search flow used by chat completions.

### 3.5 Responses request/response

```python
class ResponsesRequest(BaseModel):
    model: str
    input: str | list[ChatMessage]
    instructions: str | None = None
    stream: bool = False
    temperature: float | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
```

```python
class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    output: list[ResponsesOutputMessage]
    usage: ResponsesUsage | dict[str, Any] | None = None
```

Notes:

- `input` may be a string or a message list.
- `instructions` is prepended to the query text when present.
- `previous_response_id` is accepted and included in the cache key, but the current proxy does not maintain full response state.
- Tool-related fields are accepted but not executed.

### 3.6 Model list / health / refresh

```python
class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]
```

```python
class HealthResponse(BaseModel):
    status: str
    cache_enabled: bool
    authenticated: bool
    api_key_auth_enabled: bool
    model_count: int

class RefreshResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_count: int
    models: list[str]
```

These are proxy-specific management schemas rather than OpenAI schemas.

## 4) Route behavior

### 4.1 `GET /v1/models`

Returns the current proxy model list generated from the upstream model map.

Characteristics:

- `created` is set to the current timestamp at response time.
- `owned_by` is always `perplexity`.
- List ordering follows the current model map.

### 4.2 `POST /v1/chat/completions`

The main compatibility endpoint.

#### Request flow

1. Normalize the request body into `ChatRequest`.
2. Resolve `req.model` into `(mode, model)`.
3. Build a text query from the message list.
4. Build a cache key from the query, model, and request-shaping fields.
5. If `stream=true`, stream SSE output.
6. Otherwise return cached output or call upstream.

#### Query extraction

The proxy does not replay a full conversation.
It reduces the message list into a single upstream query.

Current behavior:

- the first system message is preserved as a prefix
- user messages become the primary query text
- assistant messages are included as annotated context text
- if there are multiple turns, they are joined into one text block

#### Non-streaming shape

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "sonar",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

#### Streaming shape

The stream emits SSE frames containing `chat.completion.chunk` payloads, followed by a final `data: [DONE]` marker.

### 4.3 `POST /v1/completions`

Legacy text completion support is present and implemented as a shim.

#### Request flow

1. Normalize the payload into `CompletionsRequest`.
2. Resolve the target model.
3. Convert `prompt` into query text.
4. Stream or return cached/non-cached output.

This endpoint does not implement OpenAI-style logprobs or echo behavior.
Those fields are accepted for compatibility but not acted on.

### 4.4 `POST /v1/responses`

Simplified OpenAI Responses-compatible endpoint.

#### Request flow

1. Resolve `req.model`.
2. Convert `input` into query text.
3. Prepend `instructions` when provided.
4. Build a cache key including request-shaping fields.
5. Stream or return a synthesized `ResponsesResponse`.

#### Input handling

- string input is used directly
- list input is reduced via message extraction
- other values are stringified

#### Non-streaming shape

```json
{
  "id": "resp-...",
  "object": "response",
  "created_at": 1710000000,
  "model": "sonar",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "..."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  }
}
```

#### Streaming shape

The helper emits SSE JSON events for:

1. `response.created`
2. `response.output_text.delta`
3. `response.output_text.done`
4. `response.completed`
5. `data: [DONE]`

This is a simplified Responses stream, not a full OpenAI event lifecycle.

## 5) Streaming implementation details

The streaming helpers live in `app/streaming.py`.

### 5.1 Upstream chunk handling

Perplexity upstream data arrives as raw dicts from `perplexity_async`.
These dicts may contain internal state blobs as well as real answer text.

The current rules are:

- internal blobs are filtered before extraction
- text is extracted from `blocks[].markdown_block.chunks[]` when present
- for blocks, `ask_text_0_markdown` is preferred over `ask_text`
- the first matching text block is returned immediately
- legacy top-level keys (`delta`, `content`, `text`, `answer`) remain as fallback support
- raw dicts are never stringified as content

### 5.2 Internal blob filtering

Internal Perplexity state is dropped when a chunk carries keys like:

- `backend_uuid`
- `frontend_context_uuid`
- `classifier_results`
- `context_uuid`
- `read_write_token`
- `search_implementation_mode`
- `final_sse_message`
- `message_mode`

This prevents internal protocol blobs from leaking to the client.

### 5.3 Text extraction behavior

The extraction order is intentionally conservative:

1. `blocks[].markdown_block.chunks[]` for `ask_text_0_markdown` or `ask_text`
2. legacy top-level string fields
3. `None` if no real text is present

That design avoids both blob leakage and duplicated text assembly from multiple blocks in the same chunk.

### 5.4 SSE framing

Both stream helpers emit SSE using the same framing:

- JSON payloads are sent as `data: <json>\n\n`
- stream termination is `data: [DONE]\n\n`

## 6) Error handling

### 6.1 HTTP error shape

Route errors are returned as JSON payloads with an `error` object.

### 6.2 Known upstream mappings

The proxy maps Perplexity exceptions to HTTP statuses such as:

- authentication issues → `401`
- rate limiting → `429`
- malformed/invalid inputs → `400`
- upstream failures → `502` or `503`

### 6.3 Route-level errors

- invalid model → `400`
- missing or bad API key → `401`
- upstream Perplexity failures → `502`
- unexpected proxy failure → `500`

### 6.4 Global fallback handler

A generic FastAPI handler returns a proxy-specific internal error envelope when something escapes route handling.

## 7) Authentication and management

### 7.1 Cookie auth

`PERPLEXITY_COOKIES` is used to authenticate the upstream Perplexity session.
If configured, startup performs an auth health check.

### 7.2 API-key auth

Optional bearer auth can be enforced for `/v1/*` routes when any of these are configured:

- `API_KEY_1`
- `API_KEY_2`
- `API_KEY_3`

When enabled:

- missing bearer token → `401`
- invalid bearer token → `401`
- exempt routes still remain accessible

### 7.3 Model refresh secret

`POST /v1/models/refresh` requires a bearer token derived from `REFRESH_SECRET`.

## 8) Model map behavior

The exposed model list comes from the dynamic upstream model map.
Each proxy-visible model is converted into a model object with:

- `id` = proxy model name
- `object` = `model`
- `created` = current timestamp
- `owned_by` = `perplexity`

Model resolution converts a proxy model name into a Perplexity mode/model pair.

## 9) Cache behavior

The cache stores serialized response payloads keyed by a hash of:

- query text
- requested proxy model
- request type
- selected request-shaping fields

Included fields vary by route, but the cache key now incorporates more than just model + query.

### Important limitation

The proxy still does not reproduce the full OpenAI request semantics.
So two requests that normalize to the same upstream query can still share cache entries even if the client payload differed in unsupported ways.

## 10) Compatibility gaps

### Present

- `/v1/models`
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/responses`
- SSE streaming
- dynamic model discovery
- cache-backed non-streaming responses
- optional auth and health endpoints

### Simplified or missing

- true OpenAI Responses event lifecycle parity
- tool-call execution
- full multi-turn conversation replay
- websocket-based Responses support
- logprobs / top logprobs behavior
- assistant/tool message lifecycle fidelity
- complete request-shape preservation in responses

## 11) Current practical contract

### `/v1/chat/completions`

- accepts OpenAI-style chat payloads
- returns one assistant answer
- streams chunked deltas when requested
- terminates with a final stop chunk and `[DONE]`

### `/v1/completions`

- accepts legacy prompt input
- returns one text completion choice
- streams via the same upstream content extraction path

### `/v1/responses`

- accepts either string input or message lists
- prepends instructions when present
- returns a simplified `response` object
- streams simplified SSE events

## 12) Summary

`perplexity-proxy` is a lightweight OpenAI compatibility layer over Perplexity.
It now includes:

- chat completions
- legacy completions
- responses
- model listing
- refresh and health endpoints

The most important implementation detail is the streaming path:
Perplexity internal blobs are filtered early, and real text is extracted from `blocks[].markdown_block.chunks[]` before any SSE formatting occurs.

That keeps the proxy stable, avoids blob leakage, and preserves user-visible text exactly once.

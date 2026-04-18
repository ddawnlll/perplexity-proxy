from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict


class ProxyModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ChatMessage(ProxyModel):
    role: str
    content: str | list[Any] | None = None


class ChatRequest(ProxyModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None


class ChatResponseMessage(ProxyModel):
    role: str
    content: str | list[Any] | None = None
    tool_calls: list[Any] | None = None


class ChatChoice(ProxyModel):
    index: int
    message: ChatResponseMessage
    finish_reason: str | None = None


class ChatUsage(ProxyModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionsRequest(ProxyModel):
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


class CompletionsChoice(ProxyModel):
    index: int
    text: str
    finish_reason: str | None = None


class CompletionsResponse(ProxyModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionsChoice]
    usage: ChatUsage | None = None


class ChatResponse(ProxyModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage | dict[str, Any] | None = None


class StreamDelta(ProxyModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[Any] | None = None


class StreamChoice(ProxyModel):
    index: int
    delta: StreamDelta
    finish_reason: str | None = None


class StreamChunk(ProxyModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class ResponsesUsage(ProxyModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


ResponsesInput: TypeAlias = str | list[ChatMessage]


class ResponsesRequest(ProxyModel):
    model: str
    input: ResponsesInput
    instructions: str | None = None
    stream: bool = False
    temperature: float | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    tools: list[Any] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None


class ResponsesOutputText(ProxyModel):
    type: Literal["text"] = "text"
    text: str


class ResponsesOutputMessage(ProxyModel):
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ResponsesOutputText]


class ResponsesResponse(ProxyModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    output: list[ResponsesOutputMessage]
    usage: ResponsesUsage | dict[str, Any] | None = None


class ResponsesStreamEvent(ProxyModel):
    event: str | None = None
    data: Any | None = None


class ModelObject(ProxyModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(ProxyModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class HealthResponse(ProxyModel):
    status: str
    cache_enabled: bool
    authenticated: bool
    api_key_auth_enabled: bool
    model_count: int


class RefreshResponse(ProxyModel):
    status: Literal["ok"] = "ok"
    model_count: int
    models: list[str]


__all__ = [
    "ChatChoice",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatResponseMessage",
    "ChatUsage",
    "CompletionsChoice",
    "CompletionsRequest",
    "CompletionsResponse",
    "HealthResponse",
    "RefreshResponse",
    "ModelList",
    "ModelObject",
    "ProxyModel",
    "ResponsesInput",
    "ResponsesOutputMessage",
    "ResponsesOutputText",
    "ResponsesRequest",
    "ResponsesResponse",
    "ResponsesStreamEvent",
    "ResponsesUsage",
    "StreamChoice",
    "StreamChunk",
    "StreamDelta",
]

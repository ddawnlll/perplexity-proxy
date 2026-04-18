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


class ChatResponseMessage(ProxyModel):
    role: str
    content: str | list[Any] | None = None


class ChatChoice(ProxyModel):
    index: int
    message: ChatResponseMessage
    finish_reason: str | None = None


class ChatUsage(ProxyModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
    model_count: int


__all__ = [
    "ChatChoice",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatResponseMessage",
    "ChatUsage",
    "HealthResponse",
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

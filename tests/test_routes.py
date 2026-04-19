from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.router import _clean_user_content, _extract_query
from app.models import ChatResponse, ResponsesResponse
from app.state import follow_up_store
from perplexity.exceptions import AuthenticationError, PerplexityError, RateLimitError


@pytest.fixture
def client(mocker):
    mocker.patch("app.main.init_client", new=AsyncMock(return_value=None))
    mocker.patch("app.main.close_client", new=AsyncMock(return_value=None))
    mocker.patch("app.main.check_perplexity_session", new=AsyncMock(return_value={"ok": True, "authenticated": True, "status_code": 200}))
    settings.API_KEY_1 = ""
    settings.API_KEY_2 = ""
    settings.API_KEY_3 = ""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clear_follow_up_store():
    asyncio.run(follow_up_store.clear())
    yield
    asyncio.run(follow_up_store.clear())


@pytest.fixture
def cache_mocks(mocker):
    mocker.patch("app.router.cache.get", new=AsyncMock(return_value=None))
    mocker.patch("app.router.cache.set", new=AsyncMock(return_value=None))


@pytest.fixture
def search_mock(mocker):
    return mocker.patch("app.router.search", new=AsyncMock())


async def _stream_gen():
    yield "Hello"
    yield " world"


def _parse_sse_json_chunks(body: str) -> list[dict]:
    events: list[dict] = []
    for chunk in body.split("\n\n"):
        if not chunk.startswith("data: "):
            continue
        payload = chunk.removeprefix("data: ")
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def test_get_v1_models_returns_list(client):
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert isinstance(payload["data"], list)


def test_get_health_returns_expected_fields(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {"status", "cache_enabled", "authenticated", "api_key_auth_enabled", "model_count"}
    assert payload["status"] == "ok"
    assert payload["api_key_auth_enabled"] is False


def test_chat_completions_success_returns_message_content(client, cache_mocks, search_mock):
    search_mock.return_value = "Paris is the capital of France."

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "Paris is the capital of France."


def test_chat_completions_wraps_roo_requests_as_attempt_completion_tool_call(client, cache_mocks, search_mock):
    search_mock.return_value = "All done."

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "attempt_completion"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "result": "All done."
    }


def test_chat_completions_roo_request_reads_file_before_editing(client, cache_mocks, search_mock):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "can you edit calculator.py"}],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "calculator.py"
    }
    search_mock.assert_not_awaited()


def test_chat_completions_roo_request_uses_injected_read_file_block_for_write(client, cache_mocks, search_mock):
    search_mock.return_value = """```python
def add(a, b):
    return a + b

def divide(a, b):
    return a / b
```"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<user_message>\nadd divide function to calculator.py\n</user_message>",
                        },
                        {
                            "type": "text",
                            "text": (
                                "[read_file for 'calculator.py']\n"
                                "File: calculator.py\n"
                                " 1 | def add(a, b):\n"
                                " 2 |     return a + b\n"
                            ),
                        },
                        {"type": "text", "text": "<environment_details>cwd=/tmp/project</environment_details>"},
                    ],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    args = json.loads(payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "write_to_file"
    assert args["path"] == "calculator.py"
    assert args["content"] == "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b"
    assert args["line_count"] == 5

    query = search_mock.await_args.args[0]
    assert "Current file content:" in query
    assert "[read_file for 'calculator.py']" in query
    assert "User request: add divide function to calculator.py" in query


def test_chat_completions_roo_request_attempts_completion_after_write(client, cache_mocks, search_mock):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_to_file",
                                "arguments": (
                                    "{\"path\":\"calculator.py\",\"content\":\"def add(a, b):\\n    return a + b\\n\","
                                    "\"line_count\":2}"
                                ),
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "File updated successfully"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<environment_details>cwd=/tmp/project</environment_details>"}
                    ],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    args = json.loads(payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "attempt_completion"
    assert args["result"] == "The file `calculator.py` has been updated successfully."
    search_mock.assert_not_awaited()


def test_chat_completions_roo_request_uses_last_read_path_for_write_to_file(client, cache_mocks, search_mock):
    search_mock.return_value = """[1]
def add(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
programiz+1"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {
                    "role": "system",
                    "content": "SYSTEM INFORMATION\nOS: macOS\n====\nCurrent Workspace\n/tmp/project",
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\"calculator.py\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "def add(a, b):\n    return a + b\n",
                },
                {"role": "user", "content": "add a divide function"},
            ],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "write_to_file"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "calculator.py",
        "content": "def add(a, b):\n    return a + b\n\n"
        "def divide(a, b):\n"
        "    if b == 0:\n"
        '        raise ValueError("Cannot divide by zero")\n'
        "    return a / b",
        "line_count": 7,
    }

    query = search_mock.await_args.args[0]
    assert "COMPLETE file content" in query
    assert "[CURRENT FILE CONTENT]" in query
    assert "def add(a, b):" in query
    assert "[TASK]" in query


def test_chat_completions_roo_write_to_file_strips_code_fences_and_citations(client, cache_mocks, search_mock):
    search_mock.return_value = """**calculator.py**
```python
[1]
def add(a, b):
    return a + b

def divide(a, b):
    return a / b
programiz+1
```"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\":\"calculator.py\"}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "def add(a, b):\n    return a + b\n",
                },
                {"role": "user", "content": "add a divide function"},
            ],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    args = json.loads(payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "calculator.py"
    assert args["content"] == "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b"
    assert args["line_count"] == 5


def test_extract_query_prefers_last_user_message_from_roo_text_blocks():
    messages = [
        {"role": "system", "content": "You are Roo."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<user_message>\nhello!\n</user_message>"},
                {"type": "text", "text": "<environment_details>old env</environment_details>"},
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"calculator.py\"}"},
                }
            ],
        },
        {"role": "tool", "content": "def add(a, b):\n    return a + b\n"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "<user_message>\nadd divide function to calculator.py\n</user_message>\n"
                        "<environment_details>cwd=/tmp/project</environment_details>"
                    ),
                }
            ],
        },
    ]

    assert _extract_query(messages) == "add divide function to calculator.py"


def test_extract_query_skips_tool_result_only_user_messages():
    messages = [
        {"role": "user", "content": "hello!"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_123", "content": "def add(a, b):\n    return a + b\n"},
                {"type": "text", "text": "<environment_details>cwd=/tmp/project</environment_details>"},
            ],
        },
    ]

    assert _extract_query(messages) == "hello!"


def test_clean_user_content_extracts_inner_user_message():
    text = (
        "<user_message>\nadd divide function\n</user_message>\n"
        "<environment_details>\n# Current Mode\ncode\n</environment_details>"
    )

    assert _clean_user_content(text) == "add divide function"


def test_chat_completions_normalizes_responses_style_payload(client, cache_mocks, search_mock):
    search_mock.return_value = "Be concise and helpful."

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "instructions": "Be concise",
            "input": "Hello there",
            "stream": False,
        },
    )

    assert response.status_code == 200
    assert search_mock.await_count == 1
    query = search_mock.await_args.args[0]
    assert query.startswith("Be concise")
    assert "Hello there" in query


def test_chat_completions_prepends_system_and_history_to_query(client, cache_mocks, search_mock):
    search_mock.return_value = "Bonjour"

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {"role": "system", "content": "Reply in French"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ],
            "stream": False,
        },
    )

    assert response.status_code == 200
    query = search_mock.await_args.args[0]
    assert query.startswith("Reply in French")
    assert "[assistant]: Hi" in query
    assert query.endswith("How are you?")


def test_completions_shim_returns_text_completion_object(client, cache_mocks, search_mock):
    search_mock.return_value = "Counted to three."

    response = client.post(
        "/v1/completions",
        json={"model": "gpt-5.2", "prompt": "Count to 3", "stream": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == "Counted to three."


def test_completions_shim_stream_returns_text_completion_chunks(client, cache_mocks, search_mock):
    search_mock.return_value = _stream_gen()

    response = client.post(
        "/v1/completions",
        json={"model": "gpt-5.2", "prompt": "Count to 3", "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "text_completion" in response.text


def test_responses_success_string_input_returns_text_output(client, cache_mocks, search_mock):
    search_mock.return_value = "Paris is the capital of France."

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.2", "input": "What is the capital of France?", "stream": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "Paris is the capital of France."


def test_responses_success_list_input_returns_text_output(client, cache_mocks, search_mock):
    search_mock.return_value = "Paris is the capital of France."

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.2",
            "input": [{"role": "user", "content": "What is the capital of France?"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["output"][0]["content"][0]["text"] == "Paris is the capital of France."


def test_responses_previous_response_id_reuses_upstream_follow_up(client, cache_mocks, search_mock):
    search_mock.side_effect = [
        {"answer": "First answer", "backend_uuid": "backend-1", "attachments": ["file-1"]},
        {"answer": "Second answer", "backend_uuid": "backend-2", "attachments": ["file-1"]},
    ]

    first = client.post(
        "/v1/responses",
        json={"model": "gpt-5.2", "input": "Hello", "stream": False},
    )
    assert first.status_code == 200
    first_id = first.json()["id"]

    second = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.2",
            "input": "Continue",
            "previous_response_id": first_id,
            "stream": False,
        },
    )
    assert second.status_code == 200
    assert search_mock.await_count == 2
    assert search_mock.await_args_list[1].kwargs["follow_up"] == {
        "backend_uuid": "backend-1",
        "attachments": ["file-1"],
    }


def test_chat_multi_turn_reuses_follow_up_from_previous_assistant_turn(client, cache_mocks, search_mock):
    search_mock.side_effect = [
        {"answer": "Hi there", "backend_uuid": "backend-1", "attachments": []},
        {"answer": "Still here", "backend_uuid": "backend-2", "attachments": []},
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert first.status_code == 200
    assistant_text = first.json()["choices"][0]["message"]["content"]

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": "Are you still there?"},
            ],
            "stream": False,
        },
    )
    assert second.status_code == 200
    assert search_mock.await_count == 2
    assert search_mock.await_args_list[1].kwargs["follow_up"] == {
        "backend_uuid": "backend-1",
        "attachments": [],
    }


def test_roo_chat_multi_turn_reuses_follow_up_across_tool_turns(client, cache_mocks, search_mock):
    search_mock.side_effect = [
        {"answer": "def add(a, b):\n    return a + b\n", "backend_uuid": "backend-1", "attachments": []},
        {"answer": "from dataclasses import dataclass\n", "backend_uuid": "backend-2", "attachments": []},
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<user_message>\ncreate calculator/core.py and calculator/history.py\n</user_message>",
                        },
                        {"type": "text", "text": "<environment_details>cwd=/tmp/project</environment_details>"},
                    ],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )
    assert first.status_code == 200
    first_call = first.json()["choices"][0]["message"]["tool_calls"][0]
    assert first_call["function"]["name"] == "write_to_file"

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<user_message>\ncreate calculator/core.py and calculator/history.py\n</user_message>",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "tool_calls": [first_call],
                },
                {"role": "tool", "content": "File written"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<environment_details>cwd=/tmp/project</environment_details>"},
                    ],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": False,
        },
    )
    assert second.status_code == 200
    assert search_mock.await_count == 2
    second_call = second.json()["choices"][0]["message"]["tool_calls"][0]
    assert second_call["function"]["name"] == "write_to_file"
    assert json.loads(second_call["function"]["arguments"])["path"] == "calculator/history.py"
    assert search_mock.await_args_list[1].kwargs["follow_up"] == {
        "backend_uuid": "backend-1",
        "attachments": [],
    }


def test_streamed_responses_store_follow_up_for_previous_response_id(client, cache_mocks, search_mock):
    async def streamed_first():
        yield {"answer": "H", "backend_uuid": "backend-1", "attachments": ["file-1"]}
        yield {"answer": "Hi", "backend_uuid": "backend-1", "attachments": ["file-1"]}

    search_mock.side_effect = [
        streamed_first(),
        {"answer": "Second answer", "backend_uuid": "backend-2", "attachments": ["file-1"]},
    ]

    first = client.post(
        "/v1/responses",
        json={"model": "gpt-5.2", "input": "Hello", "stream": True},
    )
    assert first.status_code == 200
    streamed_events = _parse_sse_json_chunks(first.text)
    first_id = streamed_events[0]["response"]["id"]

    second = client.post(
        "/v1/responses",
        json={
            "model": "gpt-5.2",
            "input": "Continue",
            "previous_response_id": first_id,
            "stream": False,
        },
    )
    assert second.status_code == 200
    assert search_mock.await_count == 2
    assert search_mock.await_args_list[1].kwargs["follow_up"] == {
        "backend_uuid": "backend-1",
        "attachments": ["file-1"],
    }


def test_chat_unknown_model_returns_400(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "unknown-model", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 400


def test_responses_unknown_model_returns_400(client):
    response = client.post(
        "/v1/responses",
        json={"model": "unknown-model", "input": "Hi"},
    )

    assert response.status_code == 400


def test_chat_stream_returns_event_stream(client, cache_mocks, search_mock):
    search_mock.return_value = _stream_gen()

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_chat_stream_wraps_roo_requests_as_attempt_completion_tool_call(client, cache_mocks, search_mock):
    search_mock.return_value = "Hello world"

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_json_chunks(response.text)
    final_choice = events[-1]["choices"][0]
    assert final_choice["finish_reason"] == "tool_calls"
    assert final_choice["delta"]["tool_calls"][0]["function"]["name"] == "attempt_completion"
    assert json.loads(final_choice["delta"]["tool_calls"][0]["function"]["arguments"]) == {
        "result": "Hello world"
    }


def test_chat_stream_roo_injected_read_file_returns_write_to_file_sse(client, cache_mocks, search_mock):
    search_mock.return_value = """```python
def add(a, b):
    return a + b

def divide(a, b):
    return a / b
```"""

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<user_message>\nhello, can you edit 'calculator.py' file please add a new function\n</user_message>",
                        },
                        {
                            "type": "text",
                            "text": (
                                "[read_file for 'calculator.py']\n"
                                "File: calculator.py\n"
                                " 1 | def add(a, b):\n"
                                " 2 |     return a + b\n"
                            ),
                        },
                    ],
                }
            ],
            "tools": [{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_json_chunks(response.text)
    final_choice = events[-1]["choices"][0]
    assert final_choice["finish_reason"] == "tool_calls"
    assert final_choice["delta"]["tool_calls"][0]["function"]["name"] == "write_to_file"
    args = json.loads(final_choice["delta"]["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "calculator.py"
    assert args["content"] == "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b"
    assert args["line_count"] == 5


def test_responses_stream_returns_event_stream(client, cache_mocks, search_mock):
    search_mock.return_value = _stream_gen()

    response = client.post(
        "/v1/responses",
        json={"model": "gpt-5.2", "input": "Hello", "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_stream_returns_json_error_when_upstream_fails_before_first_byte(client, cache_mocks, mocker):
    async def broken_gen():
        if False:
            yield "never"
        raise PerplexityError("stream failed before first byte")

    mocker.patch("app.router.search", new=AsyncMock(return_value=broken_gen()))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.2", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )

    assert response.status_code == 502
    payload = response.json()
    assert payload["error"]["type"] == "upstream_error"


def test_cache_hit_skips_search(client, mocker):
    cached_chat = ChatResponse(
        id="chatcmpl-cached",
        object="chat.completion",
        created=1,
        model="gpt-5.2",
        choices=[],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    ).model_dump_json()
    get_mock = mocker.patch("app.router.cache.get", new=AsyncMock(return_value=cached_chat))
    set_mock = mocker.patch("app.router.cache.set", new=AsyncMock(return_value=None))
    search_mock = mocker.patch("app.router.search", new=AsyncMock())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5.2",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )

    assert response.status_code == 200
    search_mock.assert_not_called()
    get_mock.assert_awaited()
    set_mock.assert_not_awaited()


def test_authentication_error_maps_to_401(client, mocker):
    class FakeClient:
        async def search(self, *args, **kwargs):
            raise AuthenticationError("bad auth")

    mocker.patch("app.client.get_client", return_value=FakeClient())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.2", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 401


def test_rate_limit_error_maps_to_429(client, mocker):
    class FakeClient:
        async def search(self, *args, **kwargs):
            raise RateLimitError("slow down")

    mocker.patch("app.client.get_client", return_value=FakeClient())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5.2", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 429


def test_router_has_no_double_prefixed_routes():
    paths = [route.path for route in app.router.routes]
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/responses" in paths
    assert "/health" in paths
    assert not any(path.startswith("/v1/v1/") for path in paths)


def test_openapi_and_swagger_endpoints_are_exposed(client):
    openapi_response = client.get("/openapi.json")
    docs_response = client.get("/docs")
    redoc_response = client.get("/redoc")

    assert openapi_response.status_code == 200
    assert docs_response.status_code == 200
    assert redoc_response.status_code == 200
    schema = openapi_response.json()
    assert "/v1/chat/completions" in schema["paths"]
    assert "/v1/responses" in schema["paths"]
    assert "/v1/models" in schema["paths"]
    assert "/health" in schema["paths"]


def test_api_key_is_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_1", "key-1")
    monkeypatch.setattr(settings, "API_KEY_2", "")
    monkeypatch.setattr(settings, "API_KEY_3", "")

    response = client.get("/v1/models")

    assert response.status_code == 401
    payload = response.json()
    assert "detail" not in payload
    assert payload["error"]["message"] == "Missing API key"
    assert response.headers["www-authenticate"] == "Bearer"


def test_api_key_allows_access_when_bearer_matches(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_1", "key-1")
    monkeypatch.setattr(settings, "API_KEY_2", "key-2")
    monkeypatch.setattr(settings, "API_KEY_3", "key-3")

    response = client.get("/v1/models", headers={"Authorization": "Bearer key-2"})

    assert response.status_code == 200


def test_health_reports_api_key_auth_enabled_when_keys_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "API_KEY_1", "key-1")
    monkeypatch.setattr(settings, "API_KEY_2", "")
    monkeypatch.setattr(settings, "API_KEY_3", "")

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key_auth_enabled"] is True

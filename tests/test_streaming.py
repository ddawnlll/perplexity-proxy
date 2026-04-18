from __future__ import annotations

import json

import pytest

from app.streaming import _is_internal_chunk, chat_completions_stream, completions_stream, responses_stream


async def _collect_async(gen):
    items = []
    async for item in gen:
        items.append(item)
    return items


def _parse_data_events(events: list[str]) -> list[dict]:
    parsed = []
    for event in events:
        if event == "data: [DONE]\n\n":
            continue
        assert event.endswith("\n\n")
        data_line = event.removeprefix("data: ").removesuffix("\n\n")
        parsed.append(json.loads(data_line))
    return parsed


@pytest.mark.asyncio
async def test_chat_completions_stream_three_chunks_yields_deltas_stop_and_done():
    async def generator():
        yield "Hello"
        yield " world"
        yield "!"

    events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    parsed = _parse_data_events(events)

    assert len(parsed) == 4
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[0]["choices"][0]["delta"]["content"] == "Hello"
    assert parsed[1]["choices"][0]["delta"]["content"] == " world"
    assert parsed[2]["choices"][0]["delta"]["content"] == "!"
    assert parsed[3]["choices"][0]["delta"] == {}
    assert parsed[3]["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_completions_stream_zero_chunks_yields_stop_and_done_only():
    async def generator():
        if False:
            yield "never"

    events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    parsed = _parse_data_events(events)

    assert len(parsed) == 1
    assert parsed[0]["choices"][0]["delta"] == {}
    assert parsed[0]["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_completions_stream_skips_empty_chunks():
    async def generator():
        yield None
        yield ""
        yield "Hello"

    events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    parsed = _parse_data_events(events)

    assert len(parsed) == 2
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[0]["choices"][0]["delta"]["content"] == "Hello"
    assert parsed[1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_chat_completions_stream_first_event_has_assistant_role():
    async def generator():
        yield "Hello"

    events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    parsed = _parse_data_events(events)

    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_completions_stream_all_events_end_with_newlines():
    async def generator():
        yield "Hello"

    events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    assert all(event.endswith("\n\n") for event in events)


@pytest.mark.asyncio
async def test_chat_completions_stream_emits_synthetic_roo_tool_call():
    async def generator():
        yield "Hello"
        yield " world"

    events = await _collect_async(
        chat_completions_stream(generator(), "sonar", "chatcmpl-123", roo_mode=True)
    )
    parsed = _parse_data_events(events)

    assert parsed[0]["choices"][0]["delta"]["content"] == "Hello"
    assert parsed[1]["choices"][0]["delta"]["content"] == " world"
    final_choice = parsed[2]["choices"][0]
    assert final_choice["finish_reason"] == "tool_calls"
    assert final_choice["delta"]["tool_calls"][0]["function"]["name"] == "attempt_completion"
    assert json.loads(final_choice["delta"]["tool_calls"][0]["function"]["arguments"]) == {
        "result": "Hello world"
    }
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_completions_stream_transforms_chat_chunks():
    async def generator():
        yield "Hello"
        yield " world"

    events = await _collect_async(completions_stream(generator(), "sonar", "chatcmpl-123"))
    parsed = _parse_data_events(events)

    assert parsed[0]["object"] == "text_completion"
    assert parsed[0]["choices"][0]["text"] == "Hello"
    assert parsed[1]["choices"][0]["text"] == " world"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_chat_completions_stream_converts_cumulative_answer_snapshots_to_deltas():
    async def generator():
        yield {"answer": "H"}
        yield {"answer": "He"}
        yield {"answer": "Hel"}
        yield {"answer": "Hell"}
        yield {"answer": "Hello"}

    events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    parsed = _parse_data_events(events)

    assert [event["choices"][0]["delta"].get("content", "") for event in parsed[:-1]] == [
        "H",
        "e",
        "l",
        "l",
        "o",
    ]
    assert "".join(event["choices"][0]["delta"].get("content", "") for event in parsed[:-1]) == "Hello"
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_completions_stream_converts_cumulative_answer_snapshots_to_deltas():
    async def generator():
        yield {"answer": "Hel"}
        yield {"answer": "Hell"}
        yield {"answer": "Hello"}

    events = await _collect_async(completions_stream(generator(), "sonar", "cmpl-123"))
    parsed = _parse_data_events(events)

    assert [event["choices"][0]["text"] for event in parsed[:-1]] == ["Hel", "l", "o"]
    assert "".join(event["choices"][0]["text"] for event in parsed[:-1]) == "Hello"
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_responses_stream_three_chunks_yields_created_delta_done_completed_and_done():
    async def generator():
        yield "Hello"
        yield " world"
        yield "!"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    parsed = _parse_data_events(events)

    assert len(parsed) == 6
    assert parsed[0]["type"] == "response.created"
    assert parsed[1]["type"] == "response.output_text.delta"
    assert parsed[1]["delta"] == "Hello"
    assert parsed[2]["type"] == "response.output_text.delta"
    assert parsed[3]["type"] == "response.output_text.delta"
    assert parsed[4]["type"] == "response.output_text.done"
    assert parsed[4]["text"] == "Hello world!"
    assert parsed[5]["type"] == "response.completed"
    assert parsed[5]["response"]["output"][0]["content"][0]["text"] == "Hello world!"
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_responses_stream_converts_cumulative_answer_snapshots_to_deltas():
    async def generator():
        yield {"answer": "H"}
        yield {"answer": "He"}
        yield {"answer": "Hel"}
        yield {"answer": "Hell"}
        yield {"answer": "Hello"}

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    parsed = _parse_data_events(events)

    delta_events = [event for event in parsed if event.get("type") == "response.output_text.delta"]
    assert [event["delta"] for event in delta_events] == ["H", "e", "l", "l", "o"]
    assert "".join(event["delta"] for event in delta_events) == "Hello"
    assert parsed[-2]["text"] == "Hello"
    assert parsed[-1]["response"]["output"][0]["content"][0]["text"] == "Hello"


@pytest.mark.asyncio
async def test_streams_use_markdown_block_snapshots_as_cumulative_text():
    async def generator():
        yield {
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "markdown_block": {"chunks": ["Hel"]},
                }
            ]
        }
        yield {
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "markdown_block": {"chunks": ["Hell"]},
                }
            ]
        }
        yield {
            "blocks": [
                {
                    "intended_usage": "ask_text_0_markdown",
                    "markdown_block": {"chunks": ["Hello"]},
                }
            ]
        }

    chat_events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    chat_parsed = _parse_data_events(chat_events)
    assert [event["choices"][0]["delta"].get("content", "") for event in chat_parsed[:-1]] == ["Hel", "l", "o"]

    response_events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    response_parsed = _parse_data_events(response_events)
    response_deltas = [event["delta"] for event in response_parsed if event.get("type") == "response.output_text.delta"]
    assert response_deltas == ["Hel", "l", "o"]
    assert response_parsed[-2]["text"] == "Hello"


@pytest.mark.asyncio
async def test_responses_stream_skips_empty_chunks():
    async def generator():
        yield None
        yield ""
        yield "Hello"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    parsed = _parse_data_events(events)

    assert parsed[0]["type"] == "response.created"
    assert parsed[1]["type"] == "response.output_text.delta"
    assert parsed[1]["delta"] == "Hello"
    assert parsed[2]["type"] == "response.output_text.done"
    assert parsed[3]["type"] == "response.completed"


def test_is_internal_chunk_dict():
    assert _is_internal_chunk({"backend_uuid": "x", "classifier_results": {}}) is True
    assert _is_internal_chunk({"delta": "hello"}) is False
    assert _is_internal_chunk({}) is False


def test_is_internal_chunk_string():
    assert _is_internal_chunk("{'backend_uuid': 'abc', 'context_uuid': 'xyz'}") is True
    assert _is_internal_chunk('{"backend_uuid": "abc"}') is True
    assert _is_internal_chunk("Hello world") is False
    assert _is_internal_chunk("") is False


@pytest.mark.asyncio
async def test_state_blob_chunks_are_filtered_from_stream():
    async def chat_generator():
        yield {"backend_uuid": "abc123", "classifier_results": {}}
        yield {"content": "{'backend_uuid': 'abc123', 'status': 'PENDING'}"}
        yield "hello"

    chat_events = await _collect_async(chat_completions_stream(chat_generator(), "sonar", "chatcmpl-123"))
    chat_parsed = _parse_data_events(chat_events)
    assert chat_parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chat_parsed[0]["choices"][0]["delta"]["content"] == "hello"

    async def response_generator():
        yield {"backend_uuid": "abc123", "classifier_results": {}}
        yield {"content": "{'backend_uuid': 'abc123', 'status': 'PENDING'}"}
        yield "hello"

    response_events = await _collect_async(responses_stream(response_generator(), "sonar", "resp-123"))
    response_parsed = _parse_data_events(response_events)
    assert response_parsed[0]["type"] == "response.created"
    assert response_parsed[1]["type"] == "response.output_text.delta"
    assert response_parsed[1]["delta"] == "hello"

    for event in chat_events + response_events:
        assert "backend_uuid" not in event
        assert "classifier_results" not in event
        assert "mhe_predictions" not in event


@pytest.mark.asyncio
async def test_responses_stream_delta_event_type():
    async def generator():
        yield "Hello"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    parsed = _parse_data_events(events)

    assert parsed[1]["type"] == "response.output_text.delta"


@pytest.mark.asyncio
async def test_responses_stream_all_events_end_with_newlines():
    async def generator():
        yield "Hello"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    assert all(event.endswith("\n\n") for event in events)


def test_snapshot_to_delta_handles_append_repeat_and_reset():
    from app.streaming import _snapshot_to_delta

    assert _snapshot_to_delta("Hello", "") == ("Hello", "Hello")
    assert _snapshot_to_delta("Hello", "Hello") == (None, "Hello")
    assert _snapshot_to_delta("Hello!", "Hello") == ("!", "Hello!")
    assert _snapshot_to_delta("Reset", "Hello!") == ("Reset", "Hello!Reset")

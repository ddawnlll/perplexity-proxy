from __future__ import annotations

import json

import pytest

from app.streaming import chat_completions_stream, responses_stream


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
async def test_chat_completions_stream_keeps_empty_chunks():
    async def generator():
        yield None
        yield ""
        yield "Hello"

    events = await _collect_async(chat_completions_stream(generator(), "sonar", "chatcmpl-123"))
    parsed = _parse_data_events(events)

    assert len(parsed) == 3
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert parsed[0]["choices"][0]["delta"]["content"] == ""
    assert parsed[1]["choices"][0]["delta"]["content"] == "Hello"
    assert parsed[2]["choices"][0]["finish_reason"] == "stop"


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
async def test_responses_stream_three_chunks_yields_deltas_done_completed_and_done():
    async def generator():
        yield "Hello"
        yield " world"
        yield "!"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    parsed = _parse_data_events(events)

    assert len(parsed) == 5
    assert parsed[0]["type"] == "response.output_text.delta"
    assert parsed[0]["delta"] == "Hello"
    assert parsed[1]["type"] == "response.output_text.delta"
    assert parsed[2]["type"] == "response.output_text.delta"
    assert parsed[3]["type"] == "response.output_text.done"
    assert parsed[3]["text"] == "Hello world!"
    assert parsed[4]["type"] == "response.completed"
    assert parsed[4]["response"]["output"][0]["content"][0]["text"] == "Hello world!"
    assert events[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_responses_stream_keeps_empty_chunks():
    async def generator():
        yield None
        yield ""
        yield "Hello"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    parsed = _parse_data_events(events)

    assert parsed[0]["type"] == "response.output_text.delta"
    assert parsed[0]["delta"] == ""
    assert parsed[1]["type"] == "response.output_text.delta"
    assert parsed[1]["delta"] == "Hello"
    assert parsed[2]["type"] == "response.output_text.done"
    assert parsed[3]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_state_blob_chunks_are_filtered_from_stream():
    async def chat_generator():
        yield {"content": "{'backend_uuid': 'abc123', 'status': 'PENDING'}"}
        yield "hello"

    chat_events = await _collect_async(chat_completions_stream(chat_generator(), "sonar", "chatcmpl-123"))
    chat_parsed = _parse_data_events(chat_events)
    assert chat_parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chat_parsed[0]["choices"][0]["delta"]["content"] == "hello"

    async def response_generator():
        yield {"content": "{'backend_uuid': 'abc123', 'status': 'PENDING'}"}
        yield "hello"

    response_events = await _collect_async(responses_stream(response_generator(), "sonar", "resp-123"))
    response_parsed = _parse_data_events(response_events)
    assert response_parsed[0]["type"] == "response.output_text.delta"
    assert response_parsed[0]["delta"] == "hello"


@pytest.mark.asyncio
async def test_responses_stream_delta_event_type():
    async def generator():
        yield "Hello"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    parsed = _parse_data_events(events)

    assert parsed[0]["type"] == "response.output_text.delta"


@pytest.mark.asyncio
async def test_responses_stream_all_events_end_with_newlines():
    async def generator():
        yield "Hello"

    events = await _collect_async(responses_stream(generator(), "sonar", "resp-123"))
    assert all(event.endswith("\n\n") for event in events)

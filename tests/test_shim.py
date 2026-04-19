from __future__ import annotations

import json

from app.tools.shim import build_perplexity_instruction, decide_tool, wrap_as_tool_response


def test_decide_tool_returns_write_to_file_after_read_file():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\":\"calculator.py\"}"}}],
        },
        {"role": "tool", "content": "def add(a,b): return a+b"},
    ]

    decision = decide_tool(messages, "add a divide function")

    assert decision["tool"] == "write_to_file"
    assert decision["args_hint"]["path"] == "calculator.py"


def test_decide_tool_returns_read_file_for_first_file_edit_request():
    decision = decide_tool([], "can you edit calculator.py")

    assert decision["tool"] == "read_file"
    assert "calculator.py" in decision["args_hint"]["path"]


def test_decide_tool_falls_back_to_attempt_completion():
    decision = decide_tool([], "hello")

    assert decision["tool"] == "attempt_completion"


def test_decide_tool_uses_injected_read_file_block_for_write():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<user_message>\nadd divide to calculator.py\n</user_message>"},
                {
                    "type": "text",
                    "text": "[read_file for 'calculator.py']\nFile: calculator.py\n 1 | def add(a, b):\n 2 |     return a + b",
                },
            ],
        }
    ]

    decision = decide_tool(messages, "add divide to calculator.py")

    assert decision["tool"] == "write_to_file"
    assert decision["args_hint"]["path"] == "calculator.py"
    assert decision["content_source"] == "injected_read"


def test_decide_tool_returns_attempt_completion_after_write_to_file():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_to_file",
                        "arguments": "{\"path\":\"calculator.py\",\"content\":\"def add(a, b):\\n    return a + b\\n\",\"line_count\":2}",
                    }
                }
            ],
        },
        {"role": "tool", "content": "File updated successfully"},
        {"role": "user", "content": [{"type": "text", "text": "<environment_details>cwd=/tmp/project</environment_details>"}]},
    ]

    decision = decide_tool(messages, "")

    assert decision["tool"] == "attempt_completion"
    assert decision["static_result"] == "The file `calculator.py` has been updated successfully."
    assert decision["content_source"] == "post_write"


def test_build_perplexity_instruction_includes_injected_file_block():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<user_message>\nadd divide to calculator.py\n</user_message>"},
                {
                    "type": "text",
                    "text": "[read_file for 'calculator.py']\nFile: calculator.py\n 1 | def add(a, b):\n 2 |     return a + b",
                },
            ],
        }
    ]

    prompt = build_perplexity_instruction(
        {
            "tool": "write_to_file",
            "perplexity_instruction": "Return ONLY the complete updated content of calculator.py.",
        },
        "add divide to calculator.py",
        messages=messages,
    )

    assert prompt is not None
    assert "Current file content:" in prompt
    assert "[read_file for 'calculator.py']" in prompt
    assert "User request: add divide to calculator.py" in prompt


def test_wrap_as_tool_response_uses_decision_directly():
    payload = wrap_as_tool_response(
        "def add(a, b):\n    return a + b\n",
        "gpt-5.2",
        "chatcmpl-123",
        {"tool": "write_to_file", "args_hint": {"path": "calculator.py"}},
    )
    args = json.loads(payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

    assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "write_to_file"
    assert args["path"] == "calculator.py"
    assert "return a + b" in args["content"]
    assert args["line_count"] == 2

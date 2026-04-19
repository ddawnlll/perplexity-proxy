from __future__ import annotations

import json

from app.tools.shim import (
    Phase,
    build_perplexity_instruction,
    decide_tool,
    detect_phase,
    wrap_as_tool_response,
)


def test_detect_phase_starts_in_planning_without_prior_tool_calls():
    phase, context = detect_phase([], "Create calculator/core.py and calculator/history.py.")

    assert phase == Phase.PLANNING
    assert "calculator/core.py" in context["query"]


def test_detect_phase_generates_after_read_file():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\":\"calculator.py\"}"}}],
        },
        {"role": "tool", "content": "def add(a, b):\n    return a + b\n"},
    ]

    phase, context = detect_phase(messages, "add divide")

    assert phase == Phase.GENERATING
    assert context["file"] == "calculator.py"
    assert "def add" in context["current_content"]


def test_detect_phase_reads_next_pending_file_after_write():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_to_file",
                        "arguments": "{\"path\":\"calculator/core.py\",\"content\":\"pass\\n\",\"line_count\":1}",
                    }
                }
            ],
        }
    ]

    phase, context = detect_phase(
        messages,
        "Create calculator/core.py and calculator/history.py, then run `python -m pytest calculator/tests/ -v`.",
    )

    assert phase == Phase.READING
    assert context["path"] == "calculator/history.py"


def test_detect_phase_moves_to_testing_after_last_write_when_command_exists():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_to_file",
                        "arguments": "{\"path\":\"calculator/core.py\",\"content\":\"pass\\n\",\"line_count\":1}",
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_to_file",
                        "arguments": "{\"path\":\"calculator/history.py\",\"content\":\"pass\\n\",\"line_count\":1}",
                    }
                }
            ],
        },
    ]

    phase, context = detect_phase(
        messages,
        "Create calculator/core.py and calculator/history.py, then run `python -m pytest calculator/tests/ -v`.",
    )

    assert phase == Phase.TESTING
    assert context["command"] == "python -m pytest calculator/tests/ -v"


def test_detect_phase_reads_failing_file_after_test_error():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_command",
                        "arguments": "{\"command\":\"python -m pytest calculator/tests/ -v\"}",
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                "ERROR collecting test_formatter.py\n"
                "E   ImportError: cannot import name 'format_error' from 'calculator' "
                "(/Users/hootie/src/bugbot/calculator/__init__.py)\n"
            ),
        },
    ]

    phase, context = detect_phase(messages, "Fix the problem please")

    assert phase == Phase.READING
    assert context["path"] == "calculator/__init__.py"


def test_detect_phase_fixes_after_failed_command_with_injected_content():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_command",
                        "arguments": "{\"command\":\"python -m pytest calculator/tests/ -v\"}",
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "ImportError: cannot import name 'format_error' from 'calculator'",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "[read_file for 'calculator/__init__.py']\nFile: calculator/__init__.py\n 1 | from .formatter import format_result",
                }
            ],
        },
    ]

    phase, context = detect_phase(messages, "Fix the problem please")

    assert phase == Phase.FIXING
    assert context["file"] == "calculator/__init__.py"


def test_build_perplexity_instruction_uses_phase_prompt_for_generating():
    decision = {
        "tool": "write_to_file",
        "phase": Phase.GENERATING.value,
        "phase_context": {
            "file": "calculator.py",
            "current_content": "def add(a, b):\n    return a + b\n",
            "query": "add divide function",
        },
    }

    prompt = build_perplexity_instruction(decision, "add divide function")

    assert prompt is not None
    assert "Return ONLY the complete updated content of calculator.py." in prompt
    assert "Current content:" in prompt
    assert "Task: add divide function" in prompt


def test_decide_tool_returns_phase_based_tool_shapes():
    planning = decide_tool([], "Create calculator/core.py")
    assert planning["tool"] == "attempt_completion"
    assert planning["phase"] == Phase.PLANNING.value

    reading = decide_tool(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_to_file",
                            "arguments": "{\"path\":\"calculator/core.py\",\"content\":\"pass\\n\",\"line_count\":1}",
                        }
                    }
                ],
            }
        ],
        "Create calculator/core.py and calculator/history.py",
    )
    assert reading["tool"] == "read_file"
    assert reading["args_hint"]["path"] == "calculator/history.py"


def test_wrap_as_tool_response_handles_phase_tools():
    payload = wrap_as_tool_response(
        "def add(a, b):\n    return a + b\n",
        "gpt-5.2",
        "chatcmpl-123",
        {"tool": "write_to_file", "args_hint": {"path": "calculator.py"}},
    )
    args = json.loads(payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "calculator.py"
    assert args["line_count"] == 2

    command_payload = wrap_as_tool_response(
        None,
        "gpt-5.2",
        "chatcmpl-456",
        {"tool": "execute_command", "args_hint": {"command": "python -m pytest calculator/tests/ -v"}},
    )
    command_args = json.loads(command_payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert command_args["command"] == "python -m pytest calculator/tests/ -v"

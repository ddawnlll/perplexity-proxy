from __future__ import annotations

import json

from app.tools.shim import (
    Agent,
    CodingPhase,
    Phase,
    build_conversational_prompt,
    build_full_context_prompt,
    build_hermes_prompt,
    build_perplexity_instruction,
    build_conversational_tool_decision,
    coding_shim,
    decide_tool,
    detect_agent,
    detect_coding_phase,
    detect_phase,
    extract_task_history,
    normalize_tool_request_text,
    parse_tool_request,
    wrap_as_tool_response,
    wrap_for_hermes,
)


def test_detect_agent_classifies_roo_hermes_and_generic_requests():
    assert detect_agent(None) == Agent.GENERIC
    assert detect_agent([]) == Agent.GENERIC
    assert detect_agent([{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}]) == Agent.ROO
    assert detect_agent([{"type": "function", "function": {"name": "finish", "parameters": {}}}]) == Agent.HERMES
    assert detect_agent([{"type": "function", "function": {"name": "unknown_tool", "parameters": {}}}]) == Agent.GENERIC


def test_detect_agent_prefers_hermes_overlapping_tools_and_uses_user_agent():
    overlapping_tools = [
        {"type": "function", "function": {"name": "execute_command", "parameters": {}}},
        {"type": "function", "function": {"name": "finish", "parameters": {}}},
    ]

    assert detect_agent(overlapping_tools) == Agent.HERMES
    assert detect_agent(overlapping_tools, user_agent="OpenAI/Python 2.32.0") == Agent.HERMES
    assert detect_agent(overlapping_tools, user_agent="RooCode/3.52.1") == Agent.ROO


def test_detect_agent_recognizes_live_hermes_browser_tools():
    browser_tools = [
        {"type": "function", "function": {"name": "browser_back", "parameters": {}}},
        {"type": "function", "function": {"name": "browser_click", "parameters": {}}},
        {"type": "function", "function": {"name": "browser_console", "parameters": {}}},
        {"type": "function", "function": {"name": "browser_get_images", "parameters": {}}},
        {"type": "function", "function": {"name": "browser_navigate", "parameters": {}}},
    ]

    assert detect_agent(browser_tools) == Agent.HERMES
    assert detect_agent(browser_tools, user_agent="OpenAI/Python 2.32.0") == Agent.HERMES


def test_detect_agent_uses_openai_python_as_last_hermes_fallback_when_tools_are_unknown():
    unknown_tools = [{"type": "function", "function": {"name": "totally_unknown_tool", "parameters": {}}}]

    assert detect_agent(unknown_tools) == Agent.GENERIC
    assert detect_agent(unknown_tools, user_agent="OpenAI/Python 2.32.0") == Agent.HERMES


def test_detect_agent_recognizes_live_hermes_core_tools():
    hermes_tools = [
        {"type": "function", "function": {"name": "terminal", "parameters": {}}},
        {"type": "function", "function": {"name": "write_file", "parameters": {}}},
        {"type": "function", "function": {"name": "patch", "parameters": {}}},
        {"type": "function", "function": {"name": "execute_code", "parameters": {}}},
        {"type": "function", "function": {"name": "clarify", "parameters": {}}},
    ]

    assert detect_agent(hermes_tools) == Agent.HERMES


def test_detect_agent_recognizes_pi_coding_tools():
    pi_tools = [
        {"type": "function", "function": {"name": "read", "parameters": {}}},
        {"type": "function", "function": {"name": "edit", "parameters": {}}},
        {"type": "function", "function": {"name": "bash", "parameters": {}}},
        {"type": "function", "function": {"name": "ls", "parameters": {}}},
    ]

    assert detect_agent(pi_tools) == Agent.ROO


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

    assert phase == Phase.GENERATING
    assert context["action"] == "read_file"
    assert context["path"] == "calculator/history.py"


def test_detect_phase_moves_to_testing_after_last_write_when_command_exists():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "attempt_completion",
                        "arguments": json.dumps(
                            {
                                "result": (
                                    "1. `pytest test_core.py -v`\n"
                                    "2. `pytest test_formatter.py -v`\n"
                                    "3. `pytest -v`"
                                )
                            }
                        ),
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
    assert context["command"] == "pytest test_core.py -v"


def test_detect_phase_skips_testing_when_execute_command_is_not_declared():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "attempt_completion",
                        "arguments": json.dumps({"result": "1. `python3 -m pytest -v`"}),
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
                        "arguments": "{\"path\":\"calculator/core.py\",\"content\":\"pass\\n\",\"line_count\":1}",
                    }
                }
            ],
        },
    ]

    phase, context = detect_phase(
        messages,
        "Create calculator/core.py, then run tests.",
        tools=[{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
    )

    assert phase == Phase.COMPLETE
    assert context["written"] == "calculator/core.py"


def test_detect_phase_runs_direct_command_requests():
    phase, context = detect_phase(
        [],
        "can you run command pwd",
        tools=[{"type": "function", "function": {"name": "execute_command", "parameters": {}}}],
    )

    assert phase == Phase.TESTING
    assert context["command"] == "pwd"


def test_detect_phase_runs_short_direct_commands():
    phase, context = detect_phase(
        [],
        "can you pwd",
        tools=[{"type": "function", "function": {"name": "execute_command", "parameters": {}}}],
    )

    assert phase == Phase.TESTING
    assert context["command"] == "pwd"


def test_detect_phase_advances_to_next_planned_command_after_success():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "attempt_completion",
                        "arguments": json.dumps(
                            {
                                "result": (
                                    "1. `pytest test_core.py -v`\n"
                                    "2. `pytest test_formatter.py -v`\n"
                                    "3. `pytest -v`"
                                )
                            }
                        ),
                    }
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_command",
                        "arguments": "{\"command\":\"pytest test_core.py -v\"}",
                    }
                }
            ],
        },
        {"role": "tool", "content": "================ 7 passed ================\n"},
    ]

    phase, context = detect_phase(messages, "test it please")

    assert phase == Phase.TESTING
    assert context["command"] == "pytest test_formatter.py -v"


def test_detect_phase_prioritizes_explicit_testing_request_over_stale_file_context():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<user_message>Create calculator/core.py and calculator/history.py.</user_message>",
                }
            ],
        },
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
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": '{"path":"calculator/core.py","operation":"modified","notice":"You do not need to re-read the file"}',
                },
                {
                    "type": "text",
                    "text": "Command executed in terminal within working directory '/Users/hootie/src/bugbot'. Exit code: 0",
                },
                {
                    "type": "text",
                    "text": "<user_message>\ntest it please\n</user_message>",
                },
            ],
        },
    ]

    phase, context = detect_phase(messages, "test it please")

    assert phase == Phase.TESTING
    assert context["command"] == "python3 -m pytest -v"


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

    assert phase == Phase.FIXING
    assert context["action"] == "read_file"
    assert context["path"] == "calculator/__init__.py"


def test_detect_phase_returns_failure_until_user_asks_to_fix():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "execute_command",
                        "arguments": "{\"command\":\"pytest test_formatter.py -v\"}",
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "ERROR collecting test_formatter.py\nImportError: cannot import name 'format_error' from 'calculator'\n",
        },
    ]

    phase, context = detect_phase(messages, "test it please")

    assert phase == Phase.COMPLETE
    assert context["failed"] is True


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


def test_build_perplexity_instruction_includes_system_context_for_planning():
    decision = {
        "tool": "attempt_completion",
        "phase": Phase.PLANNING.value,
        "phase_context": {
            "query": "test it please",
        },
    }

    prompt = build_perplexity_instruction(
        decision,
        "test it please",
        messages=[
            {
                "role": "system",
                "content": (
                    "SYSTEM INFORMATION\nProject uses Python 3.11 and pytest.\n====\n"
                    "Current Workspace\n/Users/hootie/src/bugbot\n====\n"
                    "RULES\nAll source files are Python.\n"
                ),
            }
        ],
    )

    assert prompt is not None
    assert "Use the project context below to infer the language" in prompt
    assert "Project uses Python 3.11 and pytest." in prompt
    assert "All source files are Python." in prompt
    assert "Task: test it please" in prompt
    assert "wrap each full command in backticks" in prompt


def test_build_perplexity_instruction_preserves_task_history():
    decision = {
        "tool": "write_to_file",
        "phase": Phase.GENERATING.value,
        "phase_context": {
            "file": "calculator.py",
            "query": "add divide function",
            "current_content": "def add(a, b):\n    return a + b\n",
        },
    }
    prompt = build_perplexity_instruction(
        decision,
        "add divide function",
        messages=[
            {"role": "user", "content": "Create calculator.py"},
            {"role": "assistant", "content": "Sure"},
            {"role": "user", "content": [{"type": "text", "text": "<user_message>add divide function</user_message>"}]},
        ],
    )

    assert prompt is not None
    assert "Conversation history:" in prompt
    assert "Create calculator.py" in prompt
    assert "add divide function" in prompt


def test_build_hermes_prompt_requests_plain_terminal_text():
    prompt = build_hermes_prompt(
        "hello",
        system_message="# heading\nYou are Hermes.\nKeep terminal output concise.\nNo markdown.\n",
    )

    assert "Reply in plain text only." in prompt
    assert "Write as if outputting to a terminal." in prompt
    assert "Persona context: You are Hermes. Keep terminal output concise. No markdown." in prompt
    assert prompt.endswith("User message: hello")


def test_build_conversational_prompt_includes_full_history_and_tool_formats():
    prompt = build_conversational_prompt(
        "make a calculator app",
        messages=[
            {"role": "user", "content": "make a calculator app"},
            {"role": "assistant", "content": "I should inspect the repo"},
            {"role": "user", "content": "please continue"},
        ],
        system_message="Project uses Python",
    )

    assert "conversational coding partner" in prompt
    assert "Please read [filename]" in prompt
    assert "Use the Pi tools directly when possible: read, edit, ls, and bash." in prompt
    assert "Current conversation:" in prompt
    assert "make a calculator app" in prompt
    assert "please continue" in prompt
    assert "Latest user request:" in prompt


def test_parse_tool_request_recognizes_read_write_execute_and_complete():
    assert parse_tool_request("Please read main.py") == ("read_file", "main.py")
    assert parse_tool_request("Please execute pytest -q") == ("execute_command", "pytest -q")
    assert parse_tool_request("Here's the updated calculator.py:\ndef add(a, b):\n    return a + b") == ("write_to_file", "calculator.py")
    assert parse_tool_request("Looks good! All done.") == ("attempt_completion", None)


def test_parse_tool_request_trims_follow_up_instructions_from_targets():
    assert parse_tool_request("I should run ls -la. Please execute ls -la") == ("execute_command", "ls -la")
    assert parse_tool_request("can you run command pwd") == ("execute_command", "pwd")
    assert parse_tool_request("can you pwd") == ("execute_command", "pwd")
    assert parse_tool_request("I need to list files in /Users/hootie/src/test. Please list files in /Users/hootie/src/test") == (
        "list_files",
        "/Users/hootie/src/test",
    )
    assert parse_tool_request("Please read main.py. Then tell me what you see.") == ("read_file", "main.py")


def test_normalize_tool_request_text_strips_write_header_and_fences():
    text = "Here\'s the updated calculator.py:\n```python\ndef add(a, b):\n    return a + b\n```"
    normalized = normalize_tool_request_text("write_to_file", text, "calculator.py")
    assert normalized == "def add(a, b):\n    return a + b"


def test_build_conversational_tool_decision_maps_arguments():
    decision = build_conversational_tool_decision("execute_command", "pytest -q")
    assert decision["tool"] == "execute_command"
    assert decision["args_hint"]["command"] == "pytest -q"


def test_extract_task_history_preserves_read_blocks_and_recent_turns():
    history = extract_task_history(
        [
            {"role": "user", "content": "first request"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<user_message>second request</user_message>"},
                    {"type": "text", "text": "[read_file for 'calculator.py']\nFile: calculator.py\n 1 | def add(a, b):\n"},
                ],
            },
        ]
    )

    assert "first request" in history
    assert "second request" in history
    assert "[read_file for 'calculator.py']" in history


def test_detect_coding_phase_tracks_file_workflow():
    assert detect_coding_phase([], "Create calculator.py") == CodingPhase.PLANNING
    assert detect_coding_phase(
        [{"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\":\"calculator.py\"}"}}]}],
        "edit it",
    ) == CodingPhase.FILE_EDIT


def test_build_full_context_prompt_includes_task_history_and_test_output():
    prompt = build_full_context_prompt(
        CodingPhase.FIXING,
        {
            "task_history": "Create calculator.py\n\nEdit calculator.py",
            "read_files": ["calculator.py"],
            "written_files": ["calculator.py"],
            "test_output": "ImportError: missing helper",
            "current_file": {"path": "calculator.py", "content": "def add(a, b): pass"},
        },
        system_message="Project uses Python",
    )

    assert "FULL PROJECT CONTEXT" in prompt
    assert "Task history:" in prompt
    assert "Read files: calculator.py" in prompt
    assert "Written files: calculator.py" in prompt
    assert "Latest test failure:" in prompt
    assert "Project uses Python" in prompt


def test_coding_shim_returns_roo_state_with_full_context_prompt():
    state = coding_shim(
        [
            {"role": "user", "content": "Create calculator.py"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": "{\"path\":\"calculator.py\"}"}}]},
            {"role": "tool", "content": "def add(a, b):\n    return a + b\n"},
        ],
        "Create calculator.py",
        tools=[{"type": "function", "function": {"name": "attempt_completion", "parameters": {}}}],
        user_agent="RooCode/3.52.1",
        system_message="Project uses Python",
    )

    assert state["agent"] == Agent.ROO
    assert state["response_wrapper"] == "roo_tool"
    assert "FULL PROJECT CONTEXT" in state["full_context_prompt"]
    assert "Return ONLY the complete updated content of calculator.py." in state["perplexity_prompt"]


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
    assert reading["phase"] == Phase.GENERATING.value


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


def test_wrap_as_tool_response_maps_pi_style_tools():
    edit_payload = wrap_as_tool_response(
        "def add(a, b):\n    return a + b\n\n",
        "gpt-5.2",
        "chatcmpl-pi-edit",
        {
            "tool": "write_to_file",
            "tool_style": "pi",
            "args_hint": {"path": "calculator.py"},
            "phase_context": {"current_content": "def add(a, b):\n    return a + b\n"},
        },
    )
    edit_call = edit_payload["choices"][0]["message"]["tool_calls"][0]["function"]
    assert edit_call["name"] == "edit"
    edit_args = json.loads(edit_call["arguments"])
    assert edit_args["path"] == "calculator.py"
    assert edit_args["edits"][0]["oldText"] == "def add(a, b):\n    return a + b\n"
    assert edit_args["edits"][0]["newText"] == "def add(a, b):\n    return a + b"

    complete_payload = wrap_as_tool_response(
        "All done.",
        "gpt-5.2",
        "chatcmpl-pi-complete",
        {"tool": "attempt_completion", "tool_style": "pi", "args_hint": {}},
    )
    complete_choice = complete_payload["choices"][0]
    assert complete_choice["finish_reason"] == "stop"
    assert complete_choice["message"]["content"] == "All done."
    assert "tool_calls" not in complete_choice["message"]


def test_wrap_as_tool_response_falls_back_for_empty_write_and_empty_completion():
    empty_write = wrap_as_tool_response(
        "",
        "gpt-5.2",
        "chatcmpl-empty-write",
        {"tool": "write_to_file", "args_hint": {"path": "calculator.py"}},
    )
    empty_write_call = empty_write["choices"][0]["message"]["tool_calls"][0]["function"]
    assert empty_write_call["name"] == "attempt_completion"
    assert json.loads(empty_write_call["arguments"]) == {"result": "Cannot write empty content."}

    empty_completion = wrap_as_tool_response(
        None,
        "gpt-5.2",
        "chatcmpl-empty-complete",
        {"tool": "attempt_completion", "args_hint": {}},
    )
    empty_completion_call = empty_completion["choices"][0]["message"]["tool_calls"][0]["function"]
    assert empty_completion_call["name"] == "attempt_completion"
    assert json.loads(empty_completion_call["arguments"]) == {"result": "Ready."}


def test_wrap_for_hermes_returns_terminal_tool_call_with_content():
    payload = wrap_for_hermes(None, "gpt-5.2", "chatcmpl-hermes")

    choice = payload["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Ready."
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "terminal"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "echo", "output": "Ready."}

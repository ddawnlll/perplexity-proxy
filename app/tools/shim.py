from __future__ import annotations

import json
import re
import time
import uuid
from enum import Enum
from typing import Any

from app.tools.prompt_builder import _context_snippets


class Agent(str, Enum):
    ROO = "roo"
    HERMES = "hermes"
    GENERIC = "generic"

ROO_TOOL_NAMES = frozenset(
    {
        "attempt_completion",
        "ask_followup_question",
        "read_file",
        "write_to_file",
        "replace_in_file",
        "list_files",
        "search_files",
        "execute_command",
        "browser_action",
        "use_mcp_tool",
    }
)

HERMES_TOOL_NAMES = frozenset(
    {
        "terminal",
        "write_file",
        "patch",
        "execute_code",
        "clarify",
        "memory",
        "skill_view",
        "skill_manage",
        "skill_list",
        "skills_list",
        "session_search",
        "todo",
        "cronjob",
        "delegate_task",
        "process",
        "text_to_speech",
        "vision_analyze",
        "read_file",
        "search_files",
        "finish",
        "respond",
        "final_response",
        "bash",
        "think",
        "str_replace_editor",
        "computer",
        "code_interpreter",
        "browser_back",
        "browser_click",
        "browser_console",
        "browser_get_images",
        "browser_navigate",
        "browser_press",
        "browser_scroll",
        "browser_snapshot",
        "browser_type",
        "browser_vision",
    }
)

FAILURE_MARKERS = (
    "error collecting",
    "traceback",
    "importerror",
    "assertionerror",
    "failed",
    "error:",
    "exception",
    "no module named",
)


class Phase(str, Enum):
    PLANNING = "planning"
    GENERATING = "generating"
    TESTING = "testing"
    FIXING = "fixing"
    COMPLETE = "complete"


class CodingPhase(str, Enum):
    PLANNING = "planning"
    FILE_READ = "file_read"
    FILE_EDIT = "file_edit"
    TESTING = "testing"
    FIXING = "fixing"
    COMPLETE = "complete"


def detect_agent(tools: list | None, user_agent: str = "") -> Agent:
    if "RooCode/" in user_agent:
        return Agent.ROO

    names = _tool_name_set(tools)
    if not names:
        return Agent.GENERIC

    if names & HERMES_TOOL_NAMES:
        return Agent.HERMES
    if any(name.startswith("browser_") and name != "browser_action" for name in names):
        return Agent.HERMES
    if names & ROO_TOOL_NAMES:
        return Agent.ROO
    if "OpenAI/Python" in user_agent and tools:
        return Agent.HERMES
    return Agent.GENERIC


def is_roo_request(tools: list | None) -> bool:
    return detect_agent(tools) == Agent.ROO


def _last_tool_call(messages: list) -> tuple[str, dict] | None:
    """Return (tool_name, arguments) for the most recent assistant tool call."""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            fn = tool_call.get("function", {})
            if not isinstance(fn, dict):
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                return name, args if isinstance(args, dict) else {}
    return None


def _mentions_file(text: str) -> str | None:
    match = re.search(r"""['"`]?([a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10})['"`]?""", text)
    return match.group(1) if match else None


def _mentioned_files(text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"""['"`]?([a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10})['"`]?""", text):
        path = match.group(1)
        if path not in paths:
            paths.append(path)
    return paths


def _has_injected_file_content(messages: list) -> tuple[bool, str]:
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "")
            if not isinstance(text, str):
                continue
            match = re.search(r"\[read_file for ['\"]([^'\"]+)['\"]\]", text)
            if match:
                return True, match.group(1)
        break
    return False, ""


def _injected_file_block(messages: list | None) -> str | None:
    if not messages:
        return None
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "")
            if isinstance(text, str) and re.search(r"\[read_file for ['\"][^'\"]+['\"]\]", text):
                return text
        break
    return None


def _task_text(messages: list, user_query: str) -> str:
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text", "")
            if not isinstance(text, str):
                continue
            if "<user_message>" in text:
                match = re.search(r"<user_message>\s*(.*?)\s*</user_message>", text, re.DOTALL)
                cleaned = match.group(1).strip() if match else ""
            else:
                cleaned = text.strip()
                noise_prefixes = (
                    "[read_file for ",
                    "Command executed in terminal",
                    '{"path":"',
                    '{"path": "',
                    "File: ",
                    "Task was interrupted",
                )
                if cleaned.startswith(noise_prefixes):
                    cleaned = ""
            if not cleaned:
                continue
            if cleaned not in parts:
                parts.append(cleaned)
    if user_query and user_query not in parts:
        parts.append(user_query)
    return "\n".join(parts)


def _is_testing_request(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    if any(
        phrase in lowered
        for phrase in (
            "create ",
            "edit ",
            "modify ",
            "add ",
            "fix ",
            "implement ",
            "write ",
        )
    ):
        return False
    return any(
        phrase in lowered
        for phrase in (
            "test it",
            "run tests",
            "run pytest",
            "pytest",
            "test the code",
            "test again",
            "run the tests",
            "please run pytest",
        )
    )


def _is_fix_request(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "fix",
            "debug",
            "repair",
            "search bugs",
            "search bug",
            "diagnosis",
            "confirm the diagnosis",
        )
    )


def _written_files(messages: list) -> list[str]:
    paths: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            fn = tool_call.get("function", {})
            if not isinstance(fn, dict) or fn.get("name") != "write_to_file":
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                continue
            path = args.get("path")
            if isinstance(path, str) and path and path not in paths:
                paths.append(path)
    return paths


def _read_files(messages: list) -> list[str]:
    paths: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                fn = tool_call.get("function", {})
                if not isinstance(fn, dict) or fn.get("name") != "read_file":
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    continue
                path = args.get("path")
                if isinstance(path, str) and path and path not in paths:
                    paths.append(path)
        if msg.get("role") == "user":
            content = msg.get("content", "")
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                if not isinstance(text, str):
                    continue
                match = re.search(r"\[read_file for ['\"]([^'\"]+)['\"]\]", text)
                if match and match.group(1) not in paths:
                    paths.append(match.group(1))
    return paths


def _requested_commands(text: str) -> list[str]:
    commands: list[str] = []
    for match in re.finditer(r"`([^`]+)`", text):
        command = match.group(1).strip()
        if command and command not in commands:
            commands.append(command)
    return commands


def _assistant_completion_results(messages: list) -> list[str]:
    results: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            fn = tool_call.get("function", {})
            if not isinstance(fn, dict) or fn.get("name") != "attempt_completion":
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            result = args.get("result")
            if isinstance(result, str) and result.strip():
                results.append(result)
    return results


def _executed_commands(messages: list) -> list[str]:
    commands: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            fn = tool_call.get("function", {})
            if not isinstance(fn, dict) or fn.get("name") != "execute_command":
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            command = args.get("command")
            if isinstance(command, str) and command.strip() and command.strip() not in commands:
                commands.append(command.strip())
    return commands


def _commands_from_plan_text(text: str) -> list[str]:
    commands = _requested_commands(text)
    if commands:
        return commands
    inferred: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(?:[-*]|\d+[.)])\s*(.+)$", stripped)
        candidate = match.group(1).strip() if match else stripped
        if not candidate:
            continue
        if re.match(r"^(pytest|python3?\s+-m\s+pytest|python3?\s+-c|cargo test|npm test|npx vitest run)\b", candidate):
            if candidate not in inferred:
                inferred.append(candidate)
    return inferred


def _planned_commands(messages: list, task_text: str) -> list[str]:
    commands: list[str] = []
    for result in _assistant_completion_results(messages):
        for command in _commands_from_plan_text(result):
            if command not in commands:
                commands.append(command)
    for command in _requested_commands(task_text):
        if command not in commands:
            commands.append(command)
    return commands


def _default_test_command(task_text: str, messages: list) -> str | None:
    requested = _planned_commands(messages, task_text)
    if requested:
        return requested[0]
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            fn = tool_call.get("function", {})
            if not isinstance(fn, dict) or fn.get("name") != "execute_command":
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            command = args.get("command")
            if isinstance(command, str) and command.strip():
                return command.strip()
    if ".py" in task_text or "python" in task_text or "pytest" in task_text:
        return "python3 -m pytest -v"
    return None


def _last_tool_result_text(messages: list) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for block in reversed(content):
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    result = block.get("content", "")
                    if isinstance(result, str) and result.strip():
                        return result
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        match = re.search(r"\[execute_command.*?\]\n?(.*)", text, re.DOTALL)
                        if match:
                            return match.group(1).strip()
    return ""


def _has_test_failure(output: str) -> bool:
    lowered = output.lower()
    if not lowered.strip():
        return False
    if "0 failed" in lowered and "errors" not in lowered and "traceback" not in lowered:
        return False
    return any(marker in lowered for marker in FAILURE_MARKERS)


def _failing_file_from_output(output: str) -> str | None:
    import_match = re.search(r"from '([^']+)' \(([^)]+)\)", output)
    if import_match:
        package_path = import_match.group(1).replace(".", "/")
        return f"{package_path}/__init__.py"
    path_match = re.search(r"(/[^\s)]+?\.[A-Za-z0-9]+)", output)
    if path_match:
        path = path_match.group(1)
        parts = [part for part in path.split("/") if part]
        for marker in ("calculator", "app", "src", "tests"):
            if marker in parts:
                return "/".join(parts[parts.index(marker) :])
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]
    file_match = re.search(r"\b(test_[A-Za-z0-9_]+\.[A-Za-z0-9]+)\b", output)
    if file_match:
        return file_match.group(1)
    return None


def _tool_name_set(tools: list | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", {})
        if isinstance(function, dict):
            name = function.get("name") or tool.get("name")
        else:
            name = tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def detect_phase(messages: list, user_query: str, tools: list | None = None) -> tuple[Phase, dict[str, Any]]:
    task_text = _task_text(messages, user_query)
    last = _last_tool_call(messages)
    has_injected, injected_path = _has_injected_file_content(messages)
    injected_block = _injected_file_block(messages) or ""
    last_output = _last_tool_result_text(messages)
    mentioned_files = _mentioned_files(task_text)
    written_files = set(_written_files(messages))
    read_files = set(_read_files(messages))
    planned_commands = _planned_commands(messages, task_text)
    executed_commands = set(_executed_commands(messages))
    next_command = next((command for command in planned_commands if command not in executed_commands), None)
    tool_names = _tool_name_set(tools)
    can_execute = not tools or "execute_command" in tool_names

    if can_execute and _is_testing_request(user_query) and not (
        last and last[0] == "execute_command" and _has_test_failure(last_output)
    ):
        command = next_command or _default_test_command(task_text, messages)
        if command:
            return Phase.TESTING, {"command": command, "query": task_text or user_query}

    if has_injected:
        phase = Phase.FIXING if last and last[0] == "execute_command" and _has_test_failure(last_output) else Phase.GENERATING
        return phase, {
            "file": injected_path,
            "current_content": injected_block,
            "query": task_text or user_query,
            "output": last_output,
            "action": "write_to_file",
        }

    if last is None:
        return Phase.PLANNING, {"query": task_text or user_query}

    if last[0] == "read_file":
        return Phase.GENERATING, {
            "file": last[1].get("path", ""),
            "current_content": last_output,
            "query": task_text or user_query,
            "action": "write_to_file",
        }

    if last[0] == "write_to_file":
        pending_files = [path for path in mentioned_files if path not in written_files and path not in read_files]
        if pending_files:
            return Phase.GENERATING, {"path": pending_files[0], "query": task_text or user_query, "action": "read_file"}
        if can_execute and next_command:
            return Phase.TESTING, {"command": next_command, "query": task_text or user_query}
        return Phase.COMPLETE, {
            "written": last[1].get("path", ""),
            "output": last_output,
        }

    if last[0] == "execute_command":
        if _has_test_failure(last_output):
            if _is_fix_request(user_query):
                failing_file = _failing_file_from_output(last_output)
                if failing_file and failing_file not in read_files:
                    return Phase.FIXING, {
                        "path": failing_file,
                        "query": task_text or user_query,
                        "output": last_output,
                        "action": "read_file",
                    }
                return Phase.FIXING, {
                    "file": failing_file or _mentions_file(task_text) or "",
                    "current_content": injected_block or last_output,
                    "query": task_text or user_query,
                    "output": last_output,
                    "action": "write_to_file",
                }
            return Phase.COMPLETE, {"output": last_output, "failed": True}
        if can_execute and next_command:
            return Phase.TESTING, {"command": next_command, "query": task_text or user_query}
        return Phase.COMPLETE, {"output": last_output}

    return Phase.PLANNING, {"query": task_text or user_query}


def build_phase_prompt(
    phase: Phase,
    context: dict[str, Any],
    system_message: str | None = None,
    task_history: str | None = None,
) -> str | None:
    if phase == Phase.PLANNING:
        query = context.get("query", "")
        parts = [
            "You are a coding assistant. Analyze this task and provide a concise implementation plan.",
            "Plan and implement the task deterministically, one file and one command at a time.",
            "Use the project context below to infer the language, framework, test runner, and file structure. Do not assume JavaScript if the context indicates another stack.",
        ]
        if task_history:
            parts.append(f"Conversation history:\n{task_history}")
        if system_message:
            snippets = _context_snippets(system_message)
            if snippets:
                parts.append("Project context:\n" + "\n\n".join(snippets))
            else:
                parts.append(f"Project context:\n{system_message}")
        parts.append(f"Task: {query}")
        parts.append(
            "Identify the files to create or edit, the order to handle them, and the exact shell commands to run in order."
        )
        parts.append(
            "When listing commands, keep them in execution order and wrap each full command in backticks so they can be reused exactly."
        )
        return "\n\n".join(parts)

    if phase == Phase.GENERATING:
        if context.get("action") == "read_file":
            return None
        file_path = context.get("file", "")
        query = context.get("query", "")
        current_content = context.get("current_content", "")
        parts = [
            f"Return ONLY the complete updated content of {file_path}.",
            "No markdown fences, no filename header, no explanations.",
            f"Task: {query}",
        ]
        if task_history:
            parts.append(f"Conversation history:\n{task_history}")
        if system_message:
            parts.append(f"System context:\n{system_message}")
        if current_content:
            parts.append(f"Current content:\n{current_content}")
        return "\n\n".join(parts)

    if phase == Phase.FIXING:
        if context.get("action") == "read_file":
            return None
        file_path = context.get("file", "")
        query = context.get("query", "")
        current_content = context.get("current_content", "")
        output = context.get("output", "")
        parts = [
            f"Fix the failing implementation in {file_path}. Return ONLY the complete corrected file content.",
            "No markdown fences, no filename header, no explanations.",
            f"Task context: {query}",
        ]
        if task_history:
            parts.append(f"Conversation history:\n{task_history}")
        if output:
            parts.append(f"Test output:\n{output}")
        if current_content:
            parts.append(f"Current content:\n{current_content}")
        return "\n\n".join(parts)

    return None


def build_hermes_prompt(user_query: str, system_message: str | None = None) -> str:
    persona = ""
    if system_message:
        lines = system_message.splitlines()
        persona_lines = [line.strip() for line in lines[:20] if line.strip() and not line.lstrip().startswith("#")]
        if persona_lines:
            persona = " ".join(persona_lines[:3])

    parts = [
        "Reply in plain text only. No markdown, no bullet points, no headers, no citation numbers, no links.",
        "Write as if outputting to a terminal.",
    ]
    if persona:
        parts.append(f"Persona context: {persona}")
    parts.append(f"User message: {user_query}")
    return "\n\n".join(parts)


def _phase_to_roo_decision(phase: Phase, context: dict[str, Any]) -> dict[str, Any]:
    if phase == Phase.PLANNING:
        return {
            "tool": "attempt_completion",
            "args_hint": {},
            "perplexity_instruction": None,
            "phase": phase.value,
            "phase_context": context,
        }
    if phase in {Phase.GENERATING, Phase.FIXING}:
        action = context.get("action")
        if action == "read_file":
            return {
                "tool": "read_file",
                "args_hint": {"path": context.get("path", "")},
                "perplexity_instruction": None,
                "phase": phase.value,
                "phase_context": context,
            }
        return {
            "tool": "write_to_file",
            "args_hint": {"path": context.get("file", "")},
            "perplexity_instruction": None,
            "phase": phase.value,
            "phase_context": context,
        }
    if phase == Phase.TESTING:
        return {
            "tool": "execute_command",
            "args_hint": {"command": context.get("command", "")},
            "perplexity_instruction": None,
            "phase": phase.value,
            "phase_context": context,
        }
    return {
        "tool": "attempt_completion",
        "args_hint": {},
        "perplexity_instruction": None,
        "static_result": (
            f"Command failed.\n{context.get('output', '').strip()}"
            if context.get("failed")
            else context.get("output") or f"Completed {context.get('written', 'the task')}".strip()
        ),
        "phase": phase.value,
        "phase_context": context,
    }


def decide_tool(messages: list, user_query: str, tools: list | None = None) -> dict[str, Any]:
    phase, context = detect_phase(messages, user_query, tools=tools)
    return _phase_to_roo_decision(phase, context)


def build_perplexity_instruction(decision: dict[str, Any], user_query: str, messages: list | None = None) -> str | None:
    phase_name = decision.get("phase")
    if not phase_name:
        return decision.get("perplexity_instruction")
    try:
        phase = Phase(phase_name)
    except ValueError:
        return decision.get("perplexity_instruction")
    system_message = None
    if messages:
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                system_message = content
                break
    task_history = extract_task_history(messages or []) if messages else None
    context = dict(decision.get("phase_context", {}))
    context.setdefault("query", user_query)
    if task_history:
        context.setdefault("task_history", task_history)
    return build_phase_prompt(phase, context, system_message, task_history)


def wrap_as_tool_response(
    prose: str | None,
    model: str,
    req_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    tool_name = decision["tool"]
    args_hint = decision.get("args_hint", {})

    if tool_name == "write_to_file":
        content = (prose or "").strip("\n")
        if not content.strip():
            tool_name = "attempt_completion"
            arguments = {"result": "Cannot write empty content."}
        else:
            arguments = {
                "path": args_hint.get("path", ""),
                "content": content,
                "line_count": len(content.splitlines()),
            }
    elif tool_name == "read_file":
        arguments = {"path": args_hint.get("path", "")}
    elif tool_name == "execute_command":
        arguments = {"command": args_hint.get("command", "")}
    else:
        arguments = {"result": prose or "Ready."}

    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments),
        },
    }

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def wrap_for_hermes(prose: str | None, model: str, req_id: str) -> dict[str, Any]:
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": "terminal",
            "arguments": json.dumps({"command": "echo", "output": prose or "Ready."}),
        },
    }
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": prose or "Ready.",
                    "tool_calls": [tool_call],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _message_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content", "")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return [{"type": "text", "text": content}]


def _clean_user_block_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "<user_message>" in text:
        match = re.search(r"<user_message>\s*(.*?)\s*</user_message>", text, re.DOTALL)
        if match:
            text = match.group(1)
    text = re.sub(r"<environment_details>.*?</environment_details>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?user_message>", "", text)
    text = text.strip()
    noise_prefixes = (
        "[read_file for ",
        "Command executed in terminal",
        '{"path":"',
        '{"path": "',
        "File: ",
        "Task was interrupted",
    )
    if text.startswith(noise_prefixes):
        return ""
    return text


def extract_task_history(messages: list, limit: int = 5) -> str:
    turns: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        blocks = _message_content_blocks(msg)
        turn_parts: list[str] = []
        for block in blocks:
            if block.get("type") != "text":
                continue
            raw_text = block.get("text", "")
            if not isinstance(raw_text, str) or not raw_text:
                continue
            if re.search(r"\[read_file for ['\"][^'\"]+['\"]\]", raw_text):
                turn_parts.append(raw_text.strip())
                continue
            cleaned = _clean_user_block_text(raw_text)
            if cleaned:
                turn_parts.append(cleaned)
        if turn_parts:
            turns.append("\n".join(turn_parts))
    if not turns:
        return ""
    return "\n\n".join(turns[-limit:])


def extract_read_files(messages: list) -> list[str]:
    paths: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "assistant":
            for tool_call in msg.get("tool_calls") or []:
                fn = tool_call.get("function", {})
                if not isinstance(fn, dict) or fn.get("name") != "read_file":
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                path = args.get("path")
                if isinstance(path, str) and path and path not in paths:
                    paths.append(path)
        if msg.get("role") == "user":
            for block in _message_content_blocks(msg):
                if block.get("type") != "text":
                    continue
                raw_text = block.get("text", "")
                if not isinstance(raw_text, str):
                    continue
                match = re.search(r"\[read_file for ['\"]([^'\"]+)['\"]\]", raw_text)
                if match and match.group(1) not in paths:
                    paths.append(match.group(1))
    return paths


def extract_written_files(messages: list) -> list[str]:
    return _written_files(messages)


def extract_test_commands(messages: list, task_history: str | None = None) -> list[str]:
    history = task_history or extract_task_history(messages)
    return _planned_commands(messages, history)


def extract_executed_commands(messages: list) -> list[str]:
    return _executed_commands(messages)


def detect_coding_phase(messages: list, user_query: str, tools: list | None = None) -> CodingPhase:
    task_history = extract_task_history(messages)
    last = _last_tool_call(messages)
    last_output = _last_tool_result_text(messages)
    read_files = set(extract_read_files(messages))
    written_files = set(extract_written_files(messages))
    planned_commands = extract_test_commands(messages, task_history)
    executed_commands = set(extract_executed_commands(messages))
    next_command = next((command for command in planned_commands if command not in executed_commands), None)
    tool_names = _tool_name_set(tools)
    can_execute = not tools or "execute_command" in tool_names

    if not last:
        return CodingPhase.PLANNING
    if last[0] == "read_file":
        return CodingPhase.FILE_EDIT
    if last[0] == "write_to_file":
        if can_execute and next_command:
            return CodingPhase.TESTING
        return CodingPhase.COMPLETE
    if last[0] == "execute_command":
        if _has_test_failure(last_output):
            return CodingPhase.FIXING
        if can_execute and next_command:
            return CodingPhase.TESTING
        return CodingPhase.COMPLETE
    if read_files and not written_files:
        return CodingPhase.FILE_READ
    if can_execute and _is_testing_request(user_query) and next_command:
        return CodingPhase.TESTING
    return CodingPhase.PLANNING


def build_full_context_prompt(
    phase: CodingPhase,
    context: dict[str, Any],
    system_message: str | None = None,
) -> str:
    parts = [
        "FULL PROJECT CONTEXT — multi-turn coding task in progress.",
        "Preserve all prior work. Continue exactly where left off.",
        f"Current phase: {phase.value}",
    ]
    task_history = context.get("task_history")
    if task_history:
        parts.append(f"Task history:\n{task_history}")
    read_files = context.get("read_files") or []
    if read_files:
        parts.append(f"Read files: {', '.join(read_files)}")
    written_files = context.get("written_files") or []
    if written_files:
        parts.append(f"Written files: {', '.join(written_files)}")
    current_file = context.get("current_file")
    if isinstance(current_file, dict):
        path = current_file.get("path", "")
        content = current_file.get("content", "")
        if path or content:
            parts.append(f"Current file {path}:\n{content}")
    test_output = context.get("test_output")
    if test_output:
        parts.append(f"Latest test failure:\n{test_output}")
    if system_message:
        snippets = _context_snippets(system_message)
        if snippets:
            parts.append("Project context:\n" + "\n\n".join(snippets))
        else:
            parts.append(f"Project context:\n{system_message}")
    parts.append("Respond with ONLY the next concrete action needed.")
    return "\n\n".join(parts)


def coding_shim(
    messages: list,
    user_query: str,
    tools: list | None = None,
    user_agent: str = "",
    system_message: str | None = None,
) -> dict[str, Any]:
    agent = detect_agent(tools, user_agent=user_agent)
    task_history = extract_task_history(messages)
    if agent == Agent.GENERIC:
        return {
            "agent": agent,
            "phase": CodingPhase.PLANNING,
            "task_history": task_history,
            "perplexity_prompt": user_query,
            "response_wrapper": "generic",
        }
    if agent == Agent.HERMES:
        prompt = build_hermes_prompt(user_query, system_message)
        return {
            "agent": agent,
            "phase": CodingPhase.PLANNING,
            "task_history": task_history,
            "perplexity_prompt": prompt,
            "response_wrapper": "hermes_terminal",
        }

    phase = detect_coding_phase(messages, user_query, tools=tools)
    context: dict[str, Any] = {
        "query": user_query,
        "task_history": task_history,
        "read_files": extract_read_files(messages),
        "written_files": extract_written_files(messages),
        "test_output": _last_tool_result_text(messages),
    }
    last = _last_tool_call(messages)
    if last and last[0] == "read_file":
        context["current_file"] = {"path": last[1].get("path", ""), "content": _last_tool_result_text(messages)}
    decision = decide_tool(messages, user_query, tools=tools)
    prompt = build_perplexity_instruction(decision, user_query, messages=messages) or build_full_context_prompt(phase, context, system_message)
    return {
        "agent": agent,
        "phase": phase,
        "task_history": task_history,
        "perplexity_prompt": prompt,
        "full_context_prompt": build_full_context_prompt(phase, context, system_message),
        "response_wrapper": "roo_tool",
        "tool_name": decision.get("tool"),
        "tool_args": decision.get("args_hint", {}),
        "decision": decision,
        "context": context,
    }

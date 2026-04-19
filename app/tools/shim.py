from __future__ import annotations

import json
import re
import time
import uuid
from enum import Enum
from typing import Any


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


class Phase(str, Enum):
    PLANNING = "planning"
    READING = "reading"
    GENERATING = "generating"
    TESTING = "testing"
    FIXING = "fixing"
    COMPLETE = "complete"


def is_roo_request(tools: list | None) -> bool:
    if not tools:
        return False
    for tool in tools:
        name = None
        if isinstance(tool, dict):
            function = tool.get("function", {})
            if isinstance(function, dict):
                name = function.get("name") or tool.get("name")
            else:
                name = tool.get("name")
        if name in ROO_TOOL_NAMES:
            return True
    return False


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
            cleaned = re.sub(r"<environment_details>.*?</environment_details>", "", text, flags=re.DOTALL).strip()
            cleaned = re.sub(r"</?user_message>", "", cleaned).strip()
            if not cleaned:
                continue
            if cleaned not in parts:
                parts.append(cleaned)
    if user_query and user_query not in parts:
        parts.append(user_query)
    return "\n".join(parts)


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
    failure_markers = (
        "error collecting",
        "traceback",
        "importerror",
        "assertionerror",
        "failed",
        "error:",
        "exception",
        "no module named",
    )
    return any(marker in lowered for marker in failure_markers)


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


def detect_phase(messages: list, user_query: str) -> tuple[Phase, dict[str, Any]]:
    task_text = _task_text(messages, user_query)
    last = _last_tool_call(messages)
    has_injected, injected_path = _has_injected_file_content(messages)
    injected_block = _injected_file_block(messages) or ""
    last_output = _last_tool_result_text(messages)
    mentioned_files = _mentioned_files(task_text)
    written_files = set(_written_files(messages))
    read_files = set(_read_files(messages))
    requested_commands = _requested_commands(task_text)

    if has_injected:
        phase = Phase.FIXING if last and last[0] == "execute_command" and _has_test_failure(last_output) else Phase.GENERATING
        return phase, {
            "file": injected_path,
            "current_content": injected_block,
            "query": task_text or user_query,
            "output": last_output,
        }

    if last is None:
        return Phase.PLANNING, {"query": task_text or user_query}

    if last[0] == "read_file":
        return Phase.GENERATING, {
            "file": last[1].get("path", ""),
            "current_content": last_output,
            "query": task_text or user_query,
        }

    if last[0] == "write_to_file":
        pending_files = [path for path in mentioned_files if path not in written_files and path not in read_files]
        if pending_files:
            return Phase.READING, {"path": pending_files[0], "query": task_text or user_query}
        if requested_commands:
            return Phase.TESTING, {"command": requested_commands[0], "query": task_text or user_query}
        return Phase.COMPLETE, {
            "written": last[1].get("path", ""),
            "output": last_output,
        }

    if last[0] == "execute_command":
        if _has_test_failure(last_output):
            failing_file = _failing_file_from_output(last_output)
            if failing_file and failing_file not in read_files:
                return Phase.READING, {
                    "path": failing_file,
                    "query": task_text or user_query,
                    "output": last_output,
                }
            return Phase.FIXING, {
                "file": failing_file or _mentions_file(task_text) or "",
                "current_content": injected_block or last_output,
                "query": task_text or user_query,
                "output": last_output,
            }
        return Phase.COMPLETE, {"output": last_output}

    return Phase.PLANNING, {"query": task_text or user_query}


def build_phase_prompt(phase: Phase, context: dict[str, Any], system_message: str | None = None) -> str | None:
    if phase == Phase.PLANNING:
        query = context.get("query", "")
        return (
            "You are a coding assistant. Analyze this task and provide a concise implementation plan.\n\n"
            f"Task: {query}\n\n"
            "Identify the files to create or edit, the order to handle them, and the test command to run at the end. "
            "Be concrete and actionable."
        )

    if phase == Phase.GENERATING:
        file_path = context.get("file", "")
        query = context.get("query", "")
        current_content = context.get("current_content", "")
        parts = [
            f"Return ONLY the complete updated content of {file_path}.",
            "No markdown fences, no filename header, no explanations.",
            f"Task: {query}",
        ]
        if system_message:
            parts.append(f"System context:\n{system_message}")
        if current_content:
            parts.append(f"Current content:\n{current_content}")
        return "\n\n".join(parts)

    if phase == Phase.FIXING:
        file_path = context.get("file", "")
        query = context.get("query", "")
        current_content = context.get("current_content", "")
        output = context.get("output", "")
        parts = [
            f"Fix the failing implementation in {file_path}. Return ONLY the complete corrected file content.",
            "No markdown fences, no filename header, no explanations.",
            f"Task context: {query}",
        ]
        if output:
            parts.append(f"Test output:\n{output}")
        if current_content:
            parts.append(f"Current content:\n{current_content}")
        return "\n\n".join(parts)

    return None


def decide_tool(messages: list, user_query: str) -> dict[str, Any]:
    phase, context = detect_phase(messages, user_query)
    if phase == Phase.PLANNING:
        return {
            "tool": "attempt_completion",
            "args_hint": {},
            "perplexity_instruction": None,
            "phase": phase.value,
            "phase_context": context,
        }
    if phase == Phase.READING:
        return {
            "tool": "read_file",
            "args_hint": {"path": context.get("path", "")},
            "perplexity_instruction": None,
            "phase": phase.value,
            "phase_context": context,
        }
    if phase in {Phase.GENERATING, Phase.FIXING}:
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
        "static_result": context.get("output") or f"Completed {context.get('written', 'the task')}".strip(),
        "phase": phase.value,
        "phase_context": context,
    }


def build_perplexity_instruction(decision: dict[str, Any], user_query: str, messages: list | None = None) -> str | None:
    phase_name = decision.get("phase")
    if not phase_name:
        return decision.get("perplexity_instruction")
    try:
        phase = Phase(phase_name)
    except ValueError:
        return decision.get("perplexity_instruction")
    return build_phase_prompt(phase, decision.get("phase_context", {}), None)


def wrap_as_tool_response(
    prose: str | None,
    model: str,
    req_id: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    tool_name = decision["tool"]
    args_hint = decision.get("args_hint", {})

    if tool_name == "write_to_file":
        content = prose or ""
        arguments: dict[str, Any] = {
            "path": args_hint.get("path", ""),
            "content": content,
            "line_count": len(content.splitlines()),
        }
    elif tool_name == "read_file":
        arguments = {"path": args_hint.get("path", "")}
    elif tool_name == "execute_command":
        arguments = {"command": args_hint.get("command", "")}
    else:
        arguments = {"result": prose or ""}

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

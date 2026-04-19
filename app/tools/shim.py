from __future__ import annotations

import json
import re
import time
import uuid
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
    """Return the first filename mentioned in text, if any."""
    match = re.search(r"""['"`]?([a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10})['"`]?""", text)
    return match.group(1) if match else None


def _mentioned_files(text: str) -> list[str]:
    """Return unique filenames mentioned in text, preserving order."""
    paths: list[str] = []
    for match in re.finditer(r"""['"`]?([a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10})['"`]?""", text):
        path = match.group(1)
        if path not in paths:
            paths.append(path)
    return paths


_EDIT_KEYWORDS = frozenset(
    {
        "add",
        "edit",
        "change",
        "update",
        "modify",
        "fix",
        "refactor",
        "append",
        "insert",
        "remove",
        "delete",
        "rewrite",
        "create",
        "write",
        "implement",
        "put",
        "set",
        "replace",
    }
)


_CREATE_KEYWORDS = frozenset(
    {
        "create",
        "make",
        "new",
        "generate",
        "scaffold",
        "initialize",
        "init",
        "build",
        "write",
    }
)


_CMD_KEYWORDS = frozenset({"run", "execute", "test", "pytest", "install", "check"})


def _is_edit_intent(text: str) -> bool:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(words & _EDIT_KEYWORDS)


def _is_create_intent(text: str) -> bool:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(words & _CREATE_KEYWORDS)


def _is_command_intent(text: str) -> bool:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(words & _CMD_KEYWORDS)


def _has_injected_file_content(messages: list) -> tuple[bool, str]:
    """
    Returns (True, path) if the last user message already contains
    a Roo-injected [read_file for 'path'] block.
    """
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
    """Return the Roo-injected read_file text block from the last user message, if present."""
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
    """Build a stable task description from user messages plus the latest extracted query."""
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
                continue
            command = args.get("command")
            if isinstance(command, str) and command and command not in commands:
                commands.append(command)
    return commands


def _requested_commands(text: str) -> list[str]:
    commands: list[str] = []
    for match in re.finditer(r"`([^`]+)`", text):
        command = match.group(1).strip()
        if command and _is_command_intent(command) and command not in commands:
            commands.append(command)
    return commands


def _next_pending_file(user_query: str, messages: list) -> str | None:
    written = set(_written_files(messages))
    for path in _mentioned_files(user_query):
        if path not in written:
            return path
    return None


def _next_pending_command(user_query: str, messages: list) -> str | None:
    executed = set(_executed_commands(messages))
    for command in _requested_commands(user_query):
        if command not in executed:
            return command
    return None


def decide_tool(messages: list, user_query: str) -> dict:
    """
    Decide which Roo tool to invoke based on the latest conversation state.
    """
    task_text = _task_text(messages, user_query)
    query_lower = user_query.lower()
    task_lower = task_text.lower()
    last = _last_tool_call(messages)
    next_file = _next_pending_file(task_text, messages) if task_text else None
    next_command = _next_pending_command(task_text, messages) if task_text else None

    has_injected, injected_path = _has_injected_file_content(messages)
    if has_injected and (_is_edit_intent(query_lower) or _is_edit_intent(task_lower)):
        return {
            "tool": "write_to_file",
            "args_hint": {"path": injected_path},
            "perplexity_instruction": (
                f"Return ONLY the complete updated content of {injected_path}. "
                f"No explanation, no markdown, no code fences, no filename header. "
                f"Just the raw file content."
            ),
            "content_source": "injected_read",
        }

    if last and last[0] == "write_to_file":
        if next_file:
            return {
                "tool": "write_to_file",
                "args_hint": {"path": next_file},
                "perplexity_instruction": (
                    f"Return ONLY the complete new or updated content of {next_file}. "
                    f"No explanation, no markdown, no code fences, no filename header. "
                    f"Just the raw file content."
                ),
                "content_source": "multi_file",
            }
        if next_command:
            return {
                "tool": "execute_command",
                "args_hint": {"command": next_command},
                "perplexity_instruction": None,
                "content_source": "multi_file",
            }
        written_path = last[1].get("path", "the file")
        return {
            "tool": "attempt_completion",
            "args_hint": {},
            "perplexity_instruction": None,
            "static_result": f"The file `{written_path}` has been updated successfully.",
            "content_source": "post_write",
        }

    if last and last[0] == "read_file":
        file_path = last[1].get("path", "")
        if _is_edit_intent(task_lower) or _is_create_intent(task_lower) or file_path:
            return {
                "tool": "write_to_file",
                "args_hint": {"path": file_path},
                "perplexity_instruction": (
                    f"Return ONLY the complete updated content of {file_path}. "
                    f"No explanation, no markdown, no code fences, no filename header. "
                    f"Just the raw file content."
                ),
                "content_source": "assistant_read",
            }

    mentioned_file = _mentions_file(task_text)
    if mentioned_file and _is_create_intent(task_lower):
        return {
            "tool": "write_to_file",
            "args_hint": {"path": mentioned_file},
            "perplexity_instruction": (
                f"Return ONLY the complete new content of {mentioned_file}. "
                f"No explanation, no markdown, no code fences, no filename header. "
                f"Just the raw file content."
            ),
            "content_source": "request",
        }

    if mentioned_file and _is_edit_intent(task_lower):
        if not last or last[0] != "read_file":
            return {
                "tool": "read_file",
                "args_hint": {"path": mentioned_file},
                "perplexity_instruction": None,
                "content_source": "request",
            }

    if next_command:
        return {
            "tool": "execute_command",
            "args_hint": {"command": next_command},
            "perplexity_instruction": None,
            "content_source": "request",
        }

    if user_query.strip().endswith("?") and len(user_query) < 200:
        return {
            "tool": "ask_followup_question",
            "args_hint": {},
            "perplexity_instruction": user_query,
        }

    return {
        "tool": "attempt_completion",
        "args_hint": {},
        "perplexity_instruction": user_query,
    }


def build_perplexity_instruction(decision: dict, user_query: str, messages: list | None = None) -> str | None:
    """
    Return the prompt that should be sent to Perplexity, if any.
    """
    base = decision.get("perplexity_instruction")
    if base is None:
        return None

    if decision.get("tool") == "write_to_file" and messages:
        injected_block = _injected_file_block(messages)
        if injected_block:
            return f"{base}\n\nCurrent file content:\n{injected_block}\n\nUser request: {user_query}"

    if user_query and user_query not in base:
        return f"{base}\n\nUser request: {user_query}"
    return base


def wrap_as_tool_response(
    prose: str | None,
    model: str,
    req_id: str,
    decision: dict,
) -> dict:
    """
    Build an OpenAI-compatible chat.completion payload with a single tool call.
    """
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
    elif tool_name == "attempt_completion":
        arguments = {"result": prose or ""}
    elif tool_name == "ask_followup_question":
        arguments = {"question": prose or "", "follow_up": []}
    elif tool_name == "execute_command":
        arguments = {"command": args_hint.get("command", "")}
    else:
        arguments = dict(args_hint)

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

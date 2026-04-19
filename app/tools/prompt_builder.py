from __future__ import annotations

from typing import Any


ROO_TOOL_SYSTEM_PROMPT = """
You are editing an existing source file.

Return ONLY the complete updated file content.

RULES:
- Output raw file content only
- Do NOT include markdown code fences
- Do NOT include a filename header
- Do NOT include explanations or commentary
- Include the COMPLETE file content, not a diff or partial snippet
""".strip()


def _context_snippets(system_message: str) -> list[str]:
    snippets: list[str] = []
    for section in ("SYSTEM INFORMATION", "RULES", "Current Workspace"):
        if section not in system_message:
            continue
        start = system_message.find(section)
        end = system_message.find("\n====", start + 1)
        snippet = system_message[start : end if end > 0 else start + 500].strip()
        if snippet:
            snippets.append(snippet)
    return snippets


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts)
    return ""


def _tool_file_contents(messages: list[Any]) -> list[str]:
    file_contents: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "tool" and isinstance(content, str) and content.strip():
            file_contents.append(content.strip())

        if role == "user" and isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "tool_result":
                    continue
                inner = part.get("content", "")
                if isinstance(inner, str) and inner.strip():
                    file_contents.append(inner.strip())
                elif isinstance(inner, list):
                    for block in inner:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if isinstance(text, str) and text.strip():
                                file_contents.append(text.strip())
    return file_contents


def build_tool_aware_query(
    original_query: str,
    system_message: str | None,
    messages: list | None = None,
) -> str:
    parts: list[str] = []

    if system_message:
        for snippet in _context_snippets(system_message):
            parts.append(f"[CONTEXT]\n{snippet}")

    parts.append(ROO_TOOL_SYSTEM_PROMPT)

    if messages:
        file_contents = _tool_file_contents(messages)
        if file_contents:
            parts.append("[CURRENT FILE CONTENT]\n" + "\n---\n".join(file_contents))

    parts.append(f"[TASK]\n{original_query}")
    return "\n\n".join(parts)

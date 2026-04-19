from __future__ import annotations

from app.tools.prompt_builder import build_tool_aware_query


def test_build_tool_aware_query_requests_raw_content_and_tool_content():
    query = build_tool_aware_query(
        "add a divide function",
        None,
        messages=[{"role": "tool", "content": "def add(a,b):\n    return a+b\n"}],
    )

    assert "Return ONLY the complete updated file content." in query
    assert "Do NOT include markdown code fences" in query
    assert "[CURRENT FILE CONTENT]" in query
    assert "def add(a,b):" in query
    assert query.endswith("[TASK]\nadd a divide function")


def test_build_tool_aware_query_includes_context_snippets_from_system_message():
    system_message = """prefix
SYSTEM INFORMATION
OS: macOS
====
RULES
Be precise
====
Current Workspace
/tmp/work
"""

    query = build_tool_aware_query("fix it", system_message, messages=None)

    assert query.count("[CONTEXT]") == 3
    assert "OS: macOS" in query
    assert "Be precise" in query
    assert "/tmp/work" in query

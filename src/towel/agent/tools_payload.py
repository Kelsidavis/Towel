"""Helpers for converting Towel skill tool definitions to backend-native shapes.

Each LLM backend wants a slightly different schema for the tools list:

- OpenAI-compatible (Ollama, llama-server, transformers chat template):
    ``[{"type": "function", "function": {"name", "description", "parameters"}}, ...]``
- Anthropic Messages API:
    ``[{"name", "description", "input_schema"}, ...]``

These helpers normalise the conversion so runtimes don't reimplement it.
"""

from __future__ import annotations

from typing import Any, Iterable


def _empty_object_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {}}


def tools_as_openai_functions(tool_defs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render registered tools as OpenAI-style function dicts.

    Suitable for ``apply_chat_template(tools=...)``, Ollama's ``tools`` field,
    and llama-server's OpenAI-compatible ``/v1/chat/completions``.
    """
    out: list[dict[str, Any]] = []
    for t in tool_defs:
        params = t.get("parameters") or _empty_object_schema()
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return out


def tools_as_anthropic(tool_defs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render registered tools in Anthropic Messages-API shape.

    Anthropic uses ``input_schema`` instead of ``parameters`` and does not wrap
    each tool in a ``{"type": "function", "function": ...}`` envelope.
    """
    out: list[dict[str, Any]] = []
    for t in tool_defs:
        out.append(
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters") or _empty_object_schema(),
            }
        )
    return out

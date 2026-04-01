"""Memory skill — lets the agent remember and recall across sessions."""

from __future__ import annotations

from typing import Any

from towel.memory.store import MEMORY_TYPES, MemoryStore
from towel.skills.base import Skill, ToolDefinition


class MemorySkill(Skill):
    """Gives the agent persistent memory across conversations."""

    def __init__(self, store: MemoryStore | None = None) -> None:
        self._store = store or MemoryStore()

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Remember and recall facts across sessions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="remember",
                description=(
                    "Store a fact in persistent memory. Use for "
                    "user preferences, project details, or "
                    "anything worth remembering across sessions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": (
                                "Short identifier (e.g., "
                                "'user_name', 'preferred_language')"
                            ),
                        },
                        "content": {"type": "string", "description": "The fact to remember"},
                        "type": {
                            "type": "string",
                            "description": (
                                "Memory type: "
                                f"{', '.join(MEMORY_TYPES)} "
                                "(default: fact)"
                            ),
                        },
                    },
                    "required": ["key", "content"],
                },
            ),
            ToolDefinition(
                name="forget",
                description="Remove a memory by key.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The memory key to forget"},
                    },
                    "required": ["key"],
                },
            ),
            ToolDefinition(
                name="recall",
                description=(
                    "Search your memories. Use when the user "
                    "references something from a past session."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term (searches keys and content)",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "remember":
                key = arguments["key"]
                content = arguments["content"]
                mtype = arguments.get("type", "fact")
                if mtype not in MEMORY_TYPES:
                    mtype = "fact"
                entry = self._store.remember(key, content, memory_type=mtype)
                return f"Remembered [{entry.memory_type}] {entry.key}: {entry.content}"

            case "forget":
                key = arguments["key"]
                if self._store.forget(key):
                    return f"Forgot: {key}"
                return f"No memory found for key: {key}"

            case "recall":
                query = arguments["query"]
                results = self._store.search(query)
                if not results:
                    return "No matching memories found."
                lines = [f"Found {len(results)} memory(ies):"]
                for e in results:
                    lines.append(f"  [{e.memory_type}] {e.key}: {e.content}")
                return "\n".join(lines)

            case _:
                return f"Unknown tool: {tool_name}"

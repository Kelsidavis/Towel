"""Quick notes skill — ephemeral scratchpad for the current session."""

from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

_notes: dict[str, str] = {}


class NoteSkill(Skill):
    @property
    def name(self) -> str: return "notes"
    @property
    def description(self) -> str: return "Session scratchpad — quick notes that last until restart"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="note_set", description="Save a quick note (session-only, not persisted)",
                parameters={"type":"object","properties":{
                    "key":{"type":"string","description":"Note name"},
                    "value":{"type":"string","description":"Note content"},
                },"required":["key","value"]}),
            ToolDefinition(name="note_get", description="Retrieve a note by name",
                parameters={"type":"object","properties":{
                    "key":{"type":"string","description":"Note name"},
                },"required":["key"]}),
            ToolDefinition(name="note_list", description="List all session notes",
                parameters={"type":"object","properties":{}}),
            ToolDefinition(name="note_clear", description="Clear all session notes",
                parameters={"type":"object","properties":{}}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "note_set":
                _notes[arguments["key"]] = arguments["value"]
                return f"Noted: {arguments['key']}"
            case "note_get":
                return _notes.get(arguments["key"], f"No note: {arguments['key']}")
            case "note_list":
                if not _notes: return "No notes."
                return "\n".join(f"  {k}: {v[:80]}" for k, v in _notes.items())
            case "note_clear":
                count = len(_notes); _notes.clear()
                return f"Cleared {count} notes."
            case _: return f"Unknown tool: {tool_name}"

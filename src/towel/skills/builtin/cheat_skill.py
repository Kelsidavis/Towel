"""Cheat sheet skill — quick references via cheat.sh."""
from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class CheatSkill(Skill):
    @property
    def name(self) -> str: return "cheat"
    @property
    def description(self) -> str: return "Quick cheat sheets via cheat.sh"
    def tools(self) -> list[ToolDefinition]:
        return [ToolDefinition(name="cheat_sheet", description="Get a cheat sheet for a command or topic",
            parameters={"type":"object","properties":{"topic":{"type":"string","description":"Command or topic (e.g., curl, python/list, tar)"}},"required":["topic"]})]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "cheat_sheet": return f"Unknown: {tool_name}"
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                resp = await c.get(f"https://cheat.sh/{arguments['topic']}",
                    headers={"User-Agent": "curl"})
                # Strip ANSI color codes
                import re
                text = re.sub(r'\x1b\[[0-9;]*m', '', resp.text)
                return text[:5000]
        except Exception as e: return f"Error: {e}"

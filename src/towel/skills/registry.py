"""Skill registry — discovers, loads, and manages skills."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class SkillRegistry:
    """Central registry of loaded skills and their tools."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._tool_map: dict[str, Skill] = {}  # tool_name -> owning skill

    def register(self, skill: Skill) -> None:
        """Register a skill and index its tools."""
        self._skills[skill.name] = skill
        for tool in skill.tools():
            self._tool_map[tool.name] = skill

    def unregister(self, name: str) -> None:
        skill = self._skills.pop(name, None)
        if skill:
            for tool in skill.tools():
                self._tool_map.pop(tool.name, None)

    def get_skill(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions across all skills."""
        defs: list[dict[str, Any]] = []
        for skill in self._skills.values():
            defs.extend(t.to_dict() for t in skill.tools())
        return defs

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name, routing to the correct skill."""
        skill = self._tool_map.get(tool_name)
        if not skill:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await skill.execute(tool_name, arguments)

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

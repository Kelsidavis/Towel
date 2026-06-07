"""Skill registry — discovers, loads, and manages skills."""

from __future__ import annotations

import time
from difflib import get_close_matches
from typing import Any

from towel.audit import audit_tool_call
from towel.policy import get_policy
from towel.skills.base import Skill


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

    def has_tool(self, tool_name: str) -> bool:
        """Return whether a tool name is registered."""
        return tool_name in self._tool_map

    def tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tool_map.keys())

    def suggest_tools(self, tool_name: str, limit: int = 3) -> list[str]:
        """Return close tool-name matches for recovery from model mistakes."""
        return get_close_matches(tool_name, self.tool_names(), n=limit, cutoff=0.5)

    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return all tool definitions across all skills."""
        defs: list[dict[str, Any]] = []
        for skill in self._skills.values():
            defs.extend(t.to_dict() for t in skill.tools())
        return defs

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name, routing to the correct skill.

        Every invocation is recorded to the durable tool-call audit log
        (towel.audit) with its outcome, so a rogue-model session can be
        reconstructed after the fact. Auditing is best-effort and never
        changes the tool's result or raises on its own.
        """
        skill = self._tool_map.get(tool_name)
        if not skill:
            suggestions = self.suggest_tools(tool_name)
            audit_tool_call(
                tool_name, arguments, status="error",
                error="unknown tool",
            )
            if suggestions:
                raise ValueError(
                    f"Unknown tool: {tool_name}. Did you mean: {', '.join(suggestions)}?"
                )
            raise ValueError(f"Unknown tool: {tool_name}")
        # Gating policy: refuse blocked capabilities before they run. The
        # default 'audit' mode permits everything (non-breaking); enforce
        # mode blocks dangerous risk tiers. Refusals are audited, not run.
        denied = get_policy().evaluate(tool_name)
        if denied is not None:
            audit_tool_call(tool_name, arguments, status="blocked", result=denied)
            return denied
        start = time.monotonic()
        try:
            result = await skill.execute(tool_name, arguments)
        except Exception as exc:
            audit_tool_call(
                tool_name, arguments, status="error",
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
            raise
        # A guard refusal comes back as a normal string result, not an
        # exception — tag those as 'blocked' so they're easy to alert on.
        status = "ok"
        if isinstance(result, str) and result.startswith("refused:"):
            status = "blocked"
        audit_tool_call(
            tool_name, arguments, status=status, result=result,
            duration_ms=(time.monotonic() - start) * 1000,
        )
        return result

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

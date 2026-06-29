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

    # Canonical tool names small models routinely miss — they emit the
    # task/skill word ("shell") instead of the registered tool ("run_command").
    # Resolving these here means a <|tool_call>call:shell\n<cmd> emission
    # (parsed as {"input": "<cmd>"} by tool_parser) actually executes instead
    # of bouncing off an "Unknown tool: shell" error. The lone positional from
    # such calls is remapped onto the target tool's primary parameter.
    _TOOL_ALIASES: dict[str, str] = {
        "shell": "run_command",
        "bash": "run_command",
        "sh": "run_command",
        "exec": "run_command",
        "command": "run_command",
        "terminal": "run_command",
        "run_shell": "run_command",
    }

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

    def _primary_arg_key(self, tool_name: str) -> str | None:
        """Best-guess primary parameter for a tool — the first ``required``
        entry, else the first declared property. Used to place a lone
        positional value (e.g. an aliased ``{"input": ...}``) onto the key the
        tool actually expects.
        """
        skill = self._tool_map.get(tool_name)
        if not skill:
            return None
        for tool in skill.tools():
            if tool.name != tool_name:
                continue
            params = getattr(tool, "parameters", None) or {}
            required = params.get("required")
            if isinstance(required, list) and required:
                return str(required[0])
            props = params.get("properties")
            if isinstance(props, dict) and props:
                return next(iter(props))
            return None
        return None

    def _resolve_alias(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Map a misnamed tool to its canonical name and realign its args.

        Returns ``(tool_name, arguments)`` unchanged when no alias applies or
        the canonical target isn't registered. When an alias resolves and the
        caller passed a lone ``{"input": value}`` positional, the value is
        moved onto the target tool's primary parameter so the call can run.
        """
        canonical = self._TOOL_ALIASES.get(tool_name)
        if not canonical or canonical not in self._tool_map:
            return tool_name, arguments
        if list(arguments.keys()) == ["input"]:
            primary = self._primary_arg_key(canonical)
            if primary and primary != "input":
                arguments = {primary: arguments["input"]}
        return canonical, arguments

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
        # Recover from common name mistakes (e.g. shell -> run_command) before
        # giving up, realigning a lone positional onto the real parameter.
        if tool_name not in self._tool_map:
            tool_name, arguments = self._resolve_alias(tool_name, arguments)
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

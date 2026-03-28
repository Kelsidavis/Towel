"""Pip/Python project skill — inspect requirements, pyproject, virtual environments."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class PipSkill(Skill):
    @property
    def name(self) -> str: return "pip"
    @property
    def description(self) -> str: return "Inspect Python dependencies, virtual environments, and project config"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="pip_list", description="List installed Python packages (in current env)",
                parameters={"type":"object","properties":{
                    "filter":{"type":"string","description":"Filter by package name"},
                }}),
            ToolDefinition(name="pip_requirements", description="Parse and analyze a requirements.txt file",
                parameters={"type":"object","properties":{
                    "path":{"type":"string","description":"Path to requirements.txt (default: requirements.txt)"},
                }}),
            ToolDefinition(name="pip_venv_info", description="Show info about the current Python/virtual environment",
                parameters={"type":"object","properties":{}}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "pip_list": return await self._list(arguments.get("filter"))
            case "pip_requirements": return self._requirements(arguments.get("path", "requirements.txt"))
            case "pip_venv_info": return self._venv_info()
            case _: return f"Unknown tool: {tool_name}"

    async def _list(self, name_filter: str|None) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "list", "--format=columns",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = stdout.decode("utf-8", errors="replace").strip()
            if name_filter:
                lines = output.splitlines()
                header = lines[:2]
                filtered = [l for l in lines[2:] if name_filter.lower() in l.lower()]
                return "\n".join(header + filtered) if filtered else f"No packages matching '{name_filter}'"
            return output
        except Exception as e:
            return f"Error: {e}"

    def _requirements(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file(): return f"Not found: {path}"
        lines = p.read_text().strip().splitlines()
        pkgs = [l.strip() for l in lines if l.strip() and not l.startswith("#") and not l.startswith("-")]
        pinned = [p for p in pkgs if "==" in p]
        unpinned = [p for p in pkgs if "==" not in p and p]
        result = [f"requirements.txt ({len(pkgs)} packages):"]
        result.append(f"  Pinned: {len(pinned)}")
        result.append(f"  Unpinned: {len(unpinned)}")
        if unpinned:
            result.append(f"\n  Unpinned packages:")
            for u in unpinned[:20]: result.append(f"    {u}")
        return "\n".join(result)

    def _venv_info(self) -> str:
        lines = [f"Python: {sys.version}"]
        lines.append(f"Executable: {sys.executable}")
        lines.append(f"Prefix: {sys.prefix}")
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            lines.append(f"Virtual env: {venv}")
        elif sys.prefix != sys.base_prefix:
            lines.append(f"Virtual env: {sys.prefix} (detected)")
        else:
            lines.append("Virtual env: none (system Python)")
        lines.append(f"Platform: {sys.platform}")
        lines.append(f"PATH entries: {len(os.environ.get('PATH','').split(os.pathsep))}")
        return "\n".join(lines)

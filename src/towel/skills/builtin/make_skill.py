"""Makefile skill — parse, list targets, and run make commands."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class MakeSkill(Skill):
    @property
    def name(self) -> str:
        return "make"

    @property
    def description(self) -> str:
        return "Parse Makefiles — list targets, show recipes, run commands"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="make_targets",
                description="List all targets in a Makefile with descriptions",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to Makefile (default: Makefile)",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="make_recipe",
                description="Show the recipe/commands for a specific target",
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target name"},
                        "path": {
                            "type": "string",
                            "description": "Path to Makefile (default: Makefile)",
                        },
                    },
                    "required": ["target"],
                },
            ),
            ToolDefinition(
                name="make_run",
                description="Run a make target and return output",
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target to run"},
                        "path": {
                            "type": "string",
                            "description": "Working directory (default: cwd)",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Show commands without executing (default: false)",
                        },
                    },
                    "required": ["target"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "make_targets":
                return self._targets(arguments.get("path", "Makefile"))
            case "make_recipe":
                return self._recipe(arguments["target"], arguments.get("path", "Makefile"))
            case "make_run":
                return await self._run(
                    arguments["target"], arguments.get("path", "."), arguments.get("dry_run", False)
                )
            case _:
                return f"Unknown tool: {tool_name}"

    def _parse(self, path: str) -> tuple[Path, str] | str:
        p = Path(path).expanduser()
        if p.is_dir():
            p = p / "Makefile"
        if not p.is_file():
            return f"Not found: {p}"
        return p, p.read_text(encoding="utf-8", errors="replace")

    def _targets(self, path: str) -> str:
        result = self._parse(path)
        if isinstance(result, str):
            return result
        p, content = result
        targets = []
        lines = content.splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^([a-zA-Z_][\w.-]*)\s*:", line)
            if m and not line.startswith("\t"):
                name = m.group(1)
                # Look for comment above
                comment = ""
                if i > 0 and lines[i - 1].strip().startswith("#"):
                    comment = lines[i - 1].strip().lstrip("# ")
                # Check for .PHONY
                is_phony = any(name in line for line in lines if line.startswith(".PHONY"))
                marker = "*" if is_phony else " "
                targets.append(f"  {marker} {name:20s}  {comment}")
        if not targets:
            return f"No targets found in {p.name}"
        return f"Targets in {p.name} ({len(targets)}, * = phony):\n" + "\n".join(targets)

    def _recipe(self, target: str, path: str) -> str:
        result = self._parse(path)
        if isinstance(result, str):
            return result
        p, content = result
        lines = content.splitlines()
        found = False
        recipe = []
        for line in lines:
            if found:
                if line.startswith("\t"):
                    recipe.append(line[1:])  # strip leading tab
                elif line.strip() == "":
                    continue
                else:
                    break
            elif re.match(rf"^{re.escape(target)}\s*:", line):
                found = True
                # Dependencies
                deps = line.split(":", 1)[1].strip()
                if deps:
                    recipe.append(f"# depends on: {deps}")
        if not found:
            return f"Target not found: {target}"
        if not recipe:
            return f"Target '{target}' has no recipe."
        return f"Recipe for '{target}':\n" + "\n".join(f"  {r}" for r in recipe)

    async def _run(self, target: str, path: str, dry_run: bool) -> str:
        args = ["make", "-C", path, target]
        if dry_run:
            args.insert(1, "-n")
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            out = stdout.decode("utf-8", "replace").strip()
            err = stderr.decode("utf-8", "replace").strip()
            if proc.returncode != 0:
                return f"make {target} failed (exit {proc.returncode}):\n{err or out}"
            return out or "(no output)"
        except FileNotFoundError:
            return "make not found."
        except TimeoutError:
            return "make timed out (60s)."

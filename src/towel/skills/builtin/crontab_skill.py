"""Crontab skill — read and manage the user's actual system crontab."""

from __future__ import annotations

import asyncio
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class CrontabSkill(Skill):
    @property
    def name(self) -> str: return "crontab"
    @property
    def description(self) -> str: return "Read and manage the system crontab"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="crontab_list", description="Show the current user's crontab entries",
                parameters={"type":"object","properties":{}}),
            ToolDefinition(name="crontab_add", description="Add a new crontab entry",
                parameters={"type":"object","properties":{
                    "schedule":{"type":"string","description":"Cron expression (e.g., '*/5 * * * *')"},
                    "command":{"type":"string","description":"Command to run"},
                    "comment":{"type":"string","description":"Comment/label (optional)"},
                },"required":["schedule","command"]}),
            ToolDefinition(name="crontab_remove", description="Remove a crontab entry by line number",
                parameters={"type":"object","properties":{
                    "line":{"type":"integer","description":"Line number (from crontab_list, 1-based)"},
                },"required":["line"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "crontab_list": return await self._list()
            case "crontab_add": return await self._add(arguments["schedule"], arguments["command"], arguments.get("comment"))
            case "crontab_remove": return await self._remove(arguments["line"])
            case _: return f"Unknown tool: {tool_name}"

    async def _run(self, cmd: list[str], input_data: str|None = None) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if input_data else None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_data.encode() if input_data else None), timeout=10
        )
        return proc.returncode or 0, stdout.decode("utf-8","replace").strip(), stderr.decode("utf-8","replace").strip()

    async def _list(self) -> str:
        code, out, err = await self._run(["crontab", "-l"])
        if code != 0 or "no crontab" in err.lower():
            return "No crontab entries."
        lines = out.splitlines()
        entries = []
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith("#"):
                entries.append(f"  {i+1}. {line}")
            elif line.startswith("#"):
                entries.append(f"  {i+1}. [dim]{line}[/dim]")
        return f"Crontab ({len(entries)} lines):\n" + "\n".join(entries) if entries else "Crontab is empty."

    async def _add(self, schedule: str, command: str, comment: str|None) -> str:
        code, existing, _ = await self._run(["crontab", "-l"])
        if code != 0: existing = ""
        new_line = f"{schedule} {command}"
        if comment: new_line = f"# {comment}\n{new_line}"
        new_crontab = existing + "\n" + new_line + "\n"
        code, _, err = await self._run(["crontab", "-"], input_data=new_crontab)
        if code != 0: return f"Failed: {err}"
        return f"Added: {schedule} {command}"

    async def _remove(self, line: int) -> str:
        code, existing, _ = await self._run(["crontab", "-l"])
        if code != 0: return "No crontab."
        lines = existing.splitlines()
        if line < 1 or line > len(lines): return f"Invalid line: {line} (crontab has {len(lines)} lines)"
        removed = lines.pop(line - 1)
        new_crontab = "\n".join(lines) + "\n"
        code, _, err = await self._run(["crontab", "-"], input_data=new_crontab)
        if code != 0: return f"Failed: {err}"
        return f"Removed line {line}: {removed}"

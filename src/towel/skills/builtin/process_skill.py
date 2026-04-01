"""Process skill — manage and inspect running processes."""

from __future__ import annotations

import asyncio
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class ProcessSkill(Skill):
    @property
    def name(self) -> str:
        return "process"

    @property
    def description(self) -> str:
        return "Inspect and manage running processes (list, find, signal)"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="process_find",
                description="Find processes by name (like pgrep)",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Process name to search for"},
                    },
                    "required": ["name"],
                },
            ),
            ToolDefinition(
                name="process_info",
                description="Get detailed info about a process by PID",
                parameters={
                    "type": "object",
                    "properties": {
                        "pid": {"type": "integer", "description": "Process ID"},
                    },
                    "required": ["pid"],
                },
            ),
            ToolDefinition(
                name="process_tree",
                description="Show process tree (parent/child relationships)",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="process_ports",
                description="Show which processes are listening on network ports",
                parameters={"type": "object", "properties": {}},
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "process_find":
                return await self._find(arguments["name"])
            case "process_info":
                return await self._info(arguments["pid"])
            case "process_tree":
                return await self._tree()
            case "process_ports":
                return await self._ports()
            case _:
                return f"Unknown tool: {tool_name}"

    async def _run(self, cmd: list[str]) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            return stdout.decode("utf-8", errors="replace").strip()
        except Exception as e:
            return f"(error: {e})"

    async def _find(self, name: str) -> str:
        import platform

        if platform.system() == "Darwin":
            output = await self._run(["pgrep", "-fl", name])
        else:
            output = await self._run(["pgrep", "-a", name])

        if not output:
            return f"No processes matching '{name}'"
        lines = output.splitlines()
        return f"Found {len(lines)} process(es) matching '{name}':\n" + "\n".join(
            f"  {line}" for line in lines[:20]
        )

    async def _info(self, pid: int) -> str:
        import platform

        if platform.system() == "Darwin":
            output = await self._run(
                ["ps", "-p", str(pid), "-o", "pid,ppid,%cpu,%mem,rss,start,command"]
            )
        else:
            output = await self._run(
                ["ps", "-p", str(pid), "-o", "pid,ppid,%cpu,%mem,rss,lstart,cmd"]
            )

        if not output or len(output.splitlines()) < 2:
            return f"Process {pid} not found"
        return output

    async def _tree(self) -> str:
        import platform

        if platform.system() == "Darwin":
            output = await self._run(["pstree", "-w"])
            if "(error:" in output:
                output = await self._run(["ps", "-axo", "pid,ppid,command"])
        else:
            output = await self._run(["pstree", "-p"])
            if "(error:" in output:
                output = await self._run(["ps", "auxf"])
        return output[:5000] if output else "Could not get process tree"

    async def _ports(self) -> str:
        import platform

        if platform.system() == "Darwin":
            output = await self._run(["lsof", "-iTCP", "-sTCP:LISTEN", "-P", "-n"])
        else:
            output = await self._run(["ss", "-tlnp"])
        return output[:5000] if output else "No listening ports found"

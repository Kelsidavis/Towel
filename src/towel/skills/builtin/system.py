"""System monitoring skill — inspect CPU, memory, disk, and processes."""

from __future__ import annotations

import asyncio
import platform
import os
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class SystemSkill(Skill):
    @property
    def name(self) -> str:
        return "system"

    @property
    def description(self) -> str:
        return "Query system information: CPU, memory, disk usage, and processes"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="system_info",
                description="Get system overview: OS, CPU, memory, disk usage, uptime",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="system_processes",
                description="List top processes by CPU or memory usage",
                parameters={
                    "type": "object",
                    "properties": {
                        "sort_by": {
                            "type": "string",
                            "enum": ["cpu", "memory"],
                            "description": "Sort by CPU or memory usage (default: cpu)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of processes to return (default: 10)",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="system_disk",
                description="Show disk usage for all mounted volumes",
                parameters={"type": "object", "properties": {}},
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "system_info":
                return await self._info()
            case "system_processes":
                sort_by = arguments.get("sort_by", "cpu")
                limit = arguments.get("limit", 10)
                return await self._processes(sort_by, limit)
            case "system_disk":
                return await self._disk()
            case _:
                return f"Unknown tool: {tool_name}"

    async def _run(self, cmd: list[str], timeout: float = 10) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return stdout.decode("utf-8", errors="replace").strip()
        except Exception as e:
            return f"(error: {e})"

    async def _info(self) -> str:
        lines: list[str] = []

        # OS info
        uname = platform.uname()
        lines.append(f"System: {uname.system} {uname.release} ({uname.machine})")
        lines.append(f"Node: {uname.node}")
        lines.append(f"Python: {platform.python_version()}")

        if platform.system() == "Darwin":
            # macOS-specific
            model = await self._run(["sysctl", "-n", "machdep.cpu.brand_string"])
            cores = await self._run(["sysctl", "-n", "hw.ncpu"])
            memsize = await self._run(["sysctl", "-n", "hw.memsize"])
            uptime = await self._run(["uptime"])

            lines.append(f"CPU: {model}")
            lines.append(f"Cores: {cores}")
            try:
                mem_gb = int(memsize) / (1024 ** 3)
                lines.append(f"Total memory: {mem_gb:.1f} GB")
            except (ValueError, TypeError):
                lines.append(f"Total memory: {memsize}")

            # Memory pressure
            vm_stat = await self._run(["vm_stat"])
            if vm_stat and "Pages free" in vm_stat:
                for line in vm_stat.splitlines():
                    line = line.strip().rstrip(".")
                    if "Pages free" in line:
                        try:
                            pages = int(line.split(":")[1].strip())
                            free_mb = pages * 16384 / (1024 * 1024)
                            lines.append(f"Free memory: ~{free_mb:.0f} MB")
                        except (ValueError, IndexError):
                            pass

            lines.append(f"Uptime: {uptime}")
        else:
            # Linux
            cores = os.cpu_count() or "unknown"
            lines.append(f"CPU cores: {cores}")

            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith(("MemTotal:", "MemAvailable:")):
                            lines.append(line.strip())
            except OSError:
                pass

            uptime = await self._run(["uptime", "-p"])
            lines.append(f"Uptime: {uptime}")

        # Load average
        try:
            load = os.getloadavg()
            lines.append(f"Load average: {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}")
        except OSError:
            pass

        return "\n".join(lines)

    async def _processes(self, sort_by: str = "cpu", limit: int = 10) -> str:
        limit = max(1, min(limit, 50))

        if platform.system() == "Darwin":
            if sort_by == "memory":
                cmd = ["ps", "aux", "-m"]
            else:
                cmd = ["ps", "aux", "-r"]
        else:
            if sort_by == "memory":
                cmd = ["ps", "aux", "--sort=-%mem"]
            else:
                cmd = ["ps", "aux", "--sort=-%cpu"]

        output = await self._run(cmd)
        if not output:
            return "(no process data)"

        lines = output.splitlines()
        # Header + top N processes
        result = lines[:1] + lines[1 : 1 + limit]
        return "\n".join(result)

    async def _disk(self) -> str:
        if platform.system() == "Darwin":
            output = await self._run(["df", "-h", "-l"])
        else:
            output = await self._run(["df", "-h", "--local"])
        return output or "(no disk data)"

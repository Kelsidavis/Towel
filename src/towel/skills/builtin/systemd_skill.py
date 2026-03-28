"""Systemd skill — manage services on Linux."""
from __future__ import annotations
import asyncio
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class SystemdSkill(Skill):
    @property
    def name(self) -> str: return "systemd"
    @property
    def description(self) -> str: return "Manage systemd services (Linux)"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="service_status", description="Check status of a systemd service",
                parameters={"type":"object","properties":{"service":{"type":"string"}},"required":["service"]}),
            ToolDefinition(name="service_list", description="List active systemd services",
                parameters={"type":"object","properties":{"filter":{"type":"string","description":"Filter by name"}}}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "service_status":
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "systemctl", "status", arguments["service"], "--no-pager",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                    return stdout.decode("utf-8","replace").strip()[:3000]
                except FileNotFoundError: return "systemctl not found (not Linux?)"
                except Exception as e: return f"Error: {e}"
            case "service_list":
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "systemctl", "list-units", "--type=service", "--state=active", "--no-pager",
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                    out = stdout.decode("utf-8","replace").strip()
                    f = arguments.get("filter")
                    if f: out = "\n".join(l for l in out.splitlines() if f.lower() in l.lower())
                    return out[:5000]
                except FileNotFoundError: return "systemctl not found"
                except Exception as e: return f"Error: {e}"
            case _: return f"Unknown: {tool_name}"

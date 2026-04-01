"""Port scanner skill — scan common ports on a host."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

from towel.skills.base import Skill, ToolDefinition

COMMON_PORTS = {
    22: "SSH",
    80: "HTTP",
    443: "HTTPS",
    3000: "Dev",
    3306: "MySQL",
    5432: "PostgreSQL",
    6379: "Redis",
    8080: "HTTP-Alt",
    8443: "HTTPS-Alt",
    27017: "MongoDB",
}


class PortScannerSkill(Skill):
    @property
    def name(self) -> str:
        return "port_scan"

    @property
    def description(self) -> str:
        return "Scan common ports on a host"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="scan_ports",
                description="Scan common ports on a host",
                parameters={
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "ports": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Custom ports (default: common ports)",
                        },
                    },
                    "required": ["host"],
                },
            )
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "scan_ports":
            return f"Unknown: {tool_name}"
        host = arguments["host"]
        ports = arguments.get("ports", list(COMMON_PORTS.keys()))
        results = []

        async def check(port):
            loop = asyncio.get_event_loop()
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                r = await loop.run_in_executor(None, lambda: s.connect_ex((host, port)))
                s.close()
                svc = COMMON_PORTS.get(port, "")
                return (port, r == 0, svc)
            except Exception:
                return (port, False, "")

        tasks = [check(p) for p in ports[:50]]
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)
        results.sort(key=lambda x: x[0])
        open_ports = [(p, s) for p, o, s in results if o]
        closed = len(results) - len(open_ports)
        lines = [f"Scan: {host} ({len(open_ports)} open, {closed} closed)"]
        for p, svc in open_ports:
            lines.append(f"  {p:>5}/tcp  OPEN  {svc}")
        return "\n".join(lines)

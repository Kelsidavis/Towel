"""DNS skill — advanced DNS lookups with record types."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class DnsSkill(Skill):
    @property
    def name(self) -> str:
        return "dns"

    @property
    def description(self) -> str:
        return "Advanced DNS lookups — A, AAAA, MX, NS, TXT, reverse"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="dns_resolve",
                description="Resolve a domain to all IP addresses (A + AAAA)",
                parameters={
                    "type": "object",
                    "properties": {"domain": {"type": "string"}},
                    "required": ["domain"],
                },
            ),
            ToolDefinition(
                name="dns_mx",
                description="Get MX (mail) records for a domain",
                parameters={
                    "type": "object",
                    "properties": {"domain": {"type": "string"}},
                    "required": ["domain"],
                },
            ),
            ToolDefinition(
                name="dns_reverse",
                description="Reverse DNS lookup (IP to hostname)",
                parameters={
                    "type": "object",
                    "properties": {"ip": {"type": "string"}},
                    "required": ["ip"],
                },
            ),
            ToolDefinition(
                name="dns_records",
                description="Get DNS records using dig (A, AAAA, MX, NS, TXT, CNAME)",
                parameters={
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string"},
                        "record_type": {
                            "type": "string",
                            "enum": ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA", "ANY"],
                        },
                    },
                    "required": ["domain", "record_type"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "dns_resolve":
                return await self._resolve(arguments["domain"])
            case "dns_mx":
                return await self._dig(arguments["domain"], "MX")
            case "dns_reverse":
                return await self._reverse(arguments["ip"])
            case "dns_records":
                return await self._dig(arguments["domain"], arguments["record_type"])
            case _:
                return f"Unknown: {tool_name}"

    async def _resolve(self, domain: str) -> str:
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: socket.getaddrinfo(domain, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            )
            seen, lines = set(), [f"DNS for {domain}:"]
            for fam, _, _, _, addr in results:
                ip = addr[0]
                if ip in seen:
                    continue
                seen.add(ip)
                label = "AAAA" if fam == socket.AF_INET6 else "A"
                lines.append(f"  {label}: {ip}")
            return "\n".join(lines) if len(lines) > 1 else f"No records for {domain}"
        except socket.gaierror as e:
            return f"DNS failed: {e}"

    async def _reverse(self, ip: str) -> str:
        loop = asyncio.get_event_loop()
        try:
            hostname = await loop.run_in_executor(None, lambda: socket.gethostbyaddr(ip))
            return f"{ip} → {hostname[0]}"
        except socket.herror:
            return f"No reverse DNS for {ip}"
        except Exception as e:
            return f"Error: {e}"

    async def _dig(self, domain: str, rtype: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "dig",
                "+short",
                domain,
                rtype,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            out = stdout.decode().strip()
            return (
                f"{rtype} records for {domain}:\n{out}"
                if out
                else f"No {rtype} records for {domain}"
            )
        except FileNotFoundError:
            return "dig not found (install bind-utils)"
        except Exception as e:
            return f"Error: {e}"

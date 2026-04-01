"""Whois skill — domain registration and expiry info."""
from __future__ import annotations

import asyncio
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class WhoisSkill(Skill):
    @property
    def name(self) -> str: return "whois"
    @property
    def description(self) -> str: return "Domain WHOIS lookup — registrar, expiry, nameservers"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="whois_lookup", description="Get WHOIS info for a domain",
                parameters={"type":"object","properties":{
                    "domain":{"type":"string","description":"Domain name (e.g., example.com)"},
                },"required":["domain"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "whois_lookup": return f"Unknown: {tool_name}"
        domain = arguments["domain"]
        try:
            proc = await asyncio.create_subprocess_exec(
                "whois", domain, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
            raw = stdout.decode("utf-8", "replace")
            # Extract key fields
            import re
            fields = {}
            for pattern, key in [
                (r"Registrar:\s*(.+)", "Registrar"),
                (r"Creation Date:\s*(.+)", "Created"),
                (r"Registry Expiry Date:\s*(.+)", "Expires"),
                (r"Updated Date:\s*(.+)", "Updated"),
                (r"Name Server:\s*(.+)", "NS"),
                (r"Registrant Organization:\s*(.+)", "Organization"),
                (r"Registrant Country:\s*(.+)", "Country"),
            ]:
                m = re.search(pattern, raw, re.IGNORECASE)
                if m: fields[key] = m.group(1).strip()
            # Get all nameservers
            nservers = re.findall(r"Name Server:\s*(.+)", raw, re.IGNORECASE)
            if nservers: fields["NS"] = ", ".join(s.strip().lower() for s in nservers[:4])

            if not fields: return raw[:2000]
            lines = [f"WHOIS: {domain}"]
            for k, v in fields.items(): lines.append(f"  {k}: {v}")
            return "\n".join(lines)
        except FileNotFoundError: return "whois not found"
        except Exception as e: return f"Error: {e}"

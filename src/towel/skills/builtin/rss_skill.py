"""RSS skill — read and parse RSS/Atom feeds."""
from __future__ import annotations

import re
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class RssSkill(Skill):
    @property
    def name(self) -> str: return "rss"
    @property
    def description(self) -> str: return "Read and parse RSS/Atom feeds"
    def tools(self) -> list[ToolDefinition]:
        return [ToolDefinition(name="rss_read", description="Fetch and parse an RSS feed",
            parameters={"type":"object","properties":{"url":{"type":"string"},"limit":{"type":"integer","description":"Max items (default: 10)"}},"required":["url"]})]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "rss_read": return f"Unknown: {tool_name}"
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
                resp = await c.get(arguments["url"], headers={"User-Agent":"Towel/1.0"})
                xml = resp.text
            titles = re.findall(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", xml)
            links = re.findall(r"<link>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>", xml)
            limit = arguments.get("limit", 10)
            lines = [f"Feed ({min(len(titles)-1, limit)} items):"]
            for i, (t, l) in enumerate(zip(titles[1:limit+1], links[1:limit+1])):
                lines.append(f"  {i+1}. {t}\n     {l}")
            return "\n".join(lines) if len(lines) > 1 else "No items found."
        except Exception as e: return f"RSS error: {e}"

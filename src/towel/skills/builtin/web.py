"""Web fetch skill — retrieve URLs."""

from __future__ import annotations

from typing import Any

import httpx

from towel.skills.base import Skill, ToolDefinition

MAX_RESPONSE_BYTES = 100_000


class WebFetchSkill(Skill):
    @property
    def name(self) -> str:
        return "web"

    @property
    def description(self) -> str:
        return "Fetch content from URLs"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="fetch_url",
                description="Fetch the content of a URL and return the response body as text",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch"},
                        "method": {"type": "string", "description": "HTTP method (default GET)"},
                    },
                    "required": ["url"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "fetch_url":
            return f"Unknown tool: {tool_name}"

        url = arguments["url"]
        method = arguments.get("method", "GET").upper()

        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            try:
                resp = await client.request(method, url)
                body = resp.text[:MAX_RESPONSE_BYTES]
                return f"[{resp.status_code}] {body}"
            except httpx.HTTPError as e:
                return f"HTTP error: {e}"

"""HTTP skill — make API requests with full control over method, headers, body."""

from __future__ import annotations

import json
from typing import Any

from towel.skills.base import Skill, ToolDefinition

MAX_RESPONSE = 50_000


class HttpSkill(Skill):
    @property
    def name(self) -> str:
        return "http"

    @property
    def description(self) -> str:
        return "Make HTTP requests with full control (GET, POST, PUT, DELETE, headers, body)"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="http_request",
                description="Make an HTTP request. Returns status, headers, and body.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to request"},
                        "method": {"type": "string", "description": "HTTP method (default: GET)"},
                        "headers": {
                            "type": "object",
                            "description": "Request headers as key-value pairs",
                        },
                        "body": {"type": "string", "description": "Request body (for POST/PUT/PATCH)"},
                        "json_body": {
                            "type": "object",
                            "description": "JSON body (auto-sets Content-Type)",
                        },
                        "timeout": {"type": "number", "description": "Timeout in seconds (default: 10)"},
                    },
                    "required": ["url"],
                },
            ),
            ToolDefinition(
                name="http_head",
                description="Send a HEAD request — get headers without downloading body",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to check"},
                    },
                    "required": ["url"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "http_request":
                return await self._request(
                    arguments["url"],
                    arguments.get("method", "GET"),
                    arguments.get("headers"),
                    arguments.get("body"),
                    arguments.get("json_body"),
                    arguments.get("timeout", 10),
                )
            case "http_head":
                return await self._head(arguments["url"])
            case _:
                return f"Unknown tool: {tool_name}"

    async def _request(
        self, url: str, method: str, headers: dict | None,
        body: str | None, json_body: dict | None, timeout: float,
    ) -> str:
        import httpx

        method = method.upper()
        req_headers = dict(headers) if headers else {}
        req_headers.setdefault("User-Agent", "Towel/1.0")

        kwargs: dict[str, Any] = {"headers": req_headers, "timeout": timeout, "follow_redirects": True}

        if json_body is not None:
            kwargs["json"] = json_body
        elif body is not None:
            kwargs["content"] = body

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.request(method, url, **kwargs)

            lines = [
                f"HTTP {resp.status_code} {resp.reason_phrase}",
                f"URL: {resp.url}",
            ]

            # Key headers
            for h in ["content-type", "content-length", "server", "location",
                       "x-request-id", "x-ratelimit-remaining"]:
                val = resp.headers.get(h)
                if val:
                    lines.append(f"{h}: {val}")

            # Body
            ct = resp.headers.get("content-type", "")
            body_text = resp.text[:MAX_RESPONSE]

            if "json" in ct:
                try:
                    pretty = json.dumps(json.loads(body_text), indent=2, ensure_ascii=False)
                    lines.append(f"\nBody ({len(resp.content)} bytes):\n{pretty[:MAX_RESPONSE]}")
                except json.JSONDecodeError:
                    lines.append(f"\nBody:\n{body_text}")
            elif body_text.strip():
                lines.append(f"\nBody ({len(resp.content)} bytes):\n{body_text}")

            return "\n".join(lines)

        except httpx.TimeoutException:
            return f"Timeout after {timeout}s: {url}"
        except httpx.ConnectError as e:
            return f"Connection failed: {e}"
        except Exception as e:
            return f"Error: {e}"

    async def _head(self, url: str) -> str:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                resp = await client.head(url, headers={"User-Agent": "Towel/1.0"})

            lines = [f"HTTP {resp.status_code} {resp.reason_phrase}", f"URL: {resp.url}"]
            for k, v in resp.headers.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        except Exception as e:
            return f"Error: {e}"

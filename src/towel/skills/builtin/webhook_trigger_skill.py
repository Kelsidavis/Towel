"""Webhook trigger skill — send outbound webhooks to external services."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class WebhookTriggerSkill(Skill):
    @property
    def name(self) -> str: return "webhook_trigger"
    @property
    def description(self) -> str: return "Send outbound webhooks to Slack, Discord, or any URL"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="webhook_send", description="Send a POST to a webhook URL with JSON payload",
                parameters={"type":"object","properties":{
                    "url":{"type":"string","description":"Webhook URL"},
                    "payload":{"type":"object","description":"JSON payload to send"},
                    "headers":{"type":"object","description":"Extra headers (optional)"},
                },"required":["url","payload"]}),
            ToolDefinition(name="slack_message", description="Send a message to a Slack incoming webhook",
                parameters={"type":"object","properties":{
                    "webhook_url":{"type":"string","description":"Slack webhook URL"},
                    "text":{"type":"string","description":"Message text"},
                    "channel":{"type":"string","description":"Channel override (optional)"},
                    "username":{"type":"string","description":"Bot name (optional)"},
                },"required":["webhook_url","text"]}),
            ToolDefinition(name="discord_message", description="Send a message to a Discord webhook",
                parameters={"type":"object","properties":{
                    "webhook_url":{"type":"string","description":"Discord webhook URL"},
                    "content":{"type":"string","description":"Message content"},
                    "username":{"type":"string","description":"Bot name (optional)"},
                },"required":["webhook_url","content"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "webhook_send": return await self._send(arguments["url"], arguments["payload"], arguments.get("headers"))
            case "slack_message": return await self._slack(arguments["webhook_url"], arguments["text"], arguments.get("channel"), arguments.get("username"))
            case "discord_message": return await self._discord(arguments["webhook_url"], arguments["content"], arguments.get("username"))
            case _: return f"Unknown tool: {tool_name}"

    async def _send(self, url: str, payload: dict, headers: dict|None) -> str:
        import httpx
        h = {"Content-Type": "application/json", "User-Agent": "Towel/1.0"}
        if headers: h.update(headers)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload, headers=h)
            return f"Sent to {url[:50]}: HTTP {resp.status_code} ({len(resp.content)} bytes)"
        except Exception as e: return f"Failed: {e}"

    async def _slack(self, url: str, text: str, channel: str|None, username: str|None) -> str:
        payload: dict[str, Any] = {"text": text}
        if channel: payload["channel"] = channel
        if username: payload["username"] = username
        return await self._send(url, payload, None)

    async def _discord(self, url: str, content: str, username: str|None) -> str:
        payload: dict[str, Any] = {"content": content}
        if username: payload["username"] = username
        return await self._send(url, payload, None)

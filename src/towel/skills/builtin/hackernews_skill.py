"""Hacker News skill — browse top stories."""
from __future__ import annotations
import json
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class HackerNewsSkill(Skill):
    @property
    def name(self) -> str: return "hackernews"
    @property
    def description(self) -> str: return "Browse Hacker News top stories"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="hn_top", description="Get top Hacker News stories",
                parameters={"type":"object","properties":{
                    "limit":{"type":"integer","description":"Number of stories (default: 10)"},
                }}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "hn_top": return f"Unknown: {tool_name}"
        import httpx
        limit = min(arguments.get("limit", 10), 30)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
                ids = resp.json()[:limit]
                stories = []
                for sid in ids:
                    r = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
                    s = r.json()
                    stories.append(f"  [{s.get('score',0)}] {s.get('title','?')}\n       {s.get('url','(no url)')}")
                return f"Top {len(stories)} Hacker News stories:\n\n" + "\n\n".join(stories)
        except Exception as e: return f"HN error: {e}"

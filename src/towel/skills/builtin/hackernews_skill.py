"""Hacker News skill — browse top stories."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class HackerNewsSkill(Skill):
    @property
    def name(self) -> str:
        return "hackernews"

    @property
    def description(self) -> str:
        return "Browse Hacker News top stories"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="hn_top",
                description="Get top Hacker News stories",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of stories (default: 10)",
                        },
                    },
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "hn_top":
            return f"Unknown: {tool_name}"
        import httpx

        limit = min(arguments.get("limit", 10), 30)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://hacker-news.firebaseio.com/v0/topstories.json")
                resp.raise_for_status()
                all_ids = resp.json()
                if not isinstance(all_ids, list):
                    return "Unexpected HN API response"
                ids = all_ids[:limit]
                stories = []
                for sid in ids:
                    r = await client.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
                    r.raise_for_status()
                    s = r.json() or {}
                    stories.append(
                        f"  [{s.get('score', 0)}] "
                        f"{s.get('title', '?')}\n"
                        f"       {s.get('url', '(no url)')}"
                    )
                return f"Top {len(stories)} Hacker News stories:\n\n" + "\n\n".join(stories)
        except Exception as e:
            return f"HN error: {e}"

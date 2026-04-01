"""Stack Overflow skill — search questions and get answers."""
from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class StackOverflowSkill(Skill):
    @property
    def name(self) -> str: return "stackoverflow"
    @property
    def description(self) -> str: return "Search Stack Overflow questions and answers"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="so_search", description="Search Stack Overflow for questions",
                parameters={"type":"object","properties":{
                    "query":{"type":"string","description":"Search query"},
                    "tagged":{"type":"string","description":"Filter by tag (e.g., python, javascript)"},
                    "limit":{"type":"integer","description":"Max results (default: 5)"},
                },"required":["query"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "so_search": return f"Unknown: {tool_name}"
        import httpx
        params: dict[str, Any] = {
            "order": "desc", "sort": "relevance", "intitle": arguments["query"],
            "site": "stackoverflow", "pagesize": min(arguments.get("limit", 5), 10),
            "filter": "!nNPvSNdWme",
        }
        if arguments.get("tagged"): params["tagged"] = arguments["tagged"]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get("https://api.stackexchange.com/2.3/search/advanced", params=params)
                data = resp.json()
                items = data.get("items", [])
                if not items: return "No questions found."
                lines = ["Stack Overflow results:"]
                for q in items:
                    score = q.get("score", 0)
                    answers = q.get("answer_count", 0)
                    accepted = " ✓" if q.get("is_answered") else ""
                    title = q.get("title", "?")
                    tags = " ".join(q.get("tags", [])[:3])
                    link = q.get("link", "")
                    lines.append(f"\n  [{score}↑ {answers}a{accepted}] {title}")
                    lines.append(f"    {tags}")
                    lines.append(f"    {link}")
                return "\n".join(lines)
        except Exception as e: return f"SO error: {e}"

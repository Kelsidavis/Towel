"""Wikipedia skill — search and summarize articles."""
from __future__ import annotations
import json
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class WikipediaSkill(Skill):
    @property
    def name(self) -> str: return "wikipedia"
    @property
    def description(self) -> str: return "Search and get summaries from Wikipedia"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="wiki_search", description="Search Wikipedia for articles",
                parameters={"type":"object","properties":{
                    "query":{"type":"string","description":"Search query"},
                    "limit":{"type":"integer","description":"Max results (default: 5)"},
                },"required":["query"]}),
            ToolDefinition(name="wiki_summary", description="Get the summary of a Wikipedia article",
                parameters={"type":"object","properties":{
                    "title":{"type":"string","description":"Article title"},
                },"required":["title"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                if tool_name == "wiki_search":
                    resp = await client.get("https://en.wikipedia.org/w/api.php", params={
                        "action":"opensearch","search":arguments["query"],
                        "limit":arguments.get("limit",5),"format":"json"})
                    data = resp.json()
                    if len(data) >= 2 and data[1]:
                        return "\n".join(f"  {t}" for t in data[1])
                    return "No results."
                elif tool_name == "wiki_summary":
                    title = arguments["title"].replace(" ", "_")
                    resp = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
                    data = resp.json()
                    return f"{data.get('title','?')}\n\n{data.get('extract','No summary available.')}"
        except Exception as e: return f"Wikipedia error: {e}"
        return f"Unknown tool: {tool_name}"

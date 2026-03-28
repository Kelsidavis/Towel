"""Reddit skill — browse posts from any subreddit (no auth, uses .json API)."""
from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class RedditSkill(Skill):
    @property
    def name(self) -> str: return "reddit"
    @property
    def description(self) -> str: return "Browse Reddit posts and comments from any subreddit"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="reddit_hot", description="Get hot posts from a subreddit",
                parameters={"type":"object","properties":{
                    "subreddit":{"type":"string","description":"Subreddit name (without r/)"},
                    "limit":{"type":"integer","description":"Number of posts (default: 10)"},
                },"required":["subreddit"]}),
            ToolDefinition(name="reddit_search", description="Search Reddit for posts",
                parameters={"type":"object","properties":{
                    "query":{"type":"string","description":"Search query"},
                    "subreddit":{"type":"string","description":"Limit to subreddit (optional)"},
                    "limit":{"type":"integer","description":"Max results (default: 10)"},
                },"required":["query"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        import httpx
        headers = {"User-Agent": "Towel/1.0"}
        try:
            async with httpx.AsyncClient(timeout=10, headers=headers, follow_redirects=True) as client:
                if tool_name == "reddit_hot":
                    sub = arguments["subreddit"]
                    limit = min(arguments.get("limit", 10), 25)
                    resp = await client.get(f"https://www.reddit.com/r/{sub}/hot.json?limit={limit}")
                    return self._format_listing(resp.json(), sub)
                elif tool_name == "reddit_search":
                    q = arguments["query"]
                    sub = arguments.get("subreddit")
                    limit = min(arguments.get("limit", 10), 25)
                    url = f"https://www.reddit.com/r/{sub}/search.json" if sub else "https://www.reddit.com/search.json"
                    params = {"q": q, "limit": limit, "restrict_sr": "on" if sub else "off", "sort": "relevance"}
                    resp = await client.get(url, params=params)
                    return self._format_listing(resp.json(), sub or "all")
        except Exception as e: return f"Reddit error: {e}"
        return f"Unknown tool: {tool_name}"

    def _format_listing(self, data: dict, sub: str) -> str:
        posts = data.get("data", {}).get("children", [])
        if not posts: return f"No posts found in r/{sub}"
        lines = [f"r/{sub} ({len(posts)} posts):"]
        for p in posts:
            d = p.get("data", {})
            score = d.get("score", 0)
            title = d.get("title", "?")[:80]
            comments = d.get("num_comments", 0)
            url = d.get("url", "")[:60]
            lines.append(f"\n  [{score:>5}] {title}")
            lines.append(f"         {comments} comments · {url}")
        return "\n".join(lines)

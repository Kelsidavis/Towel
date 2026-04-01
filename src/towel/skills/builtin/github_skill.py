"""GitHub skill — search repos, view profiles, trending."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class GithubSkill(Skill):
    @property
    def name(self) -> str:
        return "github"

    @property
    def description(self) -> str:
        return "Search GitHub repos, view profiles, and get repo info"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="github_repo",
                description="Get info about a GitHub repository",
                parameters={
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "owner/repo (e.g., torvalds/linux)",
                        },
                    },
                    "required": ["repo"],
                },
            ),
            ToolDefinition(
                name="github_search",
                description="Search GitHub repositories",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results (default: 5)"},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="github_user",
                description="Get GitHub user/org profile info",
                parameters={
                    "type": "object",
                    "properties": {
                        "username": {"type": "string", "description": "GitHub username"},
                    },
                    "required": ["username"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        import httpx

        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Towel/1.0"}
        try:
            async with httpx.AsyncClient(timeout=10, headers=headers) as client:
                if tool_name == "github_repo":
                    resp = await client.get(f"https://api.github.com/repos/{arguments['repo']}")
                    if resp.status_code == 404:
                        return f"Repo not found: {arguments['repo']}"
                    r = resp.json()
                    return (
                        f"{r['full_name']} — {r.get('description', '')}\n"
                        f"  Stars: {r['stargazers_count']:,}  Forks: {r['forks_count']:,}\n"
                        f"  Language: {r.get('language', '?')}  "
                        f"License: {(r.get('license') or {}).get('spdx_id', '?')}\n"
                        f"  Created: {r['created_at'][:10]}  Updated: {r['updated_at'][:10]}\n"
                        f"  URL: {r['html_url']}"
                    )
                elif tool_name == "github_search":
                    resp = await client.get(
                        "https://api.github.com/search/repositories",
                        params={"q": arguments["query"], "per_page": arguments.get("limit", 5)},
                    )
                    items = resp.json().get("items", [])
                    if not items:
                        return "No repos found."
                    lines = []
                    for r in items:
                        lines.append(
                            f"  {r['full_name']} "
                            f"★{r['stargazers_count']:,} — "
                            f"{r.get('description', '')[:60]}"
                        )
                    return f"GitHub search: {arguments['query']}\n\n" + "\n".join(lines)
                elif tool_name == "github_user":
                    resp = await client.get(f"https://api.github.com/users/{arguments['username']}")
                    if resp.status_code == 404:
                        return f"User not found: {arguments['username']}"
                    u = resp.json()
                    return (
                        f"{u.get('name', u['login'])} (@{u['login']})\n"
                        f"  {u.get('bio', '')}\n"
                        f"  Repos: {u['public_repos']}  "
                        f"Followers: {u['followers']}  "
                        f"Following: {u['following']}\n"
                        f"  Location: {u.get('location', '?')}  Company: {u.get('company', '?')}\n"
                        f"  URL: {u['html_url']}"
                    )
        except Exception as e:
            return f"GitHub error: {e}"
        return f"Unknown tool: {tool_name}"

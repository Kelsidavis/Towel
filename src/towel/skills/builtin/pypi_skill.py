"""PyPI skill — search packages and get info."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class PypiSkill(Skill):
    @property
    def name(self) -> str:
        return "pypi"

    @property
    def description(self) -> str:
        return "Search PyPI packages and get version/dependency info"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="pypi_info",
                description="Get info about a PyPI package",
                parameters={
                    "type": "object",
                    "properties": {
                        "package": {"type": "string", "description": "Package name"},
                    },
                    "required": ["package"],
                },
            ),
            ToolDefinition(
                name="pypi_search",
                description="Search for packages on PyPI",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                if tool_name == "pypi_info":
                    pkg = arguments["package"]
                    resp = await client.get(f"https://pypi.org/pypi/{pkg}/json")
                    if resp.status_code == 404:
                        return f"Package not found: {pkg}"
                    info = resp.json()["info"]
                    return (
                        f"{info['name']} v{info['version']}\n"
                        f"  {info.get('summary', '')}\n"
                        f"  Author: {info.get('author', '?')}\n"
                        f"  License: {info.get('license', '?')}\n"
                        f"  Python: {info.get('requires_python', '?')}\n"
                        f"  Home: {info.get('home_page') or info.get('project_url', '?')}"
                    )
                elif tool_name == "pypi_search":
                    # PyPI doesn't have a search API anymore, use warehouse
                    resp = await client.get(
                        "https://pypi.org/search/",
                        params={"q": arguments["query"]},
                        headers={"Accept": "text/html"},
                        follow_redirects=True,
                    )
                    # Parse simple results from HTML
                    import re

                    packages = re.findall(
                        r'class="package-snippet__name">([^<]+)</span>\s*<span[^>]*>([^<]+)</span>',
                        resp.text,
                    )
                    if not packages:
                        return "No packages found (PyPI search is limited)."
                    lines = [f"PyPI search: {arguments['query']}"]
                    for name, version in packages[:10]:
                        lines.append(f"  {name.strip()} {version.strip()}")
                    return "\n".join(lines)
        except Exception as e:
            return f"PyPI error: {e}"
        return f"Unknown tool: {tool_name}"

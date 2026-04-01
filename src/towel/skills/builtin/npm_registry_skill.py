"""NPM registry skill — search and inspect npm packages."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class NpmRegistrySkill(Skill):
    @property
    def name(self) -> str:
        return "npm_registry"

    @property
    def description(self) -> str:
        return "Search npm registry and get package info"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="npm_search",
                description="Search npm for packages",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="npm_pkg_info",
                description="Get info about an npm package",
                parameters={
                    "type": "object",
                    "properties": {"package": {"type": "string"}},
                    "required": ["package"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as c:
                if tool_name == "npm_search":
                    resp = await c.get(
                        "https://registry.npmjs.org/-/v1/search",
                        params={"text": arguments["query"], "size": arguments.get("limit", 5)},
                    )
                    pkgs = resp.json().get("objects", [])
                    if not pkgs:
                        return "No packages found."
                    lines = ["npm search results:"]
                    for p in pkgs:
                        pkg = p.get("package", {})
                        lines.append(
                            f"  {pkg.get('name', '?')} "
                            f"v{pkg.get('version', '?')} — "
                            f"{pkg.get('description', '')[:60]}"
                        )
                    return "\n".join(lines)
                elif tool_name == "npm_pkg_info":
                    resp = await c.get(f"https://registry.npmjs.org/{arguments['package']}/latest")
                    if resp.status_code == 404:
                        return f"Not found: {arguments['package']}"
                    d = resp.json()
                    return (
                        f"{d.get('name', '?')} v{d.get('version', '?')}\n"
                        f"  {d.get('description', '')}\n"
                        f"  License: {d.get('license', '?')}\n"
                        f"  Deps: {len(d.get('dependencies', {}))}\n"
                        f"  Homepage: {d.get('homepage', '?')}"
                    )
        except Exception as e:
            return f"npm error: {e}"
        return f"Unknown: {tool_name}"

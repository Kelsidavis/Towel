"""URL skill — parse, build, shorten, and inspect URLs."""
from __future__ import annotations
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class UrlSkill(Skill):
    @property
    def name(self) -> str: return "url"
    @property
    def description(self) -> str: return "Parse, build, and inspect URLs"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="url_parse", description="Parse a URL into components",
                parameters={"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}),
            ToolDefinition(name="url_build", description="Build a URL from components",
                parameters={"type":"object","properties":{
                    "scheme":{"type":"string"},"host":{"type":"string"},
                    "path":{"type":"string"},"params":{"type":"object","description":"Query parameters"},
                },"required":["host"]}),
            ToolDefinition(name="url_extract_params", description="Extract query parameters from a URL",
                parameters={"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "url_parse":
                p = urlparse(arguments["url"])
                return (f"URL: {arguments['url']}\n  Scheme: {p.scheme}\n  Host: {p.hostname}\n"
                        f"  Port: {p.port or 'default'}\n  Path: {p.path}\n  Query: {p.query}\n  Fragment: {p.fragment}")
            case "url_build":
                q = urlencode(arguments.get("params", {}))
                return urlunparse((arguments.get("scheme","https"), arguments["host"],
                                   arguments.get("path","/"), "", q, ""))
            case "url_extract_params":
                p = urlparse(arguments["url"])
                params = parse_qs(p.query)
                if not params: return "No query parameters."
                return "\n".join(f"  {k}: {', '.join(v)}" for k, v in params.items())
            case _: return f"Unknown: {tool_name}"

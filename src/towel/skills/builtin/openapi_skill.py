"""OpenAPI skill — parse and explore API specifications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class OpenApiSkill(Skill):
    @property
    def name(self) -> str:
        return "openapi"

    @property
    def description(self) -> str:
        return "Parse and explore OpenAPI/Swagger specifications"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="openapi_summary",
                description="Show API overview — title, version, endpoints count",
                parameters={
                    "type": "object",
                    "properties": {
                        "spec": {
                            "type": "string",
                            "description": "Path to OpenAPI JSON/YAML file or raw spec text",
                        },
                    },
                    "required": ["spec"],
                },
            ),
            ToolDefinition(
                name="openapi_endpoints",
                description="List all API endpoints with methods and descriptions",
                parameters={
                    "type": "object",
                    "properties": {
                        "spec": {"type": "string", "description": "Path or raw spec"},
                        "tag": {"type": "string", "description": "Filter by tag (optional)"},
                    },
                    "required": ["spec"],
                },
            ),
            ToolDefinition(
                name="openapi_detail",
                description="Show detailed info about a specific endpoint",
                parameters={
                    "type": "object",
                    "properties": {
                        "spec": {"type": "string", "description": "Path or raw spec"},
                        "path": {"type": "string", "description": "API path (e.g., /users/{id})"},
                        "method": {"type": "string", "description": "HTTP method (default: get)"},
                    },
                    "required": ["spec", "path"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "openapi_summary":
                return self._summary(self._load(arguments["spec"]))
            case "openapi_endpoints":
                return self._endpoints(self._load(arguments["spec"]), arguments.get("tag"))
            case "openapi_detail":
                return self._detail(
                    self._load(arguments["spec"]), arguments["path"], arguments.get("method", "get")
                )
            case _:
                return f"Unknown tool: {tool_name}"

    def _load(self, spec: str) -> dict | str:
        p = Path(spec).expanduser()
        if p.is_file():
            text = p.read_text(encoding="utf-8")
        else:
            text = spec
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try YAML-like simple parse
            try:
                from towel.skills.builtin.yaml_skill import _simple_yaml_parse

                result = _simple_yaml_parse(text)
                if isinstance(result, dict):
                    return result
            except Exception:
                pass
            return "Cannot parse spec (not valid JSON)"

    def _summary(self, spec: dict | str) -> str:
        if isinstance(spec, str):
            return spec
        info = spec.get("info", {})
        paths = spec.get("paths", {})
        endpoints = sum(len(methods) for methods in paths.values())
        tags = set()
        for path_methods in paths.values():
            for op in path_methods.values():
                if isinstance(op, dict):
                    for t in op.get("tags", []):
                        tags.add(t)
        lines = [
            f"API: {info.get('title', '?')} v{info.get('version', '?')}",
            f"  {info.get('description', '')}",
            f"  Endpoints: {endpoints}",
            f"  Paths: {len(paths)}",
        ]
        if tags:
            lines.append(f"  Tags: {', '.join(sorted(tags))}")
        servers = spec.get("servers", [])
        if servers:
            lines.append(f"  Server: {servers[0].get('url', '?')}")
        return "\n".join(lines)

    def _endpoints(self, spec: dict | str, tag: str | None) -> str:
        if isinstance(spec, str):
            return spec
        paths = spec.get("paths", {})
        lines = []
        for path, methods in sorted(paths.items()):
            for method, op in methods.items():
                if not isinstance(op, dict):
                    continue
                if method.startswith("x-"):
                    continue
                if tag and tag not in op.get("tags", []):
                    continue
                summary = op.get("summary", op.get("description", ""))[:60]
                lines.append(f"  {method.upper():7s} {path:30s}  {summary}")
        if not lines:
            return "No endpoints found."
        return f"Endpoints ({len(lines)}):\n" + "\n".join(lines)

    def _detail(self, spec: dict | str, path: str, method: str) -> str:
        if isinstance(spec, str):
            return spec
        paths = spec.get("paths", {})
        if path not in paths:
            return f"Path not found: {path}"
        op = paths[path].get(method.lower())
        if not op:
            return f"Method {method.upper()} not found on {path}"
        lines = [f"{method.upper()} {path}"]
        if op.get("summary"):
            lines.append(f"  Summary: {op['summary']}")
        if op.get("description"):
            lines.append(f"  Description: {op['description'][:200]}")
        params = op.get("parameters", [])
        if params:
            lines.append(f"  Parameters ({len(params)}):")
            for p in params:
                req = " (required)" if p.get("required") else ""
                lines.append(
                    f"    {p.get('in', '?')}/{p.get('name', '?')}: "
                    f"{p.get('schema', {}).get('type', '?')}{req}"
                )
        responses = op.get("responses", {})
        if responses:
            lines.append("  Responses:")
            for code, resp in responses.items():
                desc = resp.get("description", "")[:50] if isinstance(resp, dict) else ""
                lines.append(f"    {code}: {desc}")
        return "\n".join(lines)

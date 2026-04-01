"""Environment skill — inspect and manage environment variables."""

from __future__ import annotations

import os
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class EnvSkill(Skill):
    @property
    def name(self) -> str:
        return "env"

    @property
    def description(self) -> str:
        return "Inspect environment variables and paths"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="env_get",
                description="Get the value of an environment variable",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Variable name (e.g., PATH, HOME, SHELL)",
                        },
                    },
                    "required": ["name"],
                },
            ),
            ToolDefinition(
                name="env_list",
                description="List all environment variables (or filter by prefix)",
                parameters={
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": "Only show variables starting with this prefix",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="env_path",
                description=(
                    "Show the PATH entries as a readable list, "
                    "highlighting which directories exist"
                ),
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="env_which",
                description="Find the full path of a command (like `which`)",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command name to find"},
                    },
                    "required": ["command"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "env_get":
                return self._get(arguments["name"])
            case "env_list":
                return self._list(arguments.get("prefix"))
            case "env_path":
                return self._path()
            case "env_which":
                return self._which(arguments["command"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _get(self, name: str) -> str:
        val = os.environ.get(name)
        if val is None:
            return f"{name} is not set"
        if len(val) > 2000:
            return f"{name}={val[:2000]}... (truncated, {len(val)} chars)"
        return f"{name}={val}"

    def _list(self, prefix: str | None) -> str:
        items = sorted(os.environ.items())
        if prefix:
            items = [(k, v) for k, v in items if k.upper().startswith(prefix.upper())]

        if not items:
            return f"No variables{' matching ' + prefix if prefix else ''}"

        # Redact sensitive-looking values
        sensitive = {"token", "secret", "key", "password", "credential", "auth"}
        lines = []
        for k, v in items:
            if any(s in k.lower() for s in sensitive):
                lines.append(f"  {k}=****")
            elif len(v) > 80:
                lines.append(f"  {k}={v[:77]}...")
            else:
                lines.append(f"  {k}={v}")

        return f"Environment ({len(items)} variables):\n" + "\n".join(lines)

    def _path(self) -> str:
        path = os.environ.get("PATH", "")
        entries = path.split(os.pathsep)
        lines = ["PATH entries:"]
        for i, entry in enumerate(entries):
            exists = os.path.isdir(entry)
            marker = "+" if exists else "-"
            lines.append(f"  {marker} {entry}")
        return "\n".join(lines)

    def _which(self, command: str) -> str:
        import shutil

        result = shutil.which(command)
        if result:
            return f"{command}: {result}"
        return f"{command}: not found in PATH"

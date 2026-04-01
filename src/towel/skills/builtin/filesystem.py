"""Filesystem skill — read, write, list files and directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class FileSystemSkill(Skill):
    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "Read, write, and list files and directories"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="read_file",
                description="Read the contents of a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="write_file",
                description="Write content to a file (creates parent dirs if needed)",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            ),
            ToolDefinition(
                name="list_directory",
                description="List files and subdirectories in a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default: cwd)"},
                    },
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "read_file":
                path = Path(arguments["path"]).expanduser()
                if not path.exists():
                    return f"File not found: {path}"
                if path.stat().st_size > 1_000_000:
                    return f"File too large ({path.stat().st_size} bytes). Max 1MB."
                return path.read_text(encoding="utf-8", errors="replace")

            case "write_file":
                path = Path(arguments["path"]).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(arguments["content"], encoding="utf-8")
                return f"Written {len(arguments['content'])} bytes to {path}"

            case "list_directory":
                path = Path(arguments.get("path", ".")).expanduser()
                if not path.is_dir():
                    return f"Not a directory: {path}"
                entries = sorted(path.iterdir())
                lines = []
                for entry in entries[:100]:  # cap at 100 entries
                    prefix = "d " if entry.is_dir() else "f "
                    size = entry.stat().st_size if entry.is_file() else 0
                    lines.append(
                        f"{prefix}{entry.name}  ({size}B)" if size else f"{prefix}{entry.name}/"
                    )
                if len(entries) > 100:
                    lines.append(f"... and {len(entries) - 100} more")
                return "\n".join(lines)

            case _:
                return f"Unknown tool: {tool_name}"

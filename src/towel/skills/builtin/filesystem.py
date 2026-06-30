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
                name="edit_file",
                description=(
                    "Replace an exact string in a file with new text. "
                    "Use this for targeted edits — much safer than "
                    "rewriting the whole file via write_file, especially "
                    "for large files. The old_string must match EXACTLY "
                    "(including whitespace) and must be unique in the "
                    "file or the edit is refused."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "old_string": {
                            "type": "string",
                            "description": (
                                "Exact text to find. Must be unique in the "
                                "file — include enough surrounding context "
                                "to make it unique if needed."
                            ),
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
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
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(arguments["content"], encoding="utf-8")
                except OSError as exc:
                    return f"Failed to write {path}: {exc}"
                return f"Written {len(arguments['content'])} bytes to {path}"

            case "edit_file":
                # Targeted string replacement. The Codex / Claude
                # Edit tool pattern — much safer than write_file
                # for modifying existing files because the model
                # doesn't have to reproduce the whole file from
                # memory, and any drift surfaces as a unique-match
                # failure rather than a silent rewrite.
                path = Path(arguments["path"]).expanduser()
                if not path.exists():
                    return f"File not found: {path}"
                if path.stat().st_size > 1_000_000:
                    return (
                        f"File too large for edit_file "
                        f"({path.stat().st_size} bytes). Max 1MB — use "
                        "write_file for full-rewrites or chunk the edit."
                    )
                old_string = arguments.get("old_string", "")
                new_string = arguments.get("new_string", "")
                if not old_string:
                    return (
                        "Error: old_string must be non-empty. To "
                        "create or replace a whole file use write_file."
                    )
                if old_string == new_string:
                    return (
                        "Error: old_string and new_string are identical "
                        "— no edit to perform."
                    )
                content = path.read_text(encoding="utf-8", errors="replace")
                count = content.count(old_string)
                if count == 0:
                    return (
                        f"Error: old_string not found in {path}. "
                        "Check whitespace, escape sequences, and "
                        "exact characters."
                    )
                if count > 1:
                    return (
                        f"Error: old_string matches {count} places in "
                        f"{path} — must be unique. Add surrounding "
                        "context to the old_string to disambiguate."
                    )
                new_content = content.replace(old_string, new_string, 1)
                try:
                    path.write_text(new_content, encoding="utf-8")
                except OSError as exc:
                    return f"Failed to write {path}: {exc}"
                # Surface the actual delta so the model knows what
                # changed without re-reading the file.
                delta = len(new_content) - len(content)
                sign = "+" if delta >= 0 else ""
                return (
                    f"Edited {path}: replaced 1 occurrence "
                    f"({sign}{delta} bytes)"
                )

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

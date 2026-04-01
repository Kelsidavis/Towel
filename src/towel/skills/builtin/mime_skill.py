"""MIME type skill — identify file types by extension or content."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class MimeSkill(Skill):
    @property
    def name(self) -> str: return "mime"
    @property
    def description(self) -> str: return "Identify file MIME types by extension or path"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="mime_type", description="Get MIME type for a file extension or path",
                parameters={"type":"object","properties":{"path":{"type":"string","description":"File path or extension (e.g., .json, photo.png)"}},"required":["path"]}),
            ToolDefinition(name="mime_extensions", description="Get file extensions for a MIME type",
                parameters={"type":"object","properties":{"mime":{"type":"string","description":"MIME type (e.g., application/json)"}},"required":["mime"]}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "mime_type":
            p = arguments["path"]
            if not p.startswith("."): p = Path(p).suffix or f".{p}"
            mt, enc = mimetypes.guess_type(f"file{p}")
            return f"{p}: {mt or 'unknown'}" + (f" (encoding: {enc})" if enc else "")
        elif tool_name == "mime_extensions":
            exts = mimetypes.guess_all_extensions(arguments["mime"])
            if not exts: return f"No extensions for: {arguments['mime']}"
            return f"{arguments['mime']}: {', '.join(exts)}"
        return f"Unknown: {tool_name}"

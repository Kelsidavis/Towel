"""Clipboard skill — read and write the system clipboard (macOS)."""

from __future__ import annotations

import asyncio
import platform
from typing import Any

from towel.skills.base import Skill, ToolDefinition

MAX_CLIPBOARD_SIZE = 100_000


class ClipboardSkill(Skill):
    @property
    def name(self) -> str:
        return "clipboard"

    @property
    def description(self) -> str:
        return "Read and write the system clipboard"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="clipboard_read",
                description="Read the current clipboard contents",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="clipboard_write",
                description="Write text to the clipboard",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Text to copy to clipboard"},
                    },
                    "required": ["content"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "clipboard_read":
                return await self._read()
            case "clipboard_write":
                return await self._write(arguments["content"])
            case _:
                return f"Unknown tool: {tool_name}"

    async def _read(self) -> str:
        if platform.system() == "Darwin":
            cmd = ["pbpaste"]
        elif platform.system() == "Linux":
            cmd = ["xclip", "-selection", "clipboard", "-o"]
        else:
            return "Clipboard not supported on this platform."

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
            content = stdout.decode("utf-8", errors="replace")[:MAX_CLIPBOARD_SIZE]
            if not content:
                return "(clipboard is empty)"
            return content
        except Exception as e:
            return f"Failed to read clipboard: {e}"

    async def _write(self, content: str) -> str:
        if platform.system() == "Darwin":
            cmd = ["pbcopy"]
        elif platform.system() == "Linux":
            cmd = ["xclip", "-selection", "clipboard"]
        else:
            return "Clipboard not supported on this platform."

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(
                proc.communicate(input=content.encode("utf-8")),
                timeout=5,
            )
            return f"Copied {len(content)} characters to clipboard."
        except Exception as e:
            return f"Failed to write clipboard: {e}"

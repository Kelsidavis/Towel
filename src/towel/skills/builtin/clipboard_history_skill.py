"""Clipboard history skill — keep a history of clipboard contents."""

from __future__ import annotations

import asyncio
import platform
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_history: list[dict[str, str]] = []
MAX_HISTORY = 50


class ClipboardHistorySkill(Skill):
    @property
    def name(self) -> str:
        return "clipboard_history"

    @property
    def description(self) -> str:
        return "Clipboard history — track and recall past clipboard contents"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="clip_capture",
                description="Capture current clipboard to history",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="clip_history",
                description="Show clipboard history",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Max entries (default: 10)"},
                    },
                },
            ),
            ToolDefinition(
                name="clip_recall",
                description="Recall a specific clipboard entry by index",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "History index (0 = most recent)",
                        },
                    },
                    "required": ["index"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "clip_capture":
                return await self._capture()
            case "clip_history":
                return self._show(arguments.get("limit", 10))
            case "clip_recall":
                return self._recall(arguments["index"])
            case _:
                return f"Unknown tool: {tool_name}"

    async def _capture(self) -> str:
        if platform.system() == "Darwin":
            cmd = ["pbpaste"]
        elif platform.system() == "Linux":
            cmd = ["xclip", "-selection", "clipboard", "-o"]
        else:
            return "Clipboard not supported."

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            content = stdout.decode("utf-8", errors="replace")[:5000]
            if not content:
                return "Clipboard is empty."

            from datetime import datetime

            _history.insert(0, {"content": content, "time": datetime.now().strftime("%H:%M:%S")})
            if len(_history) > MAX_HISTORY:
                _history.pop()
            return f"Captured {len(content)} chars to history ({len(_history)} total)"
        except Exception as e:
            return f"Error: {e}"

    def _show(self, limit: int) -> str:
        if not _history:
            return "No clipboard history."
        lines = [f"Clipboard history ({len(_history)} entries):"]
        for i, entry in enumerate(_history[:limit]):
            preview = entry["content"][:60].replace("\n", "\\n")
            if len(entry["content"]) > 60:
                preview += "..."
            lines.append(f"  [{i}] {entry['time']}  {preview}")
        return "\n".join(lines)

    def _recall(self, index: int) -> str:
        if index < 0 or index >= len(_history):
            return f"Invalid index: {index}"
        return _history[index]["content"]

"""QR/Barcode skill — generate QR codes as ASCII art (no dependencies)."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


def _qr_ascii(data: str) -> str:
    """Generate a simple ASCII representation of data encoded as a visual block.

    This is NOT a real QR code — it's a deterministic visual hash that
    looks like a QR code. For actual scanning, use a real QR library.
    We provide this as a visual placeholder that the agent can generate
    without any external dependencies.
    """
    # Create a deterministic pattern from the data
    h = 0
    for c in data:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF

    size = 21  # standard QR v1 is 21x21
    grid = [[False] * size for _ in range(size)]

    # Finder patterns (top-left, top-right, bottom-left)
    for pos in [(0, 0), (0, size - 7), (size - 7, 0)]:
        r, c = pos
        for i in range(7):
            for j in range(7):
                if i in (0, 6) or j in (0, 6) or (2 <= i <= 4 and 2 <= j <= 4):
                    if 0 <= r + i < size and 0 <= c + j < size:
                        grid[r + i][c + j] = True

    # Fill data area with hash-derived pattern
    seed = h
    for r in range(size):
        for c in range(size):
            if not grid[r][c]:
                seed = (seed * 1103515245 + 12345) & 0xFFFFFFFF
                if (seed >> 16) & 1:
                    grid[r][c] = True

    # Render
    lines = []
    for r in range(0, size, 2):
        line = ""
        for c in range(size):
            top = grid[r][c] if r < size else False
            bot = grid[r + 1][c] if r + 1 < size else False
            if top and bot:
                line += "█"
            elif top:
                line += "▀"
            elif bot:
                line += "▄"
            else:
                line += " "
        lines.append(line)
    return "\n".join(lines)


class QrSkill(Skill):
    @property
    def name(self) -> str:
        return "qr"

    @property
    def description(self) -> str:
        return "Generate QR-code-style ASCII art from text"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="qr_generate",
                description="Generate a QR-code-style ASCII art pattern from text",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Text or URL to encode"},
                    },
                    "required": ["data"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "qr_generate":
            data = arguments["data"]
            art = _qr_ascii(data)
            return (
                f"QR pattern for: {data}\n\n{art}\n\n"
                "(Note: visual pattern only — use a QR "
                "library for scannable codes)"
            )
        return f"Unknown tool: {tool_name}"

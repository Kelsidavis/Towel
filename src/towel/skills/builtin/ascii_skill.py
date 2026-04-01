"""ASCII art skill — generate text banners and box drawings."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition

# Simple block font (each char is 5 lines tall)
_BLOCK_FONT: dict[str, list[str]] = {
    "A": ["  █  ", "█   █", "█████", "█   █", "█   █"],
    "B": ["████ ", "█   █", "████ ", "█   █", "████ "],
    "C": [" ████", "█    ", "█    ", "█    ", " ████"],
    "D": ["████ ", "█   █", "█   █", "█   █", "████ "],
    "E": ["█████", "█    ", "███  ", "█    ", "█████"],
    "F": ["█████", "█    ", "███  ", "█    ", "█    "],
    "G": [" ████", "█    ", "█  ██", "█   █", " ████"],
    "H": ["█   █", "█   █", "█████", "█   █", "█   █"],
    "I": ["█████", "  █  ", "  █  ", "  █  ", "█████"],
    "J": ["█████", "   █ ", "   █ ", "█  █ ", " ██  "],
    "K": ["█  █ ", "█ █  ", "██   ", "█ █  ", "█  █ "],
    "L": ["█    ", "█    ", "█    ", "█    ", "█████"],
    "M": ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
    "N": ["█   █", "██  █", "█ █ █", "█  ██", "█   █"],
    "O": [" ███ ", "█   █", "█   █", "█   █", " ███ "],
    "P": ["████ ", "█   █", "████ ", "█    ", "█    "],
    "Q": [" ███ ", "█   █", "█ █ █", "█  █ ", " ██ █"],
    "R": ["████ ", "█   █", "████ ", "█ █  ", "█  █ "],
    "S": [" ████", "█    ", " ███ ", "    █", "████ "],
    "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  "],
    "U": ["█   █", "█   █", "█   █", "█   █", " ███ "],
    "V": ["█   █", "█   █", "█   █", " █ █ ", "  █  "],
    "W": ["█   █", "█   █", "█ █ █", "██ ██", "█   █"],
    "X": ["█   █", " █ █ ", "  █  ", " █ █ ", "█   █"],
    "Y": ["█   █", " █ █ ", "  █  ", "  █  ", "  █  "],
    "Z": ["█████", "   █ ", "  █  ", " █   ", "█████"],
    " ": ["     ", "     ", "     ", "     ", "     "],
    "0": [" ███ ", "█  ██", "█ █ █", "██  █", " ███ "],
    "1": ["  █  ", " ██  ", "  █  ", "  █  ", "█████"],
    "2": [" ███ ", "█   █", "  ██ ", " █   ", "█████"],
    "3": ["████ ", "    █", " ██  ", "    █", "████ "],
    "!": ["  █  ", "  █  ", "  █  ", "     ", "  █  "],
}


class AsciiSkill(Skill):
    @property
    def name(self) -> str:
        return "ascii"

    @property
    def description(self) -> str:
        return "Generate ASCII art text banners and box drawings"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="ascii_banner",
                description="Generate a large ASCII art text banner",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to render (A-Z, 0-9, space, !)",
                        },
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="ascii_box",
                description="Draw a box around text",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to put in a box"},
                        "style": {
                            "type": "string",
                            "enum": ["single", "double", "rounded", "heavy"],
                            "description": "Box style",
                        },
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="ascii_table",
                description="Draw an ASCII table from data",
                parameters={
                    "type": "object",
                    "properties": {
                        "headers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Column headers",
                        },
                        "rows": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                            "description": "Row data",
                        },
                    },
                    "required": ["headers", "rows"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "ascii_banner":
                return self._banner(arguments["text"])
            case "ascii_box":
                return self._box(arguments["text"], arguments.get("style", "single"))
            case "ascii_table":
                return self._table(arguments["headers"], arguments["rows"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _banner(self, text: str) -> str:
        text = text.upper()[:30]
        lines = [""] * 5
        for ch in text:
            glyph = _BLOCK_FONT.get(ch, _BLOCK_FONT.get(" "))
            if glyph:
                for i in range(5):
                    lines[i] += glyph[i] + " "
        return "\n".join(lines)

    def _box(self, text: str, style: str) -> str:
        chars = {
            "single": ("┌", "─", "┐", "│", "└", "┘"),
            "double": ("╔", "═", "╗", "║", "╚", "╝"),
            "rounded": ("╭", "─", "╮", "│", "╰", "╯"),
            "heavy": ("┏", "━", "┓", "┃", "┗", "┛"),
        }.get(style, ("┌", "─", "┐", "│", "└", "┘"))
        tl, h, tr, v, bl, br = chars
        content_lines = text.splitlines()
        width = max(len(line) for line in content_lines) + 2
        lines = [f"{tl}{h * width}{tr}"]
        for cl in content_lines:
            lines.append(f"{v} {cl.ljust(width - 2)} {v}")
        lines.append(f"{bl}{h * width}{br}")
        return "\n".join(lines)

    def _table(self, headers: list[str], rows: list[list[str]]) -> str:
        all_rows = [headers] + rows
        widths = [
            max(len(str(row[i])) if i < len(row) else 0 for row in all_rows)
            for i in range(len(headers))
        ]
        widths = [max(w, 3) for w in widths]
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

        def fmt_row(row):
            cells = []
            for i, w in enumerate(widths):
                val = str(row[i]) if i < len(row) else ""
                cells.append(f" {val.ljust(w)} ")
            return "|" + "|".join(cells) + "|"

        lines = [sep, fmt_row(headers), sep]
        for row in rows:
            lines.append(fmt_row(row))
        lines.append(sep)
        return "\n".join(lines)

"""String manipulation skill — encoding, hashing, escaping, case conversion."""

from __future__ import annotations
import html
import urllib.parse
from typing import Any
from towel.skills.base import Skill, ToolDefinition


class StringSkill(Skill):
    @property
    def name(self) -> str: return "string"
    @property
    def description(self) -> str: return "String manipulation — escape, unescape, encode, count, truncate"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="string_escape", description="Escape/unescape strings for different contexts (HTML, JSON, SQL, shell)",
                parameters={"type":"object","properties":{
                    "text":{"type":"string","description":"Text to escape"},
                    "format":{"type":"string","enum":["html","json","sql","shell","url","xml"],"description":"Escape format"},
                    "unescape":{"type":"boolean","description":"Unescape instead (default: false)"},
                },"required":["text","format"]}),
            ToolDefinition(name="string_pad", description="Pad a string to a specific length",
                parameters={"type":"object","properties":{
                    "text":{"type":"string","description":"Text to pad"},
                    "length":{"type":"integer","description":"Target length"},
                    "char":{"type":"string","description":"Pad character (default: space)"},
                    "side":{"type":"string","enum":["left","right","center"],"description":"Padding side"},
                },"required":["text","length"]}),
            ToolDefinition(name="string_truncate", description="Truncate a string with ellipsis",
                parameters={"type":"object","properties":{
                    "text":{"type":"string","description":"Text to truncate"},
                    "length":{"type":"integer","description":"Max length"},
                    "suffix":{"type":"string","description":"Suffix (default: '...')"},
                },"required":["text","length"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "string_escape": return self._escape(arguments["text"], arguments["format"], arguments.get("unescape",False))
            case "string_pad": return self._pad(arguments["text"], arguments["length"], arguments.get("char"," "), arguments.get("side","right"))
            case "string_truncate": return self._truncate(arguments["text"], arguments["length"], arguments.get("suffix","..."))
            case _: return f"Unknown tool: {tool_name}"

    def _escape(self, text: str, fmt: str, unescape: bool) -> str:
        if unescape:
            match fmt:
                case "html": return html.unescape(text)
                case "url": return urllib.parse.unquote(text)
                case _: return f"Unescape not supported for {fmt}"
        match fmt:
            case "html": return html.escape(text)
            case "json": import json; return json.dumps(text)[1:-1]
            case "sql": return text.replace("'", "''")
            case "shell": return "'" + text.replace("'", "'\\''") + "'"
            case "url": return urllib.parse.quote(text, safe="")
            case "xml": return html.escape(text, quote=True)
            case _: return text

    def _pad(self, text: str, length: int, char: str, side: str) -> str:
        c = char[0] if char else " "
        match side:
            case "left": return text.rjust(length, c)
            case "center": return text.center(length, c)
            case _: return text.ljust(length, c)

    def _truncate(self, text: str, length: int, suffix: str) -> str:
        if len(text) <= length: return text
        return text[:length - len(suffix)] + suffix

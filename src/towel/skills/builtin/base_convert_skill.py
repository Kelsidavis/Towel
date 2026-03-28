"""Base conversion skill — convert numbers between bases."""
from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class BaseConvertSkill(Skill):
    @property
    def name(self) -> str: return "base_convert"
    @property
    def description(self) -> str: return "Convert numbers between bases (binary, octal, decimal, hex)"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="convert_base", description="Convert a number between bases",
                parameters={"type":"object","properties":{
                    "number":{"type":"string","description":"Number to convert (e.g., 0xFF, 0b1010, 42)"},
                    "to_base":{"type":"integer","description":"Target base (2, 8, 10, 16)"},
                },"required":["number"]}),
            ToolDefinition(name="show_all_bases", description="Show a number in all common bases",
                parameters={"type":"object","properties":{"number":{"type":"string"}},"required":["number"]}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            n = int(arguments["number"], 0)  # auto-detect base from prefix
        except ValueError:
            return f"Invalid number: {arguments['number']}"
        if tool_name == "convert_base":
            base = arguments.get("to_base", 10)
            match base:
                case 2: return f"Binary: {bin(n)}"
                case 8: return f"Octal: {oct(n)}"
                case 10: return f"Decimal: {n}"
                case 16: return f"Hex: {hex(n)}"
                case _: return f"Unsupported base: {base}"
        elif tool_name == "show_all_bases":
            return f"  Decimal: {n}\n  Binary:  {bin(n)}\n  Octal:   {oct(n)}\n  Hex:     {hex(n)}"
        return f"Unknown: {tool_name}"

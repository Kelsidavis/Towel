"""Date format skill — parse and format dates between formats."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_FORMATS = {
    "iso": "%Y-%m-%dT%H:%M:%S", "date": "%Y-%m-%d", "us": "%m/%d/%Y",
    "eu": "%d/%m/%Y", "long": "%B %d, %Y", "time": "%H:%M:%S",
    "datetime": "%Y-%m-%d %H:%M:%S", "rfc2822": "%a, %d %b %Y %H:%M:%S",
    "unix": "unix", "relative": "relative",
}

class DateFormatSkill(Skill):
    @property
    def name(self) -> str: return "date_format"
    @property
    def description(self) -> str: return "Parse and convert dates between formats"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="date_convert", description="Convert a date string between formats",
                parameters={"type":"object","properties":{
                    "date":{"type":"string","description":"Date string to convert"},
                    "to_format":{"type":"string","description":f"Target: {', '.join(_FORMATS.keys())}"},
                },"required":["date","to_format"]}),
            ToolDefinition(name="date_now", description="Get current date/time in multiple formats",
                parameters={"type":"object","properties":{}}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "date_now":
            now = datetime.now(UTC)
            lines = ["Current date/time (UTC):"]
            for name, fmt in _FORMATS.items():
                if fmt == "unix": lines.append(f"  {name}: {int(now.timestamp())}")
                elif fmt == "relative": continue
                else: lines.append(f"  {name}: {now.strftime(fmt)}")
            return "\n".join(lines)
        elif tool_name == "date_convert":
            dt = self._parse(arguments["date"])
            if not dt: return f"Cannot parse: {arguments['date']}"
            fmt = _FORMATS.get(arguments["to_format"])
            if not fmt: return f"Unknown format: {arguments['to_format']}"
            if fmt == "unix": return str(int(dt.timestamp()))
            if fmt == "relative":
                delta = datetime.now(UTC) - dt.replace(tzinfo=UTC)
                days = delta.days
                if days == 0: return "today"
                if days == 1: return "yesterday"
                if days < 30: return f"{days} days ago"
                return f"{days//30} months ago"
            return dt.strftime(fmt)
        return f"Unknown: {tool_name}"

    def _parse(self, s: str) -> datetime | None:
        for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y",
                     "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"]:
            try: return datetime.strptime(s.strip(), fmt)
            except: continue
        try: return datetime.fromtimestamp(float(s))
        except: return None

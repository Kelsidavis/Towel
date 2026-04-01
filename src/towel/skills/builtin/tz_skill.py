"""Timezone skill — world clock and conversion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_TZ = {
    "UTC": 0,
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "GMT": 0,
    "BST": 1,
    "CET": 1,
    "CEST": 2,
    "EET": 2,
    "EEST": 3,
    "JST": 9,
    "KST": 9,
    "CST_CN": 8,
    "IST": 5,
    "AEST": 10,
    "AEDT": 11,
    "NZST": 12,
    "HST": -10,
}


class TimezoneSkill(Skill):
    @property
    def name(self) -> str:
        return "tz"

    @property
    def description(self) -> str:
        return "World clock and timezone conversion"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="tz_now",
                description="Show current time in multiple timezones",
                parameters={
                    "type": "object",
                    "properties": {
                        "zones": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Timezone codes",
                        }
                    },
                },
            ),
            ToolDefinition(
                name="tz_convert",
                description="Convert a time between timezones",
                parameters={
                    "type": "object",
                    "properties": {
                        "time": {"type": "string", "description": "Time (HH:MM)"},
                        "from_tz": {"type": "string"},
                        "to_tz": {"type": "string"},
                    },
                    "required": ["time", "from_tz", "to_tz"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "tz_now":
            zones = arguments.get("zones", ["UTC", "EST", "PST", "CET", "JST"])
            now = datetime.now(UTC)
            lines = ["World clock:"]
            for z in zones:
                off = _TZ.get(z.upper(), 0)
                t = now + timedelta(hours=off)
                lines.append(f"  {z:6s} {t.strftime('%H:%M %a %b %d')}")
            return "\n".join(lines)
        elif tool_name == "tz_convert":
            parts = arguments["time"].split(":")
            h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            f_off = _TZ.get(arguments["from_tz"].upper(), 0)
            t_off = _TZ.get(arguments["to_tz"].upper(), 0)
            utc_h = (h - f_off) % 24
            dest_h = (utc_h + t_off) % 24
            return (
                f"{h:02d}:{m:02d} {arguments['from_tz']} = "
                f"{dest_h:02d}:{m:02d} {arguments['to_tz']}"
            )
        return f"Unknown: {tool_name}"

"""Time and date skill — give the agent awareness of the current time."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any
import time

from towel.skills.base import Skill, ToolDefinition

# Common timezone offsets (no pytz dependency needed)
_TIMEZONE_OFFSETS: dict[str, int] = {
    "UTC": 0, "GMT": 0,
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
    "CET": 1, "CEST": 2, "EET": 2, "EEST": 3,
    "JST": 9, "KST": 9, "CST_CN": 8, "IST": 5,
    "AEST": 10, "AEDT": 11, "NZST": 12, "NZDT": 13,
    "HST": -10, "AKST": -9, "AKDT": -8,
    "BRT": -3, "ART": -3, "CLT": -4,
    "WAT": 1, "CAT": 2, "EAT": 3, "SAST": 2,
}


class TimeSkill(Skill):
    @property
    def name(self) -> str:
        return "time"

    @property
    def description(self) -> str:
        return "Get current time, date, and timezone information"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="current_time",
                description="Get the current date and time, optionally in a specific timezone",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone abbreviation (e.g., UTC, PST, EST, JST). Default: local time",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="time_between",
                description="Calculate the duration between two dates (YYYY-MM-DD format)",
                parameters={
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                    },
                    "required": ["start", "end"],
                },
            ),
            ToolDefinition(
                name="unix_timestamp",
                description="Get or convert Unix timestamps",
                parameters={
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "number",
                            "description": "Unix timestamp to convert to human-readable. Omit to get current timestamp.",
                        },
                    },
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "current_time":
                return self._current_time(arguments.get("timezone"))
            case "time_between":
                return self._time_between(arguments["start"], arguments["end"])
            case "unix_timestamp":
                return self._unix_timestamp(arguments.get("timestamp"))
            case _:
                return f"Unknown tool: {tool_name}"

    def _current_time(self, tz_name: str | None = None) -> str:
        if tz_name:
            tz_upper = tz_name.upper().replace(" ", "_")
            offset_hours = _TIMEZONE_OFFSETS.get(tz_upper)
            if offset_hours is None:
                available = ", ".join(sorted(_TIMEZONE_OFFSETS.keys()))
                return f"Unknown timezone: {tz_name}. Available: {available}"
            tz = timezone(timedelta(hours=offset_hours))
            now = datetime.now(tz)
            label = tz_name.upper()
        else:
            now = datetime.now()
            label = "local"

        lines = [
            f"Current time ({label}):",
            f"  Date: {now.strftime('%A, %B %d, %Y')}",
            f"  Time: {now.strftime('%I:%M:%S %p')}",
            f"  ISO:  {now.isoformat()}",
            f"  Unix: {int(now.timestamp())}",
        ]
        return "\n".join(lines)

    def _time_between(self, start_str: str, end_str: str) -> str:
        try:
            start = datetime.strptime(start_str.strip(), "%Y-%m-%d")
            end = datetime.strptime(end_str.strip(), "%Y-%m-%d")
        except ValueError:
            return "Invalid date format. Use YYYY-MM-DD (e.g., 2026-03-27)."

        delta = end - start
        days = abs(delta.days)
        weeks = days // 7
        months = days // 30
        years = days // 365

        direction = "from now" if delta.days >= 0 else "ago"
        parts = [f"{days} days {direction}"]
        if weeks >= 1:
            parts.append(f"  ({weeks} weeks and {days % 7} days)")
        if months >= 1:
            parts.append(f"  (~{months} months)")
        if years >= 1:
            parts.append(f"  (~{years} years and {days % 365} days)")

        return f"Duration between {start_str} and {end_str}:\n" + "\n".join(parts)

    def _unix_timestamp(self, ts: float | None = None) -> str:
        if ts is not None:
            try:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                return (
                    f"Unix timestamp {int(ts)}:\n"
                    f"  UTC: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                    f"  ISO: {dt.isoformat()}"
                )
            except (ValueError, OSError, OverflowError):
                return f"Invalid timestamp: {ts}"
        else:
            now = int(time.time())
            return f"Current Unix timestamp: {now}"

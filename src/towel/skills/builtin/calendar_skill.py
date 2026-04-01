"""Calendar skill — date math, business days, countdown, and calendar display."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class CalendarSkill(Skill):
    @property
    def name(self) -> str:
        return "calendar"

    @property
    def description(self) -> str:
        return "Calendar display, date math, business days, and countdowns"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="cal_month",
                description="Display a calendar for a month",
                parameters={
                    "type": "object",
                    "properties": {
                        "year": {"type": "integer", "description": "Year (default: current)"},
                        "month": {
                            "type": "integer",
                            "description": "Month 1-12 (default: current)",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="cal_business_days",
                description="Count business days between two dates",
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
                name="cal_add_days",
                description="Add or subtract days/weeks/months from a date",
                parameters={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Base date (YYYY-MM-DD, default: today)",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Days to add (negative to subtract)",
                        },
                        "weeks": {"type": "integer", "description": "Weeks to add"},
                        "business_days": {"type": "integer", "description": "Business days to add"},
                    },
                },
            ),
            ToolDefinition(
                name="cal_countdown",
                description="Countdown to a date — days, weeks, months remaining",
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target date (YYYY-MM-DD)"},
                        "label": {"type": "string", "description": "Event name (optional)"},
                    },
                    "required": ["target"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "cal_month":
                return self._month(arguments.get("year"), arguments.get("month"))
            case "cal_business_days":
                return self._biz_days(arguments["start"], arguments["end"])
            case "cal_add_days":
                return self._add(
                    arguments.get("date"),
                    arguments.get("days", 0),
                    arguments.get("weeks", 0),
                    arguments.get("business_days", 0),
                )
            case "cal_countdown":
                return self._countdown(arguments["target"], arguments.get("label", ""))
            case _:
                return f"Unknown tool: {tool_name}"

    def _month(self, year: int | None, month: int | None) -> str:
        today = date.today()
        y = year or today.year
        m = month or today.month
        cal = calendar.TextCalendar()
        return cal.formatmonth(y, m)

    def _biz_days(self, start_str: str, end_str: str) -> str:
        try:
            s = datetime.strptime(start_str, "%Y-%m-%d").date()
            e = datetime.strptime(end_str, "%Y-%m-%d").date()
        except ValueError:
            return "Invalid date format. Use YYYY-MM-DD."
        count = 0
        current = s
        while current <= e:
            if current.weekday() < 5:
                count += 1
            current += timedelta(days=1)
        total = (e - s).days + 1
        return (
            f"Between {start_str} and {end_str}:\n"
            f"  Calendar days: {total}\n"
            f"  Business days: {count}\n"
            f"  Weekends: {total - count}"
        )

    def _add(self, date_str: str | None, days: int, weeks: int, biz_days: int) -> str:
        try:
            base = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else date.today()
        except ValueError:
            return "Invalid date format."
        result = base + timedelta(days=days, weeks=weeks)
        if biz_days:
            added = 0
            step = 1 if biz_days > 0 else -1
            while added < abs(biz_days):
                result += timedelta(days=step)
                if result.weekday() < 5:
                    added += 1
        return f"{base} + {days}d {weeks}w {biz_days}bd = {result} ({result.strftime('%A')})"

    def _countdown(self, target_str: str, label: str) -> str:
        try:
            target = datetime.strptime(target_str, "%Y-%m-%d").date()
        except ValueError:
            return "Invalid date format."
        today = date.today()
        delta = target - today
        days = delta.days
        if days < 0:
            return f"{label or target_str} was {abs(days)} days ago."
        weeks = days // 7
        months = days // 30
        name = label or target_str
        return (
            f"Countdown to {name}:\n"
            f"  {days} days ({weeks} weeks, ~{months} months)\n"
            f"  Target: {target.strftime('%A, %B %d, %Y')}"
        )

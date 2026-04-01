"""Cron skill — parse, explain, and generate cron expressions."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_FIELD_NAMES = ["minute", "hour", "day-of-month", "month", "day-of-week"]
_MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}
_DOW_NAMES = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def _explain_field(field: str, name: str, names: dict | None = None) -> str:
    if field == "*":
        return f"every {name}"
    if field.startswith("*/"):
        return f"every {field[2:]} {name}(s)"
    if "," in field:
        parts = field.split(",")
        if names:
            parts = [names.get(int(p), p) if p.isdigit() else p for p in parts]
        return f"{name} {', '.join(parts)}"
    if "-" in field:
        a, b = field.split("-", 1)
        if names:
            a = names.get(int(a), a) if a.isdigit() else a
            b = names.get(int(b), b) if b.isdigit() else b
        return f"{name} {a} through {b}"
    if names and field.isdigit():
        return f"{name} {names.get(int(field), field)}"
    return f"{name} {field}"


def explain_cron(expr: str) -> str:
    parts = expr.strip().split()
    if len(parts) != 5:
        return f"Invalid cron: expected 5 fields, got {len(parts)}"

    minute, hour, dom, month, dow = parts
    pieces = []

    # Time
    if minute == "*" and hour == "*":
        pieces.append("Every minute")
    elif minute.startswith("*/"):
        pieces.append(f"Every {minute[2:]} minutes")
    elif hour == "*":
        pieces.append(f"At minute {minute} of every hour")
    elif minute == "0":
        pieces.append(f"At {hour}:00")
    else:
        pieces.append(f"At {hour}:{minute.zfill(2)}")

    # Day of month
    if dom != "*":
        pieces.append(_explain_field(dom, "on day"))

    # Month
    if month != "*":
        pieces.append(_explain_field(month, "in", _MONTH_NAMES))

    # Day of week
    if dow != "*":
        pieces.append(_explain_field(dow, "on", _DOW_NAMES))

    return ", ".join(pieces)


def next_runs(expr: str, count: int = 5) -> list[str]:
    """Approximate next N run times (simple, not fully cron-accurate)."""
    parts = expr.strip().split()
    if len(parts) != 5:
        return ["Invalid cron expression"]

    minute, hour, dom, month, dow = parts
    now = datetime.now()
    results = []
    t = now.replace(second=0, microsecond=0) + timedelta(minutes=1)

    for _ in range(60 * 24 * 31):  # scan up to 31 days
        if len(results) >= count:
            break
        m, h, d, mo, wd = t.minute, t.hour, t.day, t.month, t.weekday()
        wd_cron = (wd + 1) % 7  # Python: Mon=0, Cron: Sun=0

        if (
            _match(minute, m)
            and _match(hour, h)
            and _match(dom, d)
            and _match(month, mo)
            and _match(dow, wd_cron)
        ):
            results.append(t.strftime("%Y-%m-%d %H:%M"))
        t += timedelta(minutes=1)

    return results


def _match(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return value % step == 0
    for part in field.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            if int(a) <= value <= int(b):
                return True
        elif part.isdigit() and int(part) == value:
            return True
    return False


class CronSkill(Skill):
    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return "Parse, explain, and generate cron expressions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="cron_explain",
                description="Explain a cron expression in plain English",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Cron expression (5 fields)",
                        },
                    },
                    "required": ["expression"],
                },
            ),
            ToolDefinition(
                name="cron_next",
                description="Show the next N run times for a cron expression",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Cron expression"},
                        "count": {
                            "type": "integer",
                            "description": "Number of runs to show (default: 5)",
                        },
                    },
                    "required": ["expression"],
                },
            ),
            ToolDefinition(
                name="cron_build",
                description="Generate a cron expression from a plain English description",
                parameters={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": (
                                "When to run (e.g., 'every 5 minutes',"
                                " 'daily at 9am', 'weekdays at noon')"
                            ),
                        },
                    },
                    "required": ["description"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "cron_explain":
                return explain_cron(arguments["expression"])
            case "cron_next":
                runs = next_runs(arguments["expression"], arguments.get("count", 5))
                return "Next runs:\n" + "\n".join(f"  {r}" for r in runs)
            case "cron_build":
                return self._build(arguments["description"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _build(self, desc: str) -> str:
        d = desc.lower().strip()
        if "every minute" in d:
            return "* * * * *  (every minute)"
        if "every 5 min" in d:
            return "*/5 * * * *  (every 5 minutes)"
        if "every 10 min" in d:
            return "*/10 * * * *  (every 10 minutes)"
        if "every 15 min" in d:
            return "*/15 * * * *  (every 15 minutes)"
        if "every 30 min" in d or "half hour" in d:
            return "*/30 * * * *  (every 30 minutes)"
        if "every hour" in d or "hourly" in d:
            return "0 * * * *  (every hour at :00)"
        if "weekday" in d:
            import re

            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", d)
            h, mi = 9, 0
            if m:
                h = int(m.group(1))
                mi = int(m.group(2) or 0)
                if m.group(3) == "pm" and h < 12:
                    h += 12
            if "noon" in d:
                h, mi = 12, 0
            return f"{mi} {h} * * 1-5  (weekdays at {h}:{mi:02d})"
        if "daily at" in d or "every day at" in d:
            import re

            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", d)
            if m:
                h = int(m.group(1))
                mi = int(m.group(2) or 0)
                if m.group(3) == "pm" and h < 12:
                    h += 12
                if m.group(3) == "am" and h == 12:
                    h = 0
                return f"{mi} {h} * * *  (daily at {h}:{mi:02d})"
        if "midnight" in d:
            return "0 0 * * *  (daily at midnight)"
        if "noon" in d:
            return "0 12 * * *  (daily at noon)"
        if "weekday" in d:
            import re

            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", d)
            h, mi = 9, 0
            if m:
                h = int(m.group(1))
                mi = int(m.group(2) or 0)
                if m.group(3) == "pm" and h < 12:
                    h += 12
            return f"{mi} {h} * * 1-5  (weekdays at {h}:{mi:02d})"
        if "weekly" in d or "every week" in d:
            return "0 0 * * 0  (weekly on Sunday at midnight)"
        if "monthly" in d or "every month" in d:
            return "0 0 1 * *  (monthly on the 1st at midnight)"
        return (
            f"Could not parse: '{desc}'. Try: 'every 5 minutes', 'daily at 9am', 'weekdays at noon'"
        )

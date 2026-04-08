"""Google Calendar skill — read and manage calendar events."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from towel.skills.base import Skill, ToolDefinition

log = logging.getLogger("towel.skills.gcal")


class GCalSkill(Skill):
    """Read and manage Google Calendar events."""

    @property
    def name(self) -> str:
        return "gcal"

    @property
    def description(self) -> str:
        return "Google Calendar — list upcoming events, search, and get details"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="gcal_upcoming",
                description="List upcoming calendar events.",
                parameters={
                    "type": "object",
                    "properties": {
                        "hours": {
                            "type": "integer",
                            "description": "Look ahead window in hours (default: 24)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max events to return (default: 10)",
                        },
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID (default: 'primary')",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="gcal_today",
                description="List all events for today.",
                parameters={
                    "type": "object",
                    "properties": {
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID (default: 'primary')",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="gcal_search",
                description="Search calendar events by text query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search text (matches title, description, location)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results (default: 10)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="gcal_event",
                description="Get full details of a specific event by ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "description": "Calendar event ID",
                        },
                        "calendar_id": {
                            "type": "string",
                            "description": "Calendar ID (default: 'primary')",
                        },
                    },
                    "required": ["event_id"],
                },
            ),
            ToolDefinition(
                name="gcal_calendars",
                description="List all available calendars.",
                parameters={"type": "object", "properties": {}},
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            match tool_name:
                case "gcal_upcoming":
                    return await self._upcoming(
                        arguments.get("hours", 24),
                        arguments.get("max_results", 10),
                        arguments.get("calendar_id", "primary"),
                    )
                case "gcal_today":
                    return await self._today(arguments.get("calendar_id", "primary"))
                case "gcal_search":
                    return await self._search(
                        arguments["query"],
                        arguments.get("max_results", 10),
                    )
                case "gcal_event":
                    return await self._event(
                        arguments["event_id"],
                        arguments.get("calendar_id", "primary"),
                    )
                case "gcal_calendars":
                    return await self._calendars()
                case _:
                    return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Calendar error: {e}"

    def _get_service(self) -> Any:
        from towel.skills.builtin.google_auth import build_calendar_service
        return build_calendar_service()

    def _format_event(self, event: dict) -> str:
        """Format a single event for display."""
        summary = event.get("summary", "(no title)")
        start = event.get("start", {})
        end = event.get("end", {})

        start_str = start.get("dateTime", start.get("date", ""))
        end_str = end.get("dateTime", end.get("date", ""))

        # Parse and format nicely
        if "T" in start_str:
            try:
                dt = datetime.fromisoformat(start_str)
                start_str = dt.strftime("%I:%M %p")
                dt_end = datetime.fromisoformat(end_str)
                end_str = dt_end.strftime("%I:%M %p")
                time_str = f"{start_str} - {end_str}"
            except Exception:
                time_str = f"{start_str} - {end_str}"
        else:
            time_str = "All day"

        location = event.get("location", "")
        attendees = event.get("attendees", [])
        description = event.get("description", "")
        event_id = event.get("id", "")

        lines = [f"- **{summary}** ({time_str})"]
        if location:
            lines.append(f"  Location: {location}")
        if attendees:
            names = [a.get("displayName", a.get("email", "")) for a in attendees[:5]]
            if len(attendees) > 5:
                names.append(f"+{len(attendees) - 5} more")
            lines.append(f"  Attendees: {', '.join(names)}")
        if description:
            desc_short = description[:150].replace("\n", " ")
            if len(description) > 150:
                desc_short += "..."
            lines.append(f"  Notes: {desc_short}")
        lines.append(f"  ID: `{event_id}`")

        return "\n".join(lines)

    async def _upcoming(self, hours: int, max_results: int, calendar_id: str) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)

        now = datetime.now(timezone.utc)
        time_max = now + timedelta(hours=hours)

        events_result = await asyncio.to_thread(
            lambda: svc.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                timeMax=time_max.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        )

        events = events_result.get("items", [])
        if not events:
            return f"No events in the next {hours} hours."

        lines = [f"**{len(events)} upcoming events (next {hours}h):**\n"]
        for event in events:
            lines.append(self._format_event(event))
            lines.append("")

        return "\n".join(lines)

    async def _today(self, calendar_id: str) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)

        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        events_result = await asyncio.to_thread(
            lambda: svc.events().list(
                calendarId=calendar_id,
                timeMin=start_of_day.isoformat(),
                timeMax=end_of_day.isoformat(),
                maxResults=20,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        )

        events = events_result.get("items", [])
        if not events:
            return "No events today."

        lines = [f"**{len(events)} events today:**\n"]
        for event in events:
            lines.append(self._format_event(event))
            lines.append("")

        return "\n".join(lines)

    async def _search(self, query: str, max_results: int) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)

        now = datetime.now(timezone.utc)

        events_result = await asyncio.to_thread(
            lambda: svc.events().list(
                calendarId="primary",
                timeMin=now.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
                q=query,
            ).execute()
        )

        events = events_result.get("items", [])
        if not events:
            return f"No upcoming events matching '{query}'."

        lines = [f"**{len(events)} events matching '{query}':**\n"]
        for event in events:
            lines.append(self._format_event(event))
            lines.append("")

        return "\n".join(lines)

    async def _event(self, event_id: str, calendar_id: str) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)

        event = await asyncio.to_thread(
            lambda: svc.events().get(calendarId=calendar_id, eventId=event_id).execute()
        )

        summary = event.get("summary", "(no title)")
        start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date", ""))
        end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date", ""))
        location = event.get("location", "None")
        description = event.get("description", "None")
        organizer = event.get("organizer", {}).get("email", "Unknown")
        status = event.get("status", "Unknown")
        attendees = event.get("attendees", [])

        lines = [
            f"**{summary}**",
            f"Start: {start}",
            f"End: {end}",
            f"Location: {location}",
            f"Organizer: {organizer}",
            f"Status: {status}",
        ]

        if attendees:
            lines.append(f"\nAttendees ({len(attendees)}):")
            for a in attendees:
                name = a.get("displayName", a.get("email", ""))
                rsvp = a.get("responseStatus", "unknown")
                lines.append(f"  - {name} ({rsvp})")

        if description and description != "None":
            lines.append(f"\nDescription:\n{description[:2000]}")

        return "\n".join(lines)

    async def _calendars(self) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)

        result = await asyncio.to_thread(
            lambda: svc.calendarList().list().execute()
        )

        calendars = result.get("items", [])
        if not calendars:
            return "No calendars found."

        lines = ["**Your calendars:**\n"]
        for cal in calendars:
            name = cal.get("summary", "")
            cal_id = cal.get("id", "")
            primary = " (primary)" if cal.get("primary") else ""
            lines.append(f"- **{name}**{primary}\n  ID: `{cal_id}`")

        return "\n".join(lines)

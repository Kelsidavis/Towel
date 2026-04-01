"""Pomodoro skill — focus timer with work/break cycles."""
from __future__ import annotations

import time as _time
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_active: dict[str, Any] = {}

class PomodoroSkill(Skill):
    @property
    def name(self) -> str: return "pomodoro"
    @property
    def description(self) -> str: return "Pomodoro focus timer — work/break cycles"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="pomo_start", description="Start a pomodoro timer",
                parameters={"type":"object","properties":{
                    "minutes":{"type":"integer","description":"Work duration (default: 25)"},
                    "task":{"type":"string","description":"What you're working on"},
                }}),
            ToolDefinition(name="pomo_status", description="Check timer status",
                parameters={"type":"object","properties":{}}),
            ToolDefinition(name="pomo_stop", description="Stop the current timer",
                parameters={"type":"object","properties":{}}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "pomo_start":
                mins = arguments.get("minutes", 25)
                task = arguments.get("task", "Focus session")
                _active["start"] = _time.time()
                _active["duration"] = mins * 60
                _active["task"] = task
                return f"Pomodoro started: {task} ({mins} min)"
            case "pomo_status":
                if "start" not in _active: return "No active timer."
                elapsed = _time.time() - _active["start"]
                remaining = max(0, _active["duration"] - elapsed)
                if remaining == 0: return f"Timer complete! '{_active['task']}' finished."
                return f"Timer: {_active['task']} — {remaining/60:.1f} min remaining"
            case "pomo_stop":
                if "start" not in _active: return "No active timer."
                task = _active.pop("task", "?"); _active.clear()
                return f"Stopped: {task}"
            case _: return f"Unknown: {tool_name}"

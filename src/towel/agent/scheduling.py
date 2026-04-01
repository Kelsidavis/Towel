"""Scheduled tasks — run pipelines or prompts on a cron schedule.

Persists schedules to ~/.towel/schedules.json so they survive restarts.
Uses asyncio for in-process scheduling (no external cron needed).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.agent.scheduling")

SCHEDULES_FILE = TOWEL_HOME / "schedules.json"


@dataclass
class Schedule:
    """A scheduled task."""
    name: str
    cron: str  # cron expression
    action: str  # "pipeline:name" or "tool:name" or "prompt:text"
    args: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: str = ""
    run_count: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "cron": self.cron, "action": self.action,
                "args": self.args, "enabled": self.enabled,
                "last_run": self.last_run, "run_count": self.run_count}

    @classmethod
    def from_dict(cls, d: dict) -> Schedule:
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


def _load_schedules() -> list[Schedule]:
    if not SCHEDULES_FILE.exists(): return []
    try:
        return [Schedule.from_dict(s) for s in json.loads(SCHEDULES_FILE.read_text())]
    except: return []


def _save_schedules(schedules: list[Schedule]) -> None:
    SCHEDULES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULES_FILE.write_text(json.dumps([s.to_dict() for s in schedules], indent=2))


def add_schedule(name: str, cron: str, action: str, args: dict | None = None) -> Schedule:
    schedules = _load_schedules()
    s = Schedule(name=name, cron=cron, action=action, args=args or {})
    # Replace existing with same name
    schedules = [x for x in schedules if x.name != name]
    schedules.append(s)
    _save_schedules(schedules)
    return s


def remove_schedule(name: str) -> bool:
    schedules = _load_schedules()
    before = len(schedules)
    schedules = [s for s in schedules if s.name != name]
    if len(schedules) < before:
        _save_schedules(schedules)
        return True
    return False


def list_schedules() -> list[Schedule]:
    return _load_schedules()


def toggle_schedule(name: str) -> str:
    schedules = _load_schedules()
    for s in schedules:
        if s.name == name:
            s.enabled = not s.enabled
            _save_schedules(schedules)
            return f"{'Enabled' if s.enabled else 'Disabled'}: {name}"
    return f"Not found: {name}"

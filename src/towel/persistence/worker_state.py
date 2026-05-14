"""Persistence for controller-side worker operational state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from towel.config import TOWEL_HOME

DEFAULT_WORKER_STATE_PATH = TOWEL_HOME / "worker_state.json"


class WorkerStateStore:
    """JSON-backed store for per-worker operational state.

    Tracks ``enabled`` (False = excluded from dispatch), ``draining``
    (drain → migrate sessions away), and optionally ``tasks`` (operator-set
    manual task override that survives reconnects and coordinator restarts).
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_WORKER_STATE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        """Load persisted worker state."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}

        result: dict[str, dict[str, Any]] = {}
        for worker_id, state in data.items():
            if not worker_id or not isinstance(state, dict):
                continue
            entry: dict[str, Any] = {
                "enabled": bool(state.get("enabled", True)),
                "draining": bool(state.get("draining", False)),
            }
            raw_tasks = state.get("tasks")
            if isinstance(raw_tasks, list):
                # Drop non-string entries defensively; the coordinator
                # resolves these to TaskType enums and silently skips
                # unknown ones on load.
                entry["tasks"] = [t for t in raw_tasks if isinstance(t, str)]
            result[str(worker_id)] = entry
        return result

    def save(self, states: dict[str, dict[str, Any]]) -> None:
        """Persist the full worker-state mapping."""
        self.path.write_text(json.dumps(states, indent=2, sort_keys=True), encoding="utf-8")

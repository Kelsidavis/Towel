"""Persistence for controller-side session worker pins."""

from __future__ import annotations

import json
from pathlib import Path

from towel.config import TOWEL_HOME

DEFAULT_PINS_PATH = TOWEL_HOME / "session_worker_pins.json"


class SessionPinStore:
    """JSON-backed store for session->worker pin mappings."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PINS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, str]:
        """Load all persisted pins."""
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(session_id): str(worker_id)
            for session_id, worker_id in data.items()
            if session_id and worker_id
        }

    def save(self, pins: dict[str, str]) -> None:
        """Persist the full pin mapping.

        Atomic write: dumps to a sibling .tmp then renames. Without
        this, a kill / disk-full mid-write leaves a half-written
        pins file that load() classifies as corrupt and replaces
        with {}, silently losing every operator-set pin. Same
        pattern memory/store.py adopted in 5512834.
        """
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps(pins, indent=2, sort_keys=True), encoding="utf-8",
        )
        tmp.replace(self.path)

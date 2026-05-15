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
        """Load all persisted pins.

        On corruption (bad JSON, non-dict shape, OS read error) rename
        the bad file to a sibling ``.corrupted-<ts>`` before returning
        ``{}``. Without this, the very next save() overwrote the
        corrupt file with the current (post-startup, possibly empty)
        in-memory pins — silently destroying every operator-set pin.
        Same pattern memory/store.py adopted in 5512834 and
        persistence/store.py picked up in 98d1c68.
        """
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._back_up_corrupt(exc)
            return {}
        if not isinstance(data, dict):
            self._back_up_corrupt(
                ValueError(f"top-level shape is {type(data).__name__}, expected dict"),
            )
            return {}
        return {
            str(session_id): str(worker_id)
            for session_id, worker_id in data.items()
            if session_id and worker_id
        }

    def _back_up_corrupt(self, reason: Exception) -> None:
        from datetime import UTC, datetime
        backup = self.path.with_name(
            f"{self.path.name}.corrupted-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        )
        try:
            self.path.replace(backup)
            import logging
            logging.getLogger("towel.persistence.session_pins").warning(
                "Failed to load pins: %s. Backed up the bad file to %s.",
                reason, backup,
            )
        except OSError:
            pass

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

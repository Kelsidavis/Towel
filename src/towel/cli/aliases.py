"""Prompt aliases — user-defined shortcuts for common prompts.

Aliases are stored in ~/.towel/aliases.json and persist across sessions.
Use /alias to create, /aliases to list, /unalias to remove.

When a user types /myalias some input, the alias prompt is expanded
with the input appended, then sent to the agent.
"""

from __future__ import annotations

import json
import logging

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.cli.aliases")

ALIASES_FILE = TOWEL_HOME / "aliases.json"


def _load() -> dict[str, str]:
    if not ALIASES_FILE.exists():
        return {}
    try:
        data = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load aliases: {e}")
        # Rename the corrupt file aside so the next _save can't
        # overwrite the bytes with a fresh (probably empty) alias
        # dict. Same pattern the persistence stores got
        # (5512834, 98d1c68, 8a86987).
        _back_up_corrupt(e)
        return {}
    if not isinstance(data, dict):
        # JSON parsed cleanly but the top-level shape is wrong —
        # parity with the snippets fix. Callers expect a dict and
        # would crash on .get() / .items().
        log.warning(
            f"Aliases file shape is {type(data).__name__}, expected dict"
        )
        _back_up_corrupt(
            ValueError(f"top-level shape is {type(data).__name__}, expected dict")
        )
        return {}
    return data


def _back_up_corrupt(reason: Exception) -> None:
    from datetime import UTC, datetime
    backup = ALIASES_FILE.with_name(
        f"{ALIASES_FILE.name}.corrupted-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    )
    try:
        ALIASES_FILE.replace(backup)
    except OSError:
        pass


def _save(aliases: dict[str, str]) -> None:
    ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write — see persistence/store.py for rationale.
    tmp = ALIASES_FILE.with_name(ALIASES_FILE.name + ".tmp")
    tmp.write_text(
        json.dumps(aliases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(ALIASES_FILE)


def get_alias(name: str) -> str | None:
    """Look up an alias by name. Returns the prompt template or None."""
    return _load().get(name.lower())


def set_alias(name: str, prompt: str) -> None:
    """Create or update an alias."""
    aliases = _load()
    aliases[name.lower()] = prompt
    _save(aliases)


def remove_alias(name: str) -> bool:
    """Remove an alias. Returns True if it existed."""
    aliases = _load()
    if name.lower() in aliases:
        del aliases[name.lower()]
        _save(aliases)
        return True
    return False


def list_aliases() -> dict[str, str]:
    """Return all aliases."""
    return _load()

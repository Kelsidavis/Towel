"""Prompt aliases — user-defined shortcuts for common prompts.

Aliases are stored in ~/.towel/aliases.json and persist across sessions.
Use /alias to create, /aliases to list, /unalias to remove.

When a user types /myalias some input, the alias prompt is expanded
with the input appended, then sent to the agent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.cli.aliases")

ALIASES_FILE = TOWEL_HOME / "aliases.json"


def _load() -> dict[str, str]:
    if not ALIASES_FILE.exists():
        return {}
    try:
        return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load aliases: {e}")
        return {}


def _save(aliases: dict[str, str]) -> None:
    ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ALIASES_FILE.write_text(
        json.dumps(aliases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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

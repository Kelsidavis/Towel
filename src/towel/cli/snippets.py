"""Snippets — reusable text blocks you can inject into conversations.

Snippets are stored in ~/.towel/snippets.json and persist across sessions.
Unlike aliases (which are prompts sent to the agent), snippets are raw text
inserted into your message — useful for boilerplate, templates, or
frequently-pasted content.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.cli.snippets")

SNIPPETS_FILE = TOWEL_HOME / "snippets.json"


def _load() -> dict[str, str]:
    if not SNIPPETS_FILE.exists():
        return {}
    try:
        return json.loads(SNIPPETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load snippets: {e}")
        return {}


def _save(snippets: dict[str, str]) -> None:
    SNIPPETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SNIPPETS_FILE.write_text(
        json.dumps(snippets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_snippet(name: str) -> str | None:
    return _load().get(name.lower())


def set_snippet(name: str, content: str) -> None:
    snippets = _load()
    snippets[name.lower()] = content
    _save(snippets)


def remove_snippet(name: str) -> bool:
    snippets = _load()
    if name.lower() in snippets:
        del snippets[name.lower()]
        _save(snippets)
        return True
    return False


def list_snippets() -> dict[str, str]:
    return _load()

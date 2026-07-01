"""Snippets — reusable text blocks you can inject into conversations.

Snippets are stored in ~/.towel/snippets.json and persist across sessions.
Unlike aliases (which are prompts sent to the agent), snippets are raw text
inserted into your message — useful for boilerplate, templates, or
frequently-pasted content.
"""

from __future__ import annotations

import json
import logging

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.cli.snippets")

SNIPPETS_FILE = TOWEL_HOME / "snippets.json"


def _load() -> dict[str, str]:
    if not SNIPPETS_FILE.exists():
        return {}
    try:
        data = json.loads(SNIPPETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load snippets: %s", e)
        # Rename the corrupt file aside so the next _save can't
        # overwrite the bytes with a fresh (probably empty) snippet
        # dict. Same pattern the persistence stores got
        # (5512834, 98d1c68, 8a86987).
        _back_up_corrupt(e)
        return {}
    if not isinstance(data, dict):
        # JSON parsed cleanly but the top-level shape is wrong
        # (operator hand-edited to a list, a Python script wrote
        # `json.dump([...])`, etc.). Callers expect a dict and
        # would crash on .get() / .items(); back up the bad file
        # and start fresh so the next /snippets set doesn't get
        # rejected by the same caller-side AttributeError.
        log.warning(
            f"Snippets file shape is {type(data).__name__}, expected dict"
        )
        _back_up_corrupt(
            ValueError(f"top-level shape is {type(data).__name__}, expected dict")
        )
        return {}
    return data


def _back_up_corrupt(reason: Exception) -> None:
    from datetime import UTC, datetime
    backup = SNIPPETS_FILE.with_name(
        f"{SNIPPETS_FILE.name}.corrupted-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    )
    try:
        SNIPPETS_FILE.replace(backup)
    except OSError:
        pass


def _save(snippets: dict[str, str]) -> None:
    SNIPPETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write — see persistence/store.py for rationale.
    tmp = SNIPPETS_FILE.with_name(SNIPPETS_FILE.name + ".tmp")
    tmp.write_text(
        json.dumps(snippets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(SNIPPETS_FILE)


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

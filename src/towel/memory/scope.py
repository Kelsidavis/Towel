"""Project-scope derivation for the memory store.

A memory's ``scope`` field is a free-form string. Empty = global,
non-empty = restricted to callers that pass the same value. This
module supplies a convention for deriving a stable scope from a
project root path so the CLI / runtime can write project-local
memories without the operator having to remember an opaque ID.

Convention: ``proj:<short-name>:<sha8>`` where:

  * short-name is the basename of the project root path, lowercased
    and stripped of non-alphanumeric characters (so the operator can
    read it).
  * sha8 is the first 8 hex chars of sha256(realpath) — disambiguates
    two projects with the same basename, and stays stable across
    `cd`, symlinks resolved, etc.

Project root detection walks up from CWD looking for any of:
  .towel.md, .git, pyproject.toml, package.json, Cargo.toml

The directory that contains the first marker found becomes the root.
When nothing is found, the function returns ``""`` (global) — the
agent is just running in a plain directory, no project context.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_ROOT_MARKERS = (".towel.md", ".git", "pyproject.toml", "package.json", "Cargo.toml")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str, max_len: int = 24) -> str:
    s = _SLUG_RE.sub("-", name.lower()).strip("-")
    return s[:max_len] or "unnamed"


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd) until a project marker hits."""
    cur = (start or Path.cwd()).resolve()
    # Stop at filesystem root.
    for path in [cur, *cur.parents]:
        for marker in _ROOT_MARKERS:
            if (path / marker).exists():
                return path
    return None


def derive_scope(start: Path | None = None) -> str:
    """Return a stable scope string for the project rooted at or above ``start``.

    Empty string when no project is detected — the caller should
    treat that as "use global scope only", which matches the legacy
    behavior of the memory store before scopes existed.
    """
    root = find_project_root(start)
    if root is None:
        return ""
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:8]
    return f"proj:{_slug(root.name)}:{digest}"

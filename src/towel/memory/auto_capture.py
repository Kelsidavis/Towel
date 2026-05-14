"""Heuristic auto-capture — extract memorable facts from user turns.

The agent's memory only grows when something writes to it. The
``remember`` tool covers explicit cases ("remember that I use vim"),
but most useful facts get mentioned in passing without anyone naming
them as memorable. This module runs a small set of conservative regex
extractors over each user message and writes the hits to
``MemoryStore`` automatically.

Design constraints

* **False negatives over false positives.** A missed capture costs
  nothing — the next ``remember`` call (or the user repeating
  themselves) will catch it. A wrong capture pollutes the system
  prompt forever. Patterns are deliberately narrow.
* **Idempotent.** If the target key already exists in the store, the
  extractor backs off — operator-set memories outrank heuristic ones,
  and re-firing the same pattern shouldn't bump update timestamps.
* **Negation-aware.** "I'm not a data scientist" must not capture
  ``role=data scientist``. A simple lookbehind for "not"/"n't" within
  the preceding clause covers the common case.
* **Cheap.** Runs synchronously on every user turn; no LLM call, no
  embedding, no network. Pure regex over ≤ a few KB of text.

The set of patterns is intentionally small to start. The next step
(PR 3b) replaces it with a small-model extractor for cases regex
can't handle (multi-sentence context, paraphrase, indirect mentions).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from towel.memory.store import MemoryStore

log = logging.getLogger("towel.memory.auto_capture")


@dataclass(frozen=True)
class Capture:
    """A single auto-captured fact, before it's written to the store."""

    key: str
    content: str
    memory_type: str  # 'user' | 'preference' | 'project' | 'fact'
    source_pattern: str  # for telemetry / debugging


# Negation markers within the current clause invalidate a match.
# Clause boundaries are sentence punctuation (.!?) or a comma — so
# "I'm not a data scientist, I'm a designer" still captures "designer"
# because the comma resets the negation context.
_NEGATION_RE = re.compile(r"\b(not|n't|never|no longer)\b", re.IGNORECASE)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?,;\n]")


def _is_negated(text: str, match_start: int) -> bool:
    """True if a negation marker appears in the same clause to the left."""
    left = text[:match_start]
    # Trim to the start of the current clause: scan back to the most
    # recent boundary character. Without one, the whole left context
    # is in scope.
    boundary_iter = list(_CLAUSE_BOUNDARY_RE.finditer(left))
    if boundary_iter:
        clause = left[boundary_iter[-1].end() :]
    else:
        clause = left
    return bool(_NEGATION_RE.search(clause))


# Each pattern is a tuple of (compiled regex, key-builder, content-builder,
# memory_type, label). Patterns must be:
#   * anchored on at least one strong cue word ("I'm", "my", "remember") so
#     they don't fire on arbitrary user text;
#   * length-bounded on the captured payload so we don't store paragraphs.
_PATTERNS: list[tuple[re.Pattern[str], str, str, str, str]] = [
    # "I'm a senior backend engineer", "I am an SRE"
    (
        re.compile(
            r"\bI(?:'m| am)\s+an?\s+"
            r"(?P<value>[A-Za-z][A-Za-z\s/+\-]{2,40}?)"
            r"(?=[.,!?\n]|\s+(?:and|but|so|who|that|working|based)\b|$)",
            re.IGNORECASE,
        ),
        "role",
        "{value}",
        "user",
        "role",
    ),
    # "I work at OpenAI", "I work for Anthropic"
    (
        re.compile(
            r"\bI work (?:at|for)\s+"
            r"(?P<value>[A-Z][\w.&'-]+(?:\s+[\w.&'-]+){0,3})",
        ),
        "employer",
        "{value}",
        "user",
        "employer",
    ),
    # "I prefer short replies", "I like terse output", "I want concise answers"
    # Multiple preferences can coexist — key is derived from the verb+head.
    (
        re.compile(
            r"\bI (?:prefer|like|want)\s+"
            r"(?P<value>[a-z][^.!?\n]{3,80}?)"
            r"(?=[.,!?\n]|$)",
            re.IGNORECASE,
        ),
        "preference_{slug}",
        "{value}",
        "preference",
        "preference",
    ),
    # "my project is X", "I'm building X", "we're building X"
    (
        re.compile(
            r"\b(?:my (?:project|side project) is|I'm building|we're building)\s+"
            r"(?P<value>[A-Za-z0-9][^.!?\n]{2,80}?)"
            r"(?=[.,!?\n]|$)",
            re.IGNORECASE,
        ),
        "current_project",
        "{value}",
        "project",
        "project",
    ),
    # "we ship by March 15", "deadline is next Friday", "due by EOQ"
    (
        re.compile(
            r"\b(?:we (?:ship|launch|deploy|release|cut over)|deadline (?:is|of)|due (?:by|on))\s+"
            r"(?P<value>[A-Za-z0-9][^.!?\n]{2,40}?)"
            r"(?=[.,!?\n]|$)",
            re.IGNORECASE,
        ),
        "deadline",
        "{value}",
        "project",
        "deadline",
    ),
    # "remember that X", "remember: X" — explicit user signal.
    (
        re.compile(
            r"\bremember(?:\s+that|:|\s+this:)\s+"
            r"(?P<value>[A-Za-z0-9][^.!?\n]{4,200}?)"
            r"(?=[.!?\n]|$)",
            re.IGNORECASE,
        ),
        "fact_{slug}",
        "{value}",
        "fact",
        "explicit-remember",
    ),
]


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 32) -> str:
    """Turn a content string into a stable, short, key-safe identifier."""
    s = _SLUG_RE.sub("_", text.lower()).strip("_")
    return s[:max_len] or "unknown"


def extract(text: str) -> list[Capture]:
    """Run all patterns over ``text`` and return the resulting captures.

    Stateless — every call sees only the user text. Callers (see
    ``apply``) handle dedup against the existing store.
    """
    if not text:
        return []
    out: list[Capture] = []
    for pattern, key_tmpl, content_tmpl, mtype, label in _PATTERNS:
        for m in pattern.finditer(text):
            if _is_negated(text, m.start()):
                log.debug("Skipping negated capture (%s) at %d", label, m.start())
                continue
            value = m.group("value").strip().rstrip(".,!?;:")
            if not value:
                continue
            key = key_tmpl.format(value=value, slug=_slug(value))
            content = content_tmpl.format(value=value)
            out.append(
                Capture(
                    key=key,
                    content=content,
                    memory_type=mtype,
                    source_pattern=label,
                )
            )
    return out


def apply(
    text: str, store: MemoryStore, *, overwrite: bool = False
) -> list[Capture]:
    """Extract captures from ``text`` and write the new ones to ``store``.

    Returns the captures that were actually written. By default,
    ``overwrite=False`` means existing keys are left alone — operator-
    set memories and prior auto-captures outrank fresh heuristic hits.
    Pass ``overwrite=True`` if you specifically want the heuristic to
    refresh values on every turn (rarely the right call).
    """
    captures = extract(text)
    if not captures:
        return []
    written: list[Capture] = []
    for cap in captures:
        existing = store.recall(cap.key)
        if existing is not None and not overwrite:
            continue
        # Tag the source so memory stats / tidy can distinguish
        # heuristic captures from operator-set entries.
        store.remember(
            cap.key,
            cap.content,
            memory_type=cap.memory_type,
            source=f"auto_capture:{cap.source_pattern}",
        )
        log.info(
            "Auto-captured %s=%r via pattern=%s",
            cap.key, cap.content, cap.source_pattern,
        )
        written.append(cap)
    return written

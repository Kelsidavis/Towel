"""Persistent memory store — the agent's long-term brain.

Memories persist across sessions in ~/.towel/memory/memory.db, a single
SQLite file. Each entry is a key/content/type triple plus retrieval
metadata (created_at, updated_at, last_recalled_at, recall_count). A
``memories_fts`` virtual table (FTS5) shadows ``content`` so search()
can rank by BM25 instead of by substring match.

Memory types:
  - user:      Facts about the user (role, preferences, expertise)
  - project:   Ongoing work, goals, deadlines
  - fact:      Learned facts the agent should remember
  - preference: How the user likes things done

The store auto-migrates from the previous ``memories.json`` format on
first open and renames the old file to ``memories.json.migrated-<ts>``
so the import never runs twice. Atomic writes come from SQLite's WAL
journal, replacing the tmp-file rename dance the JSON store used.

Retrieval:
  - ``recall(key)``                 — exact lookup by key
  - ``recall_all(type=None)``       — full scan, filter by type
  - ``search(query, limit=5)``      — BM25 over content via FTS5;
                                      falls back to substring scan when
                                      FTS5 produces zero rows (single
                                      keyword + tiny corpus)
  - ``to_prompt_block(query=None)`` — system-prompt block. When ``query``
                                      is set, returns top-K relevant
                                      entries and bumps recall stats;
                                      otherwise behaves like the legacy
                                      "dump everything" path
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.memory")

DEFAULT_MEMORY_DIR = TOWEL_HOME / "memory"

MEMORY_TYPES = ("user", "project", "fact", "preference")

# Number of memories returned by query-relevant retrieval when injecting
# into a system prompt. AgentMemory's benchmarks land at R@5 ≥ 95% with
# BM25-only on small corpora; 8 leaves headroom for redundancy across
# memory types without exploding token cost.
_DEFAULT_PROMPT_LIMIT = 8

# Memory types that auto_forget never touches without explicit override.
# Operator-set identity/preferences should outlive heuristic pruning;
# only "fact" memories are eligible by default. "project" is preserved
# too — project facts (deadlines, current work) decay naturally as
# ``remember`` overwrites them, no need to prune.
_PROTECTED_TYPES = ("user", "preference", "project")


@dataclass
class MemoryEntry:
    """A single memory."""

    key: str
    content: str
    memory_type: str = "fact"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_recalled_at: datetime | None = None
    recall_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "content": self.content,
            "type": self.memory_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "recall_count": self.recall_count,
        }
        if self.last_recalled_at is not None:
            out["last_recalled_at"] = self.last_recalled_at.isoformat()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        last = data.get("last_recalled_at")
        return cls(
            key=data["key"],
            content=data["content"],
            memory_type=data.get("type", "fact"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_recalled_at=datetime.fromisoformat(last) if last else None,
            recall_count=int(data.get("recall_count", 0)),
        )

    def __str__(self) -> str:
        return f"[{self.memory_type}] {self.key}: {self.content}"


class MemoryStoreError(RuntimeError):
    """Raised when the store cannot be opened (e.g. SQLite without FTS5)."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    key              TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    memory_type      TEXT NOT NULL DEFAULT 'fact',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_recalled_at TEXT,
    recall_count     INTEGER NOT NULL DEFAULT 0,
    -- Reserved for the embeddings follow-up. Nullable so PR 1 doesn't
    -- need to populate it; PR 2 fills it in lazily on first recall.
    embedding        BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(content, content='memories', content_rowid='rowid', tokenize='porter unicode61');

-- Keep the FTS index in sync without manual reindexing. Triggers fire
-- inside the same transaction as the row mutation, so a crash mid-write
-- cannot leave the FTS shadow out of step with the source table.
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    last = row["last_recalled_at"]
    return MemoryEntry(
        key=row["key"],
        content=row["content"],
        memory_type=row["memory_type"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_recalled_at=datetime.fromisoformat(last) if last else None,
        recall_count=int(row["recall_count"]),
    )


class MemoryStore:
    """SQLite-backed persistent memory store with FTS5 BM25 ranking."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or DEFAULT_MEMORY_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.store_dir / "memory.db"
        self._json_path = self.store_dir / "memories.json"
        self._init_db()
        self._migrate_from_json()

    # ── connection management ─────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        # ``isolation_level=None`` plus explicit BEGIN/COMMIT gives us
        # control over transactions and avoids Python's autocommit
        # surprises. WAL allows concurrent reads while one writer is
        # active — handy for the cluster sync path that mutates from
        # multiple coroutines.
        con = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,
            timeout=5.0,
        )
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=5000")
        return con

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        con = self._connect()
        try:
            con.execute("BEGIN")
            yield con
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        con = self._connect()
        try:
            # FTS5 may be missing on a stripped-down sqlite build. Fail
            # loudly with remediation rather than silently degrading —
            # tests are easier to write and operators get a clear signal.
            try:
                con.executescript(_SCHEMA)
            except sqlite3.OperationalError as exc:
                raise MemoryStoreError(
                    f"SQLite is missing FTS5 support ({exc}). Install a "
                    "sqlite with FTS5 enabled (Python's stdlib build "
                    "ships it on macOS Homebrew and Debian 11+)."
                ) from exc
        finally:
            con.close()

    # ── migration from the legacy JSON store ──────────────────────────

    def _migrate_from_json(self) -> None:
        """Import ~/.towel/memory/memories.json into SQLite once.

        The marker for "already migrated" is the rename: after a
        successful import we move the JSON file to
        ``memories.json.migrated-<utc-timestamp>``. If the rename fails
        (read-only fs, permissions) we leave the JSON in place but skip
        re-import on future opens by checking whether the SQLite store
        is already non-empty for the same keys.
        """
        if not self._json_path.exists():
            return
        try:
            raw = self._json_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "Skipping JSON migration: cannot read %s (%s)",
                self._json_path, exc,
            )
            return
        if not isinstance(data, dict):
            log.warning(
                "Skipping JSON migration: %s is not a dict", self._json_path
            )
            return
        imported = 0
        try:
            with self._txn() as con:
                for key, blob in data.items():
                    if not isinstance(blob, dict):
                        continue
                    try:
                        entry = MemoryEntry.from_dict({"key": key, **blob})
                    except (KeyError, ValueError) as exc:
                        log.warning(
                            "Skipping malformed entry %r during migration: %s",
                            key, exc,
                        )
                        continue
                    # INSERT OR IGNORE: if the operator has already
                    # written to the SQLite store, we don't trample their
                    # newer values with stale JSON ones.
                    con.execute(
                        "INSERT OR IGNORE INTO memories "
                        "(key, content, memory_type, created_at, updated_at, "
                        " last_recalled_at, recall_count) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            entry.key,
                            entry.content,
                            entry.memory_type,
                            entry.created_at.isoformat(),
                            entry.updated_at.isoformat(),
                            entry.last_recalled_at.isoformat() if entry.last_recalled_at else None,
                            entry.recall_count,
                        ),
                    )
                    imported += 1
        except sqlite3.Error as exc:
            log.warning("JSON migration failed: %s", exc)
            return
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        target = self._json_path.with_name(
            f"{self._json_path.name}.migrated-{ts}"
        )
        try:
            self._json_path.replace(target)
            log.info(
                "Migrated %d memories from %s to %s; old file at %s",
                imported, self._json_path.name, self._db_path.name, target,
            )
        except OSError as exc:
            log.warning(
                "Migrated %d memories from %s but could not rename: %s",
                imported, self._json_path, exc,
            )

    # ── CRUD ──────────────────────────────────────────────────────────

    def remember(
        self, key: str, content: str, memory_type: str = "fact"
    ) -> MemoryEntry:
        """Store or update a memory."""
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        with self._txn() as con:
            row = con.execute(
                "SELECT created_at FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO memories "
                    "(key, content, memory_type, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, content, memory_type, now_iso, now_iso),
                )
                created_at = now
            else:
                con.execute(
                    "UPDATE memories SET content = ?, memory_type = ?, "
                    "updated_at = ? WHERE key = ?",
                    (content, memory_type, now_iso, key),
                )
                created_at = datetime.fromisoformat(row["created_at"])
        log.info("Remembered: %s", key)
        return MemoryEntry(
            key=key,
            content=content,
            memory_type=memory_type,
            created_at=created_at,
            updated_at=now,
        )

    def forget(self, key: str) -> bool:
        """Remove a memory. Returns True if it existed."""
        with self._txn() as con:
            cur = con.execute("DELETE FROM memories WHERE key = ?", (key,))
            removed = cur.rowcount > 0
        if removed:
            log.info("Forgot: %s", key)
        return removed

    def recall(self, key: str) -> MemoryEntry | None:
        """Get a specific memory by key."""
        con = self._connect()
        try:
            row = con.execute(
                "SELECT * FROM memories WHERE key = ?", (key,)
            ).fetchone()
        finally:
            con.close()
        return _row_to_entry(row) if row is not None else None

    def recall_all(self, memory_type: str | None = None) -> list[MemoryEntry]:
        """Get all memories, optionally filtered by type."""
        con = self._connect()
        try:
            if memory_type:
                rows = con.execute(
                    "SELECT * FROM memories WHERE memory_type = ? "
                    "ORDER BY updated_at DESC",
                    (memory_type,),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC"
                ).fetchall()
        finally:
            con.close()
        return [_row_to_entry(r) for r in rows]

    # ── retrieval ─────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """BM25-ranked search over memory content.

        Returns up to ``limit`` entries ordered by FTS5 rank (most
        relevant first). When the FTS5 query parses to zero rows — most
        often because the user typed a stopword or a substring that FTS5
        treats as a token boundary — falls back to a case-insensitive
        substring scan so simple "what's my role" lookups still work.

        Does NOT bump recall stats; that's the caller's job (typically
        ``to_prompt_block``) so internal callers like ``towel memory
        search`` don't inflate the counters.
        """
        query = (query or "").strip()
        if not query:
            return []
        con = self._connect()
        try:
            fts_query = _fts5_query(query)
            rows: list[sqlite3.Row] = []
            if fts_query:
                try:
                    rows = con.execute(
                        "SELECT m.* FROM memories_fts f "
                        "JOIN memories m ON m.rowid = f.rowid "
                        "WHERE f.memories_fts MATCH ? "
                        "ORDER BY f.rank LIMIT ?",
                        (fts_query, limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    # Bad FTS5 query syntax (rare given _fts5_query
                    # quoting). Fall through to substring.
                    log.debug("FTS5 query failed (%s); falling back", exc)
            if not rows:
                like = f"%{query.lower()}%"
                rows = con.execute(
                    "SELECT * FROM memories "
                    "WHERE lower(key) LIKE ? OR lower(content) LIKE ? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (like, like, limit),
                ).fetchall()
        finally:
            con.close()
        return [_row_to_entry(r) for r in rows]

    def _bump_recall(self, keys: list[str]) -> None:
        """Mark a set of memories as recently retrieved."""
        if not keys:
            return
        now_iso = datetime.now(UTC).isoformat()
        with self._txn() as con:
            con.executemany(
                "UPDATE memories SET recall_count = recall_count + 1, "
                "last_recalled_at = ? WHERE key = ?",
                [(now_iso, k) for k in keys],
            )

    def to_prompt_block(
        self,
        query: str | None = None,
        limit: int = _DEFAULT_PROMPT_LIMIT,
    ) -> str:
        """Render memories as a system-prompt block.

        Two modes:

        - ``query=None`` (legacy): dump every memory grouped by type.
          Used by tooling that wants the full picture (``towel memory
          list``, the TUI's debug view). Tokens scale with the corpus.
        - ``query="..."`` (per-turn): return up to ``limit`` entries
          ranked by relevance to ``query``. Bumps recall stats on the
          returned entries so the eventual decay pass knows what's
          actually useful.

        Either way, returns ``""`` when the corpus is empty so callers
        can concatenate unconditionally.
        """
        if query is None:
            entries = self.recall_all()
        else:
            entries = self.search(query, limit=limit)
            if not entries:
                # Empty FTS hit AND empty substring hit — fall back to a
                # short recent slice so the agent still knows who it's
                # talking to. Cap at limit so we don't blow the budget
                # for the unsupervised case.
                con = self._connect()
                try:
                    rows = con.execute(
                        "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                finally:
                    con.close()
                entries = [_row_to_entry(r) for r in rows]
            self._bump_recall([e.key for e in entries])

        if not entries:
            return ""

        lines = [
            "\n\n## Your Memory\nYou have the following persistent memories from past sessions:\n"
        ]
        by_type: dict[str, list[MemoryEntry]] = {}
        for e in entries:
            by_type.setdefault(e.memory_type, []).append(e)
        for mtype in ["user", "preference", "project", "fact"]:
            group = by_type.get(mtype, [])
            if not group:
                continue
            lines.append(f"\n**{mtype.title()}:**")
            for e in group:
                lines.append(f"- {e.key}: {e.content}")
        lines.append(
            "\nYou can use the `remember` and `forget` tools to update your memory. "
            "Proactively remember useful facts about the user and their work."
        )
        return "\n".join(lines)

    # ── decay / pruning ───────────────────────────────────────────────

    def auto_forget(
        self,
        max_age_days: float = 90.0,
        *,
        dry_run: bool = False,
        protected_types: tuple[str, ...] = _PROTECTED_TYPES,
    ) -> list[MemoryEntry]:
        """Prune stale, never-recalled memories.

        An entry is eligible iff ALL of:

        * its memory_type is not in ``protected_types`` (default:
          user, preference, project — only ``fact`` is pruneable);
        * it has never been recalled (``recall_count == 0``);
        * its ``updated_at`` is older than ``max_age_days``.

        Conservative on purpose. The decay model in ``salience()`` is
        richer than this rule and is exposed for callers that want to
        rank — auto_forget is the safe default that never throws away
        a memory the agent has actually used.

        ``dry_run=True`` returns the would-be-deleted entries without
        touching the DB. Useful for ``towel memory tidy --dry-run``.

        Returns the list of entries that were (or would be) deleted.
        """
        cutoff = (datetime.now(UTC).timestamp() - max_age_days * 86400.0)
        cutoff_iso = datetime.fromtimestamp(cutoff, UTC).isoformat()
        protected = list(protected_types)
        placeholders = ",".join(["?"] * len(protected)) if protected else "''"
        sql_where = (
            f"recall_count = 0 AND updated_at < ? AND memory_type NOT IN ({placeholders})"
        )
        con = self._connect()
        try:
            rows = con.execute(
                f"SELECT * FROM memories WHERE {sql_where}",
                (cutoff_iso, *protected),
            ).fetchall()
        finally:
            con.close()
        victims = [_row_to_entry(r) for r in rows]
        if dry_run or not victims:
            return victims
        with self._txn() as con:
            con.executemany(
                "DELETE FROM memories WHERE key = ?",
                [(v.key,) for v in victims],
            )
        log.info("auto_forget pruned %d memor(ies)", len(victims))
        return victims

    def rank_by_salience(self) -> list[tuple[MemoryEntry, float]]:
        """Return every entry with its salience score, lowest first.

        Used by ``towel memory tidy`` to show the operator which
        memories are weakest before pruning. The bottom of the list is
        the auto_forget candidate pool.
        """
        now = datetime.now(UTC)
        scored = [(e, salience(e, now)) for e in self.recall_all()]
        scored.sort(key=lambda pair: pair[1])
        return scored

    @property
    def count(self) -> int:
        con = self._connect()
        try:
            row = con.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        finally:
            con.close()
        return int(row["n"])


# ── decay / auto-forget ───────────────────────────────────────────────


def salience(entry: MemoryEntry, now: datetime | None = None) -> float:
    """Score an entry for pruning order. Higher = more worth keeping.

    Combines three signals so a single dimension can't dominate:

    * **recall_count** (log-scaled) — a memory recalled 10× is more
      valuable than one recalled once, but not 10× more; log keeps
      runaway recall counts from outranking everything else.
    * **age** — older entries decay exponentially with a 60-day
      half-life. A brand-new memory has full score; one from a year
      ago is ~1.5% of its original recency score.
    * **last_recalled_at** — same half-life applied to "time since
      last touched"; if never recalled, contributes 0.

    Pure function, side-effect free; callers compose it any way they
    like (the default ``auto_forget`` uses ``salience < threshold``
    on the bottom-N).
    """
    import math

    now = now or datetime.now(UTC)
    half_life_days = 60.0
    age_days = max(0.0, (now - entry.updated_at).total_seconds() / 86400.0)
    recency = math.exp(-age_days / half_life_days)
    recall = math.log1p(entry.recall_count)
    if entry.last_recalled_at is not None:
        since_days = max(0.0, (now - entry.last_recalled_at).total_seconds() / 86400.0)
        access_recency = math.exp(-since_days / half_life_days)
    else:
        access_recency = 0.0
    # Weights chosen so an entry recalled once recently beats an old
    # untouched entry, and an entry recalled many times survives even
    # if its updated_at is old.
    return 2.0 * recall + 1.0 * recency + 1.0 * access_recency


# FTS5's MATCH grammar treats bare strings as one of: column filters,
# prefix queries, or AND of bare tokens. User strings often contain
# characters that confuse the parser (apostrophes, hyphens). Quoting
# each token and joining with OR gives us a forgiving "any token hits"
# query that still benefits from BM25 ranking.
def _fts5_query(raw: str) -> str:
    """Turn a free-form string into a safe FTS5 MATCH expression."""
    tokens = []
    for word in raw.split():
        # Strip non-alphanumeric noise; FTS5 tokenizer already drops
        # most punctuation but quoted phrases preserve it, which causes
        # 'foo!' to never match 'foo'. Be conservative.
        cleaned = "".join(ch for ch in word if ch.isalnum() or ch in "-_")
        if not cleaned:
            continue
        tokens.append(f'"{cleaned}"')
    return " OR ".join(tokens)

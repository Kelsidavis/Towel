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
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    # Where this memory came from. "" / None means operator-set (the
    # remember() tool, CLI, or the legacy JSON import); a non-empty
    # value comes from auto_capture and names the pattern that fired.
    # Lets stats show heuristic vs deliberate, and lets tidy be picky
    # about which sources it prunes.
    source: str = ""
    # Free-form labels beyond memory_type, for operator-defined
    # grouping (e.g. project name, sensitivity). De-duplicated and
    # order-preserving — first-add wins.
    tags: list[str] = field(default_factory=list)
    # Project scope this memory belongs to. "" = global (visible to
    # every project), non-empty = restricted to callers that pass
    # the same scope value. Derived conventionally from project root
    # via towel.memory.scope.derive_scope(cwd).
    scope: str = ""

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
        if self.source:
            out["source"] = self.source
        if self.tags:
            out["tags"] = list(self.tags)
        if self.scope:
            out["scope"] = self.scope
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        last = data.get("last_recalled_at")
        raw_tags = data.get("tags") or []
        if not isinstance(raw_tags, list):
            raw_tags = []
        return cls(
            key=data["key"],
            content=data["content"],
            memory_type=data.get("type", "fact"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_recalled_at=datetime.fromisoformat(last) if last else None,
            recall_count=int(data.get("recall_count", 0)),
            source=data.get("source", ""),
            tags=[str(t) for t in raw_tags if isinstance(t, str)],
            scope=str(data.get("scope") or ""),
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
    embedding        BLOB,
    -- Capture source: "" / NULL = operator-set (remember tool, CLI,
    -- legacy JSON import); non-empty = auto_capture pattern label.
    source           TEXT NOT NULL DEFAULT '',
    -- Free-form tags as a JSON array of strings. Lets the operator
    -- group memories along axes the four fixed memory_types can't
    -- express (e.g. project-name, topic, sensitivity). Searched via
    -- LIKE on the JSON text — fine for small corpora; if this grows
    -- past tens of thousands of memories, swap for a tags table.
    tags             TEXT NOT NULL DEFAULT '[]',
    -- Scope for project-specific memories. "" / NULL means global —
    -- visible regardless of which project the agent is operating
    -- in. A non-empty value (conventionally derived from the project
    -- root path) restricts visibility to callers that explicitly
    -- ask for that scope, or to retrieval calls that pass it as the
    -- current scope alongside global.
    scope            TEXT NOT NULL DEFAULT ''
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(content, content='memories', content_rowid='rowid', tokenize='porter unicode61');

-- Co-retrieval graph. Every time the prompt-block builder pulls a
-- batch of memories for a query, each pair gets a +1 here; over time
-- it learns which memories tend to be relevant together ("user is
-- engineer" + "engineer uses vim") without an LLM call. The FK
-- cascade means forget() cleans up edges for free.
CREATE TABLE IF NOT EXISTS memory_links (
    source_key  TEXT NOT NULL,
    target_key  TEXT NOT NULL,
    weight      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source_key, target_key),
    FOREIGN KEY (source_key) REFERENCES memories(key) ON DELETE CASCADE,
    FOREIGN KEY (target_key) REFERENCES memories(key) ON DELETE CASCADE,
    -- A memory can't link to itself.
    CHECK (source_key <> target_key)
);
CREATE INDEX IF NOT EXISTS memory_links_target ON memory_links(target_key);
CREATE INDEX IF NOT EXISTS memory_links_weight ON memory_links(weight DESC);

-- Per-query recall trail. Every time to_prompt_block(query=...) runs
-- we record what was asked and which entries surfaced, so an
-- operator can answer "why did the agent remember X when I asked Y?"
-- without re-running retrieval. Capped via auto-prune in
-- record_recall(); the index on ts speeds the time-window queries
-- the CLI / endpoint use.
CREATE TABLE IF NOT EXISTS recall_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    query   TEXT NOT NULL,
    -- JSON array of the keys returned, in rank order. Storing keys
    -- rather than rowids means a forget()'d memory still leaves the
    -- log entry interpretable ("returned X, which has since been
    -- forgotten") — useful for debugging churn.
    keys    TEXT NOT NULL,
    -- Optional scope the query ran under. Empty = no scope filter
    -- (or the store had default_scope="").
    scope   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS recall_log_ts ON recall_log(ts);

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


def open_for_config(config: Any) -> MemoryStore:
    """Construct a MemoryStore wired from a TowelConfig.

    Centralizes the "scope + cap + future knobs" wiring so the five
    runtime construction sites (serve, worker, chat, mcp, ask) don't
    each have to repeat the config plumbing. Falls back to defaults
    when the relevant attributes don't exist — keeps this importable
    from non-runtime contexts that pass a stub config.
    """
    from towel.memory.scope import derive_scope

    store = MemoryStore(default_scope=derive_scope())
    cap = getattr(config, "memory_recall_log_cap", None)
    if isinstance(cap, int) and cap > 0:
        store.RECALL_LOG_CAP = cap
    return store


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    last = row["last_recalled_at"]
    # Older rows may predate the `source` or `tags` columns —
    # sqlite.Row treats missing columns as KeyError on name access,
    # so use cautious lookups. Newly-created DBs always have them.
    try:
        source = row["source"] or ""
    except (IndexError, KeyError):
        source = ""
    try:
        raw_tags = row["tags"] or "[]"
        tags_list = json.loads(raw_tags) if raw_tags else []
        if not isinstance(tags_list, list):
            tags_list = []
        tags = [str(t) for t in tags_list if isinstance(t, str)]
    except (IndexError, KeyError, json.JSONDecodeError):
        tags = []
    try:
        scope = row["scope"] or ""
    except (IndexError, KeyError):
        scope = ""
    return MemoryEntry(
        key=row["key"],
        content=row["content"],
        memory_type=row["memory_type"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_recalled_at=datetime.fromisoformat(last) if last else None,
        recall_count=int(row["recall_count"]),
        source=source,
        tags=tags,
        scope=scope,
    )


class MemoryStore:
    """SQLite-backed persistent memory store with FTS5 BM25 ranking.

    ``default_scope`` is the scope a remember() call lands in when no
    explicit scope is passed, AND the project-side scope retrieval
    methods OR with global. Set it once per process (e.g. at runtime
    init) so the operator doesn't have to specify scope on every
    write; pass scope="" explicitly to opt out for a particular call.
    """

    def __init__(
        self,
        store_dir: Path | None = None,
        default_scope: str = "",
    ) -> None:
        self.store_dir = store_dir or DEFAULT_MEMORY_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.store_dir / "memory.db"
        self._json_path = self.store_dir / "memories.json"
        self.default_scope = default_scope
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
            # Idempotent column additions for stores that predate them.
            # ADD COLUMN with a NOT NULL default backfills existing rows
            # to the default in modern SQLite, so this is safe to run on
            # every open. Wrapped in try/except in case the column is
            # already there (older sqlite raises rather than no-op).
            for ddl in (
                "ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
                "ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT ''",
            ):
                try:
                    con.execute(ddl)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
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

    _BAD_KEY = re.compile(r"\.\./|/\.\.|^/|\\")

    def remember(
        self,
        key: str,
        content: str,
        memory_type: str = "fact",
        *,
        source: str = "",
        tags: list[str] | None = None,
        scope: str | None = None,
    ) -> MemoryEntry:
        """Store or update a memory.

        ``source`` is opaque to the store but conventionally identifies
        the writer: ``""`` for operator-driven writes (remember tool,
        CLI, JSON import), or a pattern name like ``"auto_capture:role"``
        for heuristic captures. Stored on inserts only — updating an
        existing memory leaves its original source intact so we don't
        lose provenance of operator-set entries that get re-touched.
        """
        if self._BAD_KEY.search(key or ""):
            raise ValueError(
                f"Memory key {key!r} contains path traversal sequences. "
                "Keys must be plain identifiers."
            )
        now = datetime.now(UTC)
        now_iso = now.isoformat()
        # Compute the embedding outside the transaction — it can take
        # ~5ms for short strings, longer on cold start, and we don't
        # want to hold a write lock that long. None when the extras
        # aren't installed; the column just stays NULL.
        from towel.memory import embeddings as _emb
        embedding_blob = _emb.encode(content)
        # Normalize tags: drop dupes, strip whitespace, drop empties.
        # Order-preserving dedupe so the operator's first add wins.
        if tags:
            seen: set[str] = set()
            tags_norm: list[str] = []
            for t in tags:
                t = str(t).strip()
                if t and t not in seen:
                    seen.add(t)
                    tags_norm.append(t)
        else:
            tags_norm = []
        # Resolve scope: explicit param > store default > empty.
        effective_scope = scope if scope is not None else self.default_scope
        with self._txn() as con:
            row = con.execute(
                "SELECT created_at, source, tags, scope FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO memories "
                    "(key, content, memory_type, created_at, updated_at, "
                    "source, embedding, tags, scope) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        key, content, memory_type, now_iso, now_iso, source,
                        embedding_blob, json.dumps(tags_norm), effective_scope,
                    ),
                )
                created_at = now
                stored_source = source
                stored_tags = tags_norm
                stored_scope = effective_scope
            else:
                # Existing-row update: merge incoming tags with the
                # stored set so callers don't have to read-modify-
                # write to add a tag, but a tags=None call leaves the
                # existing list untouched. Pass tags=[] to clear.
                try:
                    existing_tags_json = row["tags"] or "[]"
                except (IndexError, KeyError):
                    existing_tags_json = "[]"
                try:
                    existing_tags = json.loads(existing_tags_json)
                except json.JSONDecodeError:
                    existing_tags = []
                if tags is None:
                    merged = existing_tags
                else:
                    seen2: set[str] = set()
                    merged = []
                    for t in [*existing_tags, *tags_norm]:
                        if t not in seen2:
                            seen2.add(t)
                            merged.append(t)
                # Keep the existing scope on update — operators picking
                # up an entry that was project-scoped shouldn't have
                # it silently flipped to global by a default-scope
                # remember(). Explicit scope= still overrides.
                try:
                    existing_scope = row["scope"] or ""
                except (IndexError, KeyError):
                    existing_scope = ""
                stored_scope = scope if scope is not None else existing_scope
                con.execute(
                    "UPDATE memories SET content = ?, memory_type = ?, "
                    "updated_at = ?, embedding = ?, tags = ?, scope = ? WHERE key = ?",
                    (
                        content, memory_type, now_iso, embedding_blob,
                        json.dumps(merged), stored_scope, key,
                    ),
                )
                created_at = datetime.fromisoformat(row["created_at"])
                try:
                    stored_source = row["source"] or ""
                except (IndexError, KeyError):
                    stored_source = ""
                stored_tags = merged
        log.info("Remembered: %s", key)
        return MemoryEntry(
            key=key,
            content=content,
            memory_type=memory_type,
            created_at=created_at,
            updated_at=now,
            source=stored_source,
            tags=stored_tags,
            scope=stored_scope,
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

    def _scope_filter(
        self, scope: str | None
    ) -> tuple[str, list[Any]]:
        """Build a SQL WHERE-fragment + params for the requested scope.

        Three behaviors:

        * ``scope is None``     — include the store's default_scope
                                  PLUS global ("") so callers in a
                                  project see their project memories
                                  alongside universal facts.
        * ``scope=""``          — only global memories.
        * ``scope="X"`` (non-empty) — only memories tagged with that scope.

        Returns ("", []) when no filter should be applied (i.e. when
        default_scope is empty AND scope is None — equivalent to
        "show everything", which matches the pre-scoping default).
        """
        if scope is None:
            ds = self.default_scope
            if not ds:
                return "", []
            return "scope IN (?, '')", [ds]
        if scope == "":
            return "scope = ''", []
        return "scope = ?", [scope]

    def recall_all(
        self,
        memory_type: str | None = None,
        tag: str | None = None,
        scope: str | None = None,
    ) -> list[MemoryEntry]:
        """Get all memories, optionally filtered by type, tag, and scope.

        Tag filtering is a substring match against the JSON-encoded
        tags column then re-checked in Python — cheap for small
        corpora and avoids needing a real array-contains operator on
        sqlite. Larger stores can swap for a separate tags table
        without changing the public signature.

        ``scope=None`` (default): apply the store's default_scope OR
        global. Pass ``scope="x"`` to filter exactly, ``scope=""`` for
        global only.
        """
        con = self._connect()
        try:
            clauses = []
            params: list[Any] = []
            if memory_type:
                clauses.append("memory_type = ?")
                params.append(memory_type)
            if tag:
                # LIKE pre-filter to reduce Python-side scanning.
                clauses.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            scope_sql, scope_params = self._scope_filter(scope)
            if scope_sql:
                clauses.append(scope_sql)
                params.extend(scope_params)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = con.execute(
                f"SELECT * FROM memories {where} ORDER BY updated_at DESC",
                params,
            ).fetchall()
        finally:
            con.close()
        entries = [_row_to_entry(r) for r in rows]
        if tag:
            # Re-confirm in Python in case the LIKE matched on a tag
            # substring (e.g. "work" matches "homework"). Tags are
            # short strings; this scan is fine for any realistic
            # corpus size.
            entries = [e for e in entries if tag in e.tags]
        return entries

    # ── retrieval ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 5,
        tag: str | None = None,
        scope: str | None = None,
    ) -> list[MemoryEntry]:
        """BM25-ranked search over memory content.

        Returns up to ``limit`` entries ordered by FTS5 rank (most
        relevant first). When the FTS5 query parses to zero rows — most
        often because the user typed a stopword or a substring that FTS5
        treats as a token boundary — falls back to a case-insensitive
        substring scan so simple "what's my role" lookups still work.

        ``tag``, if set, restricts the result set to entries carrying
        that exact tag — applied as a post-filter so BM25 ranking is
        preserved within the tagged subset.

        Does NOT bump recall stats; that's the caller's job (typically
        ``to_prompt_block``) so internal callers like ``towel memory
        search`` don't inflate the counters.
        """
        query = (query or "").strip()
        if not query:
            return []
        # SQLite's LIKE pattern length limit (SQLITE_MAX_LIKE_PATTERN_LENGTH,
        # default 50000) raises "LIKE or GLOB pattern too complex" when the
        # pattern overflows. A user message pasted into /api/ask (a long
        # log dump, a code paste, etc.) flowed into to_prompt_block →
        # fused_search → search() with the full body as the query, and
        # the LIKE substring fallback then crashed the request with HTTP
        # 500. Cap the LIKE-bound query at well under the SQLite limit;
        # the BM25/FTS5 path handles long queries fine and gets the full
        # text via its tokenization.
        max_like_query = 2000
        like_query = query if len(query) <= max_like_query else query[:max_like_query]
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
                like = f"%{like_query.lower()}%"
                rows = con.execute(
                    "SELECT * FROM memories "
                    "WHERE lower(key) LIKE ? OR lower(content) LIKE ? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (like, like, limit),
                ).fetchall()
        finally:
            con.close()
        entries = [_row_to_entry(r) for r in rows]
        if tag:
            entries = [e for e in entries if tag in e.tags]
        # Scope filter applied in Python so we keep BM25 ranking intact.
        # Cheap on the size of `limit` (≤ tens).
        if scope is None:
            ds = self.default_scope
            if ds:
                entries = [e for e in entries if e.scope in (ds, "")]
        elif scope == "":
            entries = [e for e in entries if e.scope == ""]
        else:
            entries = [e for e in entries if e.scope == scope]
        return entries

    def add_tag(self, key: str, tag: str) -> bool:
        """Append a tag to an existing memory. Returns True on a real change.

        No-op when the tag is already present (returns False) or when
        the key doesn't exist (returns False). Doesn't bump
        updated_at — adding a label isn't a content change.
        """
        tag = tag.strip()
        if not tag:
            return False
        with self._txn() as con:
            row = con.execute(
                "SELECT tags FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return False
            try:
                tags = json.loads(row["tags"] or "[]")
            except (json.JSONDecodeError, KeyError):
                tags = []
            if tag in tags:
                return False
            tags.append(tag)
            con.execute(
                "UPDATE memories SET tags = ? WHERE key = ?",
                (json.dumps(tags), key),
            )
        return True

    def remove_tag(self, key: str, tag: str) -> bool:
        """Drop a tag from an existing memory. Returns True on a real change."""
        with self._txn() as con:
            row = con.execute(
                "SELECT tags FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return False
            try:
                tags = json.loads(row["tags"] or "[]")
            except (json.JSONDecodeError, KeyError):
                tags = []
            if tag not in tags:
                return False
            tags = [t for t in tags if t != tag]
            con.execute(
                "UPDATE memories SET tags = ? WHERE key = ?",
                (json.dumps(tags), key),
            )
        return True

    def all_tags(self) -> dict[str, int]:
        """Return {tag: usage_count} across the corpus. Useful for filters."""
        counts: dict[str, int] = {}
        for entry in self.recall_all():
            for t in entry.tags:
                counts[t] = counts.get(t, 0) + 1
        return counts

    def set_scope(self, key: str, scope: str) -> bool:
        """Move a single memory to a new scope. Returns True on a real change.

        Use case: an entry that landed under a project scope turns
        out to be universal ("user is on macOS") and should be
        promoted to global ("") — or vice versa, a global entry is
        really specific to one project. Doesn't touch any other
        column, so updated_at is preserved.
        """
        with self._txn() as con:
            row = con.execute(
                "SELECT scope FROM memories WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return False
            try:
                old = row["scope"] or ""
            except (IndexError, KeyError):
                old = ""
            if old == scope:
                return False
            con.execute(
                "UPDATE memories SET scope = ? WHERE key = ?",
                (scope, key),
            )
        return True

    def find_near_duplicates(
        self,
        threshold: float = 0.85,
        *,
        same_scope_only: bool = True,
    ) -> list[tuple[MemoryEntry, MemoryEntry, float]]:
        """Surface candidate-duplicate pairs.

        Uses vector cosine similarity when the embeddings extra is
        installed and both entries carry a vector, otherwise falls
        back to Jaccard over content tokens. Pairs are returned only
        once (key_a < key_b) and only when similarity ≥ threshold.

        ``same_scope_only`` skips cross-scope pairs by default —
        the same content under two scopes is usually intentional
        ("preferences I have everywhere" vs. "preferences I have
        on this project"), not a duplicate.
        """
        entries = self.recall_all()
        if len(entries) < 2:
            return []
        # Group by scope when requested so the O(n^2) loop only
        # considers entries that could realistically be merged.
        if same_scope_only:
            buckets: dict[str, list[MemoryEntry]] = {}
            for e in entries:
                buckets.setdefault(e.scope or "", []).append(e)
            groups = list(buckets.values())
        else:
            groups = [entries]

        from towel.memory import embeddings as _emb

        use_vec = _emb.is_available()
        pairs: list[tuple[MemoryEntry, MemoryEntry, float]] = []
        for group in groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    score = _similarity(a, b, use_vec)
                    if score >= threshold:
                        pairs.append((a, b, score))
        # Highest similarity first so the most confident dupes are
        # surfaced before borderline cases.
        pairs.sort(key=lambda t: -t[2])
        return pairs

    def consolidate(
        self,
        pair: tuple[MemoryEntry, MemoryEntry],
    ) -> MemoryEntry:
        """Merge two memories into a single survivor.

        Survivor selection: the entry with more recalls wins; ties
        break to the older created_at (the longer-established record).
        Tags are unioned; source prefers non-empty (operator-set
        beats heuristic). Scope must match — caller responsibility,
        enforced here so a careless merge can't silently broaden
        visibility.

        Returns the surviving entry as it sits in the store after
        the merge. The losing entry is forgotten (which cascades
        any graph edges).
        """
        a, b = pair
        if (a.scope or "") != (b.scope or ""):
            raise ValueError(
                "consolidate() refuses cross-scope merges — "
                "scopes differ for these entries"
            )
        # Survivor: higher recall_count, then older created_at.
        if a.recall_count != b.recall_count:
            survivor, loser = (a, b) if a.recall_count > b.recall_count else (b, a)
        else:
            survivor, loser = (a, b) if a.created_at <= b.created_at else (b, a)
        # Union tags, preserving survivor's order then loser's
        # unique adds at the end.
        seen: set[str] = set()
        merged_tags: list[str] = []
        for t in (*survivor.tags, *loser.tags):
            if t not in seen:
                seen.add(t)
                merged_tags.append(t)
        # Source: prefer non-empty (operator-set or named source over "").
        merged_source = survivor.source or loser.source
        # Drop the loser first so a primary-key rewrite is unnecessary.
        self.forget(loser.key)
        return self.remember(
            survivor.key,
            survivor.content,
            memory_type=survivor.memory_type,
            source=merged_source,
            tags=merged_tags,
            scope=survivor.scope,
        )

    def activity(
        self,
        hours: float = 24.0,
        *,
        column: str = "created_at",
        bucket_hours: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Histogram of memory writes over a recent window.

        ``column`` is either ``"created_at"`` (new captures only) or
        ``"updated_at"`` (every touch, including re-tagging). Returns
        a list of ``{"bucket": <iso-utc-start>, "count": <int>,
        "by_source": {...}}`` from oldest to newest bucket. Buckets
        with zero writes are included so the result is dense enough
        to render a sparkline without client-side gap-filling.
        """
        if column not in ("created_at", "updated_at"):
            raise ValueError(f"column must be created_at or updated_at, got {column!r}")
        if hours <= 0 or bucket_hours <= 0:
            return []
        now = datetime.now(UTC)
        start = now.timestamp() - hours * 3600
        start_iso = datetime.fromtimestamp(start, UTC).isoformat()
        # Bucketize in Python — sqlite's strftime is awkward for
        # arbitrary fractional-hour buckets and the corpus is small.
        con = self._connect()
        try:
            rows = con.execute(
                f"SELECT {column} AS ts, source FROM memories "
                f"WHERE {column} >= ? "
                f"ORDER BY {column} ASC",
                (start_iso,),
            ).fetchall()
        finally:
            con.close()
        bucket_secs = bucket_hours * 3600
        # Build empty buckets so a quiet period doesn't disappear.
        # max(1, ...) ensures hours == bucket_hours still gives one
        # bucket. Ceiling so a fractional remainder gets a trailing
        # bucket that ends right at "now".
        num_buckets = max(1, int((hours + 1e-9) // bucket_hours))
        buckets: list[dict[str, Any]] = []
        for i in range(num_buckets):
            b_start = start + i * bucket_secs
            buckets.append({
                "bucket": datetime.fromtimestamp(b_start, UTC).isoformat(),
                "count": 0,
                "by_source": {},
            })
        # The last bucket covers any rows beyond its nominal end so a
        # write at exactly "now" still lands in the most recent slot.
        for row in rows:
            ts = datetime.fromisoformat(row["ts"]).timestamp()
            raw_idx = int((ts - start) / bucket_secs)
            idx = max(0, min(num_buckets - 1, raw_idx))
            buckets[idx]["count"] += 1
            src = (row["source"] or "") or "operator"
            buckets[idx]["by_source"][src] = buckets[idx]["by_source"].get(src, 0) + 1
        return buckets

    def embedding_dims(self) -> dict[int, int]:
        """Histogram of embedding dimensions in the corpus.

        Returns {dim: count}. Used by the doctor check to flag
        mixed-dimension corpora — usually a sign that
        $TOWEL_EMBED_MODEL changed without a `memory reembed --all`
        and the old vectors are now invisible to cosine_topk.
        """
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
        hist: dict[int, int] = {}
        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            # float32 = 4 bytes per dimension.
            dim = len(blob) // 4
            hist[dim] = hist.get(dim, 0) + 1
        return hist

    def reembed_all(self, *, only_missing: bool = True) -> int:
        """Recompute embeddings across the corpus. Returns rows touched.

        Two use cases:

        * Installing the ``[embeddings]`` extra after the corpus is
          already populated — without a backfill, none of the existing
          entries have vectors and vector_search returns empty.
        * Switching the embedding model (via $TOWEL_EMBED_MODEL) —
          old vectors have the wrong dimensionality and are silently
          skipped by ``cosine_topk``. Pass ``only_missing=False`` to
          rewrite them.
        """
        from towel.memory import embeddings as _emb

        if not _emb.is_available():
            return 0
        con = self._connect()
        try:
            if only_missing:
                rows = con.execute(
                    "SELECT key, content FROM memories WHERE embedding IS NULL"
                ).fetchall()
            else:
                rows = con.execute("SELECT key, content FROM memories").fetchall()
        finally:
            con.close()
        if not rows:
            return 0
        updates: list[tuple[bytes | None, str]] = []
        for r in rows:
            blob = _emb.encode(r["content"])
            if blob is not None:
                updates.append((blob, r["key"]))
        if not updates:
            return 0
        with self._txn() as con:
            con.executemany(
                "UPDATE memories SET embedding = ? WHERE key = ?",
                updates,
            )
        return len(updates)

    # Floor for cosine similarity in vector_search. Below this the
    # match is essentially noise — for sentence-transformer style
    # models (all-MiniLM-L6-v2 and similar 384-d encoders), unrelated
    # text typically scores 0.0–0.3 and genuine paraphrases score 0.4+.
    # Without this floor, a tiny corpus (e.g. 7 unrelated test memories)
    # made every query "match" every memory via the top-k path —
    # fused_search then RRF-blended noise into real results, every
    # /api/ask call recall-bumped all memories, and the recall log
    # showed identical 7-key results for "hi", "List three colors",
    # and "Respond with just: OK". Threshold rejects that noise so
    # genuinely-unrelated queries fall through to the prompt-block
    # fallback path (which doesn't bump recall stats).
    VECTOR_SEARCH_MIN_SCORE = 0.30

    def vector_search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Cosine-rank memories against ``query`` via stored embeddings.

        Returns ``[]`` whenever the embeddings extra isn't installed,
        no entry in the corpus has an embedding yet, or no candidate
        clears :attr:`VECTOR_SEARCH_MIN_SCORE`. Callers fall back to
        BM25 / graph in those cases (``fused_search`` does this for
        you).
        """
        from towel.memory import embeddings as _emb

        query = (query or "").strip()
        if not query or not _emb.is_available():
            return []
        query_blob = _emb.encode(query)
        if not query_blob:
            return []
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT key, embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
        finally:
            con.close()
        if not rows:
            return []
        scored = _emb.cosine_topk(
            query_blob,
            [(r["key"], r["embedding"]) for r in rows],
            k=limit,
            min_score=self.VECTOR_SEARCH_MIN_SCORE,
        )
        if not scored:
            return []
        # Materialize the entries in score order.
        out: list[MemoryEntry] = []
        for key, _score in scored:
            entry = self.recall(key)
            if entry is not None:
                out.append(entry)
        return out

    def fused_search(
        self,
        query: str,
        limit: int = 5,
        tag: str | None = None,
        scope: str | None = None,
    ) -> list[MemoryEntry]:
        """3-way Reciprocal Rank Fusion: BM25 + vector + graph.

        Standard RRF: each ranker contributes ``1 / (k + rank_i)`` to
        the score of each candidate it sees; we sum across rankers
        and pick the top-K. The constant ``k=60`` is the value from
        the original RRF paper — empirically robust without needing
        per-query tuning.

        Three rankers are blended:

        * BM25 lexical match over content (always present).
        * Vector cosine similarity (only when the embeddings extra
          is installed and the corpus has vectors).
        * Graph co-retrieval: the neighbors of whichever entry tops
          the BM25 ranking become a third rank list, sorted by edge
          weight. This is how a query that only matches one seed
          lexically still pulls in semantically-adjacent memories
          the graph has learned across previous turns.

        When any ranker is empty (e.g. no embeddings, or no edges
        yet), it simply contributes nothing — RRF degrades gracefully
        rather than over-weighting the surviving signals.
        """
        rrf_k = 60

        # Local closure: would this entry pass the scope filter?
        def _scope_ok(e: MemoryEntry) -> bool:
            if scope is None:
                ds = self.default_scope
                return (not ds) or e.scope in (ds, "")
            if scope == "":
                return e.scope == ""
            return e.scope == scope

        # Tag filter is applied to each ranker so neither BM25 nor the
        # vector path can dominate by surfacing untagged matches.
        # Scope is passed into search() so it short-circuits there;
        # vector and graph paths apply it via _scope_ok below.
        bm25 = self.search(query, limit=limit * 2, tag=tag, scope=scope)
        vec_all = self.vector_search(query, limit=limit * 2)
        vec = [
            e for e in vec_all
            if (not tag or tag in e.tags) and _scope_ok(e)
        ]
        # Graph ranker: seed from the top BM25 hit (most-confident
        # lexical anchor), pull weighted neighbors. If BM25 missed
        # too, try the top vector hit so paraphrase queries still
        # exercise the graph.
        graph_entries: list[MemoryEntry] = []
        anchor = (bm25 or vec or [None])[0]
        if anchor is not None:
            for rel, _w in self.recall_related(anchor.key, limit=limit * 2):
                if (not tag or tag in rel.tags) and _scope_ok(rel):
                    graph_entries.append(rel)

        scores: dict[str, float] = {}
        entries: dict[str, MemoryEntry] = {}
        for ranker in (bm25, vec, graph_entries):
            for rank, e in enumerate(ranker):
                scores[e.key] = scores.get(e.key, 0.0) + 1.0 / (rrf_k + rank + 1)
                entries[e.key] = e
        if not scores:
            return []
        ordered = sorted(scores, key=lambda k: -scores[k])[:limit]
        return [entries[k] for k in ordered]

    def _bump_recall(self, keys: list[str]) -> None:
        """Mark a set of memories as recently retrieved + record co-occurrence.

        Each retrieved memory gets its recall_count bumped. When more
        than one was retrieved in the same call, every ordered pair
        gets its memory_links.weight incremented — this is how the
        graph learns "these memories are relevant together" without
        any LLM or embedding step.
        """
        if not keys:
            return
        now_iso = datetime.now(UTC).isoformat()
        # Symmetric pairs (we record both directions) so recall_related
        # works the same regardless of which side the operator asked
        # about.
        pairs: list[tuple[str, str]] = []
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if i != j:
                    pairs.append((a, b))
        with self._txn() as con:
            con.executemany(
                "UPDATE memories SET recall_count = recall_count + 1, "
                "last_recalled_at = ? WHERE key = ?",
                [(now_iso, k) for k in keys],
            )
            if pairs:
                # UPSERT: insert with weight 1, or bump if the edge
                # already exists. ON CONFLICT requires SQLite 3.24+
                # (released 2018), which is comfortably below the
                # 3.53.0 the doctor check verifies.
                con.executemany(
                    "INSERT INTO memory_links (source_key, target_key, weight) "
                    "VALUES (?, ?, 1) "
                    "ON CONFLICT(source_key, target_key) "
                    "DO UPDATE SET weight = weight + 1",
                    pairs,
                )

    def recalls_returning(
        self, key: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Recent recalls whose returned keys list contains ``key``.

        Pairs with the inspect view: when an operator opens an entry,
        we can show "this was returned in response to these queries"
        so the auditing question "why does the agent know X?" turns
        from grep-the-logs into one click. JSON-LIKE prefilter +
        Python re-check for the same substring-safety reason
        recent_recalls uses.
        """
        if not key:
            return []
        con = self._connect()
        try:
            rows = con.execute(
                'SELECT ts, query, keys, scope FROM recall_log '
                'WHERE keys LIKE ? ORDER BY id DESC LIMIT ?',
                (f'%"{key}"%', limit * 4),
            ).fetchall()
        finally:
            con.close()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                keys = json.loads(r["keys"])
            except json.JSONDecodeError:
                continue
            if key not in keys:
                continue
            out.append({
                "ts": r["ts"],
                "query": r["query"],
                "rank": keys.index(key),
                "result_size": len(keys),
                "scope": r["scope"] or "",
            })
            if len(out) >= limit:
                break
        return out

    # Tunable cap on the recall_log table. The default matches the
    # original cap=5000 default for record_recall — kept as a class
    # attribute so operators can rebind it (e.g. for very long-running
    # daemons that want a fatter audit window) without forking the
    # signature on every call site.
    RECALL_LOG_CAP: int = 5000

    def recall_log_size(self) -> int:
        """How many rows the recall_log currently holds. Used by doctor."""
        con = self._connect()
        try:
            row = con.execute("SELECT COUNT(*) AS n FROM recall_log").fetchone()
        finally:
            con.close()
        return int(row["n"]) if row else 0

    def record_recall(
        self,
        query: str,
        keys: list[str],
        *,
        scope: str | None = None,
        cap: int | None = None,
    ) -> None:
        """Append a row to recall_log and trim to ``cap`` newest entries.

        Called from to_prompt_block whenever a query-relevant retrieval
        produces a non-empty result. We trim instead of TTL because
        the operator's actual concern is "the last few hours of
        activity" — a fixed-row cap keeps the table tiny without
        wall-clock logic. cap=5000 at ~150 bytes/row is well under
        a megabyte.
        """
        if not query or not keys:
            return
        # Cap the stored query so a 1MB user message (a paste of logs,
        # a code dump, etc.) doesn't bloat the recall_log table — the
        # operator's actual question text is captured in the first
        # couple hundred chars, and /memory/recalls renders this
        # column in a list view that's unreadable past one line
        # anyway. Same conceptual cap the search path uses (eb86631).
        max_logged_query = 500
        logged_query = query if len(query) <= max_logged_query else (
            query[:max_logged_query] + "…"
        )
        effective_cap = cap if cap is not None else self.RECALL_LOG_CAP
        now_iso = datetime.now(UTC).isoformat()
        scope_str = scope if scope is not None else self.default_scope
        try:
            with self._txn() as con:
                con.execute(
                    "INSERT INTO recall_log (ts, query, keys, scope) VALUES (?, ?, ?, ?)",
                    (now_iso, logged_query, json.dumps(keys), scope_str or ""),
                )
                # Trim: keep the most recent `cap` rows. Cheaper than
                # COUNT-then-DELETE; sqlite optimizes this pattern.
                con.execute(
                    "DELETE FROM recall_log WHERE id NOT IN ("
                    "  SELECT id FROM recall_log ORDER BY id DESC LIMIT ?"
                    ")",
                    (effective_cap,),
                )
        except sqlite3.Error as exc:
            # Recall logging is best-effort — must never break retrieval.
            log.debug("record_recall failed: %s", exc)

    def recent_recalls(
        self,
        limit: int = 50,
        since_hours: float | None = None,
        key_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read recall events back out. Newest first.

        ``key_filter`` is a substring match against the JSON-encoded
        keys list and the query, re-checked in Python after a LIKE
        prefilter — gives the operator "show me every recall that
        mentioned <X>" in one call.
        """
        con = self._connect()
        try:
            clauses: list[str] = []
            params: list[Any] = []
            if since_hours is not None:
                cutoff = (datetime.now(UTC).timestamp() - since_hours * 3600)
                clauses.append("ts >= ?")
                params.append(datetime.fromtimestamp(cutoff, UTC).isoformat())
            if key_filter:
                clauses.append("(keys LIKE ? OR query LIKE ?)")
                params.append(f'%"{key_filter}"%')
                params.append(f"%{key_filter}%")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = con.execute(
                f"SELECT ts, query, keys, scope FROM recall_log "
                f"{where} ORDER BY id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        finally:
            con.close()
        out: list[dict[str, Any]] = []
        for r in rows:
            try:
                keys = json.loads(r["keys"])
            except json.JSONDecodeError:
                keys = []
            if key_filter:
                # Re-check after LIKE pre-filter so "vim" doesn't
                # match a query containing "vimal".
                if key_filter not in keys and key_filter not in (r["query"] or ""):
                    continue
            out.append({
                "ts": r["ts"],
                "query": r["query"],
                "keys": keys,
                "scope": r["scope"] or "",
            })
        return out

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
            # fused_search runs 3-way RRF over BM25 + vector + graph
            # in one pass — semantically-adjacent neighbors land in
            # the result directly, no second-pass round-robin needed.
            entries = self.fused_search(query, limit=limit)
            used_fallback = False
            if not entries:
                # Empty fused-search means the query had no lexical /
                # vector / graph signal. Fall back to a short slice
                # ranked by usefulness: user-type and preference-type
                # entries are surfaced before fact-type, because a
                # bare "hi" should remind the agent who it's talking
                # to — not dump the operator's longest scratch note.
                # Within each type bucket, newer entries win.
                priority_rank = {"user": 0, "preference": 1, "project": 2, "fact": 3}
                con = self._connect()
                try:
                    rows = con.execute(
                        "SELECT * FROM memories ORDER BY updated_at DESC"
                    ).fetchall()
                finally:
                    con.close()
                ranked = sorted(
                    rows,
                    key=lambda r: priority_rank.get(r["memory_type"], 4),
                )
                entries = [_row_to_entry(r) for r in ranked[:limit]]
                used_fallback = True
            # Only the genuine fused-search hits are real recalls.
            # The fallback grabs whatever was newest by type to keep
            # the agent oriented; counting those as recalls would
            # inflate recall_count uniformly on every chat turn that
            # didn't lexically match a memory (e.g. "hi"), and would
            # pollute /memory/recalls with bogus entries that don't
            # actually answer the question. The earlier code claimed
            # in a comment to skip the log on fallback — but the code
            # didn't. Now it does.
            if not used_fallback:
                keys_in_order = [e.key for e in entries]
                self._bump_recall(keys_in_order)
                self.record_recall(query, keys_in_order)

        if not entries:
            return ""

        lines = [
            "\n\n## Persistent Memory\n"
            "These are facts remembered from past sessions. Entries under "
            "`User` describe the human user you are talking to, not the "
            "assistant. If the user asks for their name or says \"my name\", "
            "answer from `user_name` when present.\n"
        ]
        by_type: dict[str, list[MemoryEntry]] = {}
        for e in entries:
            by_type.setdefault(e.memory_type, []).append(e)
        # Cap per-entry content in the prompt block. Operators can store
        # arbitrarily long memories (a TODO list, a code snippet) but
        # dumping a 100KB entry into the system prompt blows past the
        # worker's context window — and `to_prompt_block` is called on
        # EVERY turn, so the bloat compounds. 2KB per entry preserves
        # plenty of signal for the model; CLI / /memory readers still
        # see the full body.
        prompt_content_cap = 2000
        for mtype in ["user", "preference", "project", "fact"]:
            group = by_type.get(mtype, [])
            if not group:
                continue
            lines.append(f"\n**{mtype.title()}:**")
            for e in group:
                content = e.content
                if len(content) > prompt_content_cap:
                    content = content[:prompt_content_cap] + "… [truncated]"
                lines.append(f"- {e.key}: {content}")
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
        source_prefix: str | None = None,
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
        params: list[Any] = [cutoff_iso, *protected]
        if source_prefix is not None:
            sql_where += " AND source LIKE ?"
            params.append(f"{source_prefix}%")
        con = self._connect()
        try:
            rows = con.execute(
                f"SELECT * FROM memories WHERE {sql_where}",
                params,
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

    def recall_related(
        self, key: str, limit: int = 5
    ) -> list[tuple[MemoryEntry, int]]:
        """Return entries linked to ``key`` by co-retrieval, by weight.

        The graph is populated by _bump_recall whenever multiple
        memories appear in the same prompt-block; popular pairs
        accumulate weight, occasional ones don't. ``limit`` caps the
        result so callers (CLI inspect, UI sidebar) can show a small
        meaningful subset.
        """
        if not key:
            return []
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT m.*, l.weight FROM memory_links l "
                "JOIN memories m ON m.key = l.target_key "
                "WHERE l.source_key = ? "
                "ORDER BY l.weight DESC LIMIT ?",
                (key, limit),
            ).fetchall()
        finally:
            con.close()
        return [(_row_to_entry(r), int(r["weight"])) for r in rows]

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


# ── duplicate-detection helpers ───────────────────────────────────────


import re  # noqa: E402  used by the duplicate-detection helpers below


def _content_tokens(text: str) -> set[str]:
    """Lowercase word-token set, used for the Jaccard fallback path."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3}


def _similarity(
    a: MemoryEntry, b: MemoryEntry, use_vec: bool
) -> float:
    """Score 0..1 of how likely two entries are duplicates.

    Vector path: cosine on stored embeddings when both have them.
    Falls back to Jaccard on content tokens — surprisingly robust
    for the "near-duplicate detection" use case where exact paraphrase
    quality doesn't matter, only "are these basically the same fact?"
    """
    # Trivial: identical content is always a candidate.
    if a.content == b.content:
        return 1.0
    if use_vec:
        from towel.memory import embeddings as _emb
        a_blob = _emb.encode(a.content)
        b_blob = _emb.encode(b.content)
        if a_blob and b_blob:
            try:
                import numpy as np

                va = np.frombuffer(a_blob, dtype=np.float32)
                vb = np.frombuffer(b_blob, dtype=np.float32)
                if va.shape == vb.shape:
                    # Vectors come back unit-normalized → dot = cosine.
                    return float(va @ vb)
            except Exception:
                pass
    # Jaccard fallback.
    ta, tb = _content_tokens(a.content), _content_tokens(b.content)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


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

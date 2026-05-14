"""Tests for the persistent memory system."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from towel.memory.store import MemoryEntry, MemoryStore, salience


@pytest.fixture
def store(tmp_path):
    return MemoryStore(store_dir=tmp_path)


class TestMemoryStore:
    def test_remember_and_recall(self, store):
        store.remember("user_name", "Kelsi", memory_type="user")
        entry = store.recall("user_name")
        assert entry is not None
        assert entry.content == "Kelsi"
        assert entry.memory_type == "user"

    def test_remember_updates_existing(self, store):
        store.remember("lang", "Python")
        store.remember("lang", "Rust")
        entry = store.recall("lang")
        assert entry is not None
        assert entry.content == "Rust"
        assert store.count == 1

    def test_forget(self, store):
        store.remember("temp", "delete me")
        assert store.forget("temp")
        assert store.recall("temp") is None
        assert not store.forget("temp")  # already gone

    def test_recall_nonexistent(self, store):
        assert store.recall("nope") is None

    def test_recall_all(self, store):
        store.remember("a", "1", memory_type="fact")
        store.remember("b", "2", memory_type="user")
        store.remember("c", "3", memory_type="fact")

        all_entries = store.recall_all()
        assert len(all_entries) == 3

        facts = store.recall_all(memory_type="fact")
        assert len(facts) == 2

        users = store.recall_all(memory_type="user")
        assert len(users) == 1

    def test_search_keyword_hit(self, store):
        store.remember("favorite_language", "Python is great")
        store.remember("favorite_food", "Pizza")
        store.remember("project_deadline", "March 2026")

        results = store.search("favorite")
        # Substring fallback kicks in for "favorite" (FTS5 token boundary
        # match) so both favorite_* keys come back.
        assert {e.key for e in results} == {"favorite_language", "favorite_food"}

        results = store.search("python")
        assert len(results) == 1
        assert results[0].key == "favorite_language"

    def test_count(self, store):
        assert store.count == 0
        store.remember("a", "1")
        store.remember("b", "2")
        assert store.count == 2

    def test_persistence_across_instances(self, tmp_path):
        store1 = MemoryStore(store_dir=tmp_path)
        store1.remember("persistent", "I survive restarts")

        store2 = MemoryStore(store_dir=tmp_path)
        entry = store2.recall("persistent")
        assert entry is not None
        assert entry.content == "I survive restarts"

    def test_to_prompt_block_empty(self, store):
        assert store.to_prompt_block() == ""

    def test_to_prompt_block_with_entries(self, store):
        store.remember("name", "Kelsi", memory_type="user")
        store.remember("style", "concise", memory_type="preference")
        store.remember("project", "Towel v0.2", memory_type="project")

        block = store.to_prompt_block()
        assert "Your Memory" in block
        assert "Kelsi" in block
        assert "concise" in block
        assert "Towel v0.2" in block
        assert "remember" in block.lower()

    def test_to_prompt_block_grouped_by_type(self, store):
        store.remember("a", "1", memory_type="user")
        store.remember("b", "2", memory_type="fact")
        block = store.to_prompt_block()
        assert "**User:**" in block
        assert "**Fact:**" in block


class TestBM25Ranking:
    """FTS5 BM25 ranking is the headline upgrade — search() must rank
    relevant content above lexical near-misses."""

    def test_bm25_ranks_content_match(self, store):
        store.remember("jwt", "fixed JWT auth bug in login endpoint")
        store.remember("rate", "added rate limiting to API gateway")
        store.remember("query", "optimized N+1 queries in orders pipeline")
        store.remember("role", "user is a data scientist")

        # Single-token search returns only the matching row.
        results = store.search("queries", limit=5)
        assert [e.key for e in results] == ["query"]

        # Multi-token AND-ish ranking: "rate limiting" should pick the
        # rate-limiter memory above unrelated entries.
        results = store.search("rate limiting", limit=5)
        assert results[0].key == "rate"

    def test_search_falls_back_to_substring(self, store):
        # FTS5 tokenizes on word boundaries — apostrophes and short
        # punctuation can cause MATCH to miss. The substring fallback
        # exists for that case.
        store.remember("k_apos", "user's preference is dark mode")
        results = store.search("preference")
        assert any(e.key == "k_apos" for e in results)

    def test_search_empty_query_returns_empty(self, store):
        store.remember("k", "v")
        assert store.search("") == []
        assert store.search("   ") == []

    def test_search_respects_limit(self, store):
        for i in range(10):
            store.remember(f"k{i}", f"sample text number {i}")
        results = store.search("sample", limit=3)
        assert len(results) == 3


class TestQueryRelevantPromptBlock:
    """to_prompt_block(query=…) must surface the right memories AND
    bump recall stats for downstream decay/forget passes."""

    def test_query_filters_by_relevance(self, store):
        store.remember("role", "user is a data scientist", "user")
        store.remember("project_jwt", "fixed JWT auth bug", "project")
        store.remember("project_db", "optimized N+1 database queries", "project")

        block = store.to_prompt_block(query="data scientist", limit=2)
        # "role" wins; the unrelated project entries should not all
        # appear (limit=2 caps the dump).
        assert "data scientist" in block
        assert block.count("project_") <= 1

    def test_query_bumps_recall_stats(self, store):
        store.remember("hit", "this content has the magic token")
        store.remember("miss", "unrelated text")
        assert store.recall("hit").recall_count == 0

        store.to_prompt_block(query="magic token", limit=5)
        after = store.recall("hit")
        assert after.recall_count == 1
        assert after.last_recalled_at is not None
        # Untouched entries stay at zero so we don't poison the decay
        # signal with mass-bumps.
        assert store.recall("miss").recall_count == 0

    def test_no_query_dumps_everything(self, store):
        # Legacy callers (TUI, `towel memory list`) get the full corpus.
        for i in range(15):
            store.remember(f"k{i}", f"text {i}")
        block = store.to_prompt_block()
        for i in range(15):
            assert f"k{i}" in block

    def test_empty_query_returns_recent_slice(self, store):
        # When FTS5 + substring both miss, the prompt block falls back
        # to the most recent N memories so the agent still has SOMETHING
        # personal. Better than an empty block.
        store.remember("a", "alpha", "user")
        store.remember("b", "beta", "user")
        block = store.to_prompt_block(query="completely unrelated xyzzy", limit=5)
        assert "alpha" in block or "beta" in block


class TestJsonMigration:
    """The first time the new store opens against an existing
    ~/.towel/memory/memories.json, we import it once into SQLite and
    rename the old file so we never re-import."""

    def _seed_json(self, dirpath, payload):
        (dirpath / "memories.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_migrates_entries_then_renames_marker(self, tmp_path):
        self._seed_json(tmp_path, {
            "role": {
                "key": "role", "content": "data scientist", "type": "user",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            "goal": {
                "key": "goal", "content": "fix N+1 queries", "type": "project",
                "created_at": "2026-01-02T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            },
        })
        store = MemoryStore(store_dir=tmp_path)
        assert store.count == 2
        assert store.recall("role").content == "data scientist"

        # Marker rename — the original file is gone; an archived copy
        # remains so the operator can recover if needed.
        assert not (tmp_path / "memories.json").exists()
        archives = list(tmp_path.glob("memories.json.migrated-*"))
        assert len(archives) == 1

    def test_migration_is_idempotent(self, tmp_path):
        # First pass imports.
        self._seed_json(tmp_path, {
            "k": {
                "key": "k", "content": "v", "type": "fact",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
        })
        MemoryStore(store_dir=tmp_path)
        # Subsequent opens find no JSON file and don't double-import.
        store2 = MemoryStore(store_dir=tmp_path)
        assert store2.count == 1

    def test_migration_skips_malformed_entries(self, tmp_path):
        # Single bad row shouldn't poison the whole import.
        self._seed_json(tmp_path, {
            "ok": {
                "key": "ok", "content": "good", "type": "fact",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            },
            "bad": {"this": "is missing required fields"},
        })
        store = MemoryStore(store_dir=tmp_path)
        assert store.recall("ok") is not None
        assert store.recall("bad") is None

    def test_corrupt_json_does_not_crash(self, tmp_path):
        (tmp_path / "memories.json").write_text("{not valid json", encoding="utf-8")
        # Store opens fine and is empty; corrupted file is left as-is
        # for the operator to inspect.
        store = MemoryStore(store_dir=tmp_path)
        assert store.count == 0


class TestSqliteBacking:
    def test_uses_sqlite_not_json(self, store):
        store.remember("k", "v")
        # The new on-disk format is memory.db, not memories.json. This
        # is what callers like `towel doctor` and ops scripts will key
        # off when checking for migration.
        assert (store.store_dir / "memory.db").exists()
        assert not (store.store_dir / "memories.json").exists()

    def test_fts_index_kept_in_sync_by_triggers(self, store):
        store.remember("alpha", "the quick brown fox")
        # Update content — the FTS index should reflect the new text,
        # not the old, after the trigger fires.
        store.remember("alpha", "lazy dogs sleep all afternoon")
        # FTS query for original content must miss; new content must hit.
        assert store.search("quick brown") == []
        assert any(e.key == "alpha" for e in store.search("lazy dogs"))

        # Deletes also propagate.
        store.forget("alpha")
        assert store.search("lazy dogs") == []


class TestSalience:
    def test_recent_recalled_outranks_old_untouched(self):
        now = datetime.now(UTC)
        recent = MemoryEntry(
            key="a", content="x", memory_type="fact",
            created_at=now, updated_at=now,
            last_recalled_at=now, recall_count=3,
        )
        stale = MemoryEntry(
            key="b", content="y", memory_type="fact",
            created_at=now - timedelta(days=180),
            updated_at=now - timedelta(days=180),
            last_recalled_at=None, recall_count=0,
        )
        assert salience(recent, now) > salience(stale, now)

    def test_high_recall_beats_age(self):
        # A heavily-recalled memory (even one older than the half-life)
        # should outrank a fresh one that's never been used.
        now = datetime.now(UTC)
        veteran = MemoryEntry(
            key="vet", content="x", memory_type="fact",
            created_at=now - timedelta(days=120),
            updated_at=now - timedelta(days=120),
            last_recalled_at=now - timedelta(days=1),
            recall_count=50,
        )
        rookie = MemoryEntry(
            key="rook", content="y", memory_type="fact",
            created_at=now, updated_at=now,
            recall_count=0,
        )
        assert salience(veteran, now) > salience(rookie, now)


class TestAutoForget:
    def test_prunes_old_unused_facts(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("old_fact", "stale", "fact")
        store.remember("fresh_fact", "active", "fact")

        # Manually backdate the old entry directly in SQLite — quicker
        # than freezing time, and exercises the same code path.
        import sqlite3
        old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        con = sqlite3.connect(str(tmp_path / "memory.db"))
        con.execute(
            "UPDATE memories SET updated_at = ? WHERE key = 'old_fact'",
            (old,),
        )
        con.commit()
        con.close()

        pruned = store.auto_forget(max_age_days=90)
        assert [p.key for p in pruned] == ["old_fact"]
        assert store.recall("old_fact") is None
        assert store.recall("fresh_fact") is not None

    def test_protected_types_never_pruned(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("role", "engineer", "user")
        store.remember("style", "concise", "preference")
        store.remember("project", "towel", "project")

        # Even with extreme age, protected types survive.
        old = (datetime.now(UTC) - timedelta(days=1000)).isoformat()
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "memory.db"))
        con.execute("UPDATE memories SET updated_at = ?", (old,))
        con.commit()
        con.close()

        pruned = store.auto_forget(max_age_days=30)
        assert pruned == []
        assert store.count == 3

    def test_recalled_fact_survives_even_when_old(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("important", "this gets used", "fact")
        # Touch it once via the recall-bump path so recall_count = 1.
        store._bump_recall(["important"])

        old = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "memory.db"))
        con.execute(
            "UPDATE memories SET updated_at = ? WHERE key = 'important'",
            (old,),
        )
        con.commit()
        con.close()

        pruned = store.auto_forget(max_age_days=30)
        assert pruned == []
        assert store.recall("important") is not None

    def test_dry_run_does_not_delete(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("doomed", "x", "fact")
        old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        import sqlite3
        con = sqlite3.connect(str(tmp_path / "memory.db"))
        con.execute("UPDATE memories SET updated_at = ?", (old,))
        con.commit()
        con.close()

        pruned = store.auto_forget(max_age_days=90, dry_run=True)
        # Returns what WOULD be deleted...
        assert [p.key for p in pruned] == ["doomed"]
        # ...but the row is still there.
        assert store.recall("doomed") is not None

    def test_rank_by_salience_orders_lowest_first(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("never_used", "x", "fact")
        store.remember("popular", "y", "fact")
        store._bump_recall(["popular"])
        store._bump_recall(["popular"])
        store._bump_recall(["popular"])

        ranked = store.rank_by_salience()
        keys_by_rank = [e.key for e, _ in ranked]
        # The unrecalled one should be at the bottom.
        assert keys_by_rank[0] == "never_used"
        assert keys_by_rank[-1] == "popular"


class TestSourceTracking:
    def test_remember_records_source(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("k", "v", source="auto_capture:role")
        e = store.recall("k")
        assert e.source == "auto_capture:role"

    def test_default_source_is_empty(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path)
        store.remember("k", "v")
        assert store.recall("k").source == ""

    def test_update_preserves_original_source(self, tmp_path):
        # Operator-set memories keep their provenance even when content
        # is updated — important so heuristic re-firing on the same key
        # doesn't relabel a deliberate entry as auto-captured.
        store = MemoryStore(store_dir=tmp_path)
        store.remember("role", "engineer", source="")
        store.remember("role", "senior engineer", source="auto_capture:role")
        assert store.recall("role").source == ""

    def test_auto_forget_source_prefix_filter(self, tmp_path):
        from datetime import UTC, datetime, timedelta
        import sqlite3
        store = MemoryStore(store_dir=tmp_path)
        store.remember("op_fact", "x", "fact", source="")
        store.remember("auto_fact", "y", "fact", source="auto_capture:role")
        # Age both so they're prune candidates.
        old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
        con = sqlite3.connect(str(tmp_path / "memory.db"))
        con.execute("UPDATE memories SET updated_at = ?", (old,))
        con.commit()
        con.close()

        pruned = store.auto_forget(
            max_age_days=90, source_prefix="auto_capture:"
        )
        assert [p.key for p in pruned] == ["auto_fact"]
        assert store.recall("op_fact") is not None


class TestTags:
    def test_remember_accepts_tags(self, store):
        store.remember("k", "v", tags=["work", "urgent"])
        e = store.recall("k")
        assert e.tags == ["work", "urgent"]

    def test_remember_normalizes_tags(self, store):
        store.remember("k", "v", tags=[" work ", "work", "", "  "])
        e = store.recall("k")
        assert e.tags == ["work"]

    def test_remember_merges_tags_on_update(self, store):
        store.remember("k", "v", tags=["a", "b"])
        store.remember("k", "v2", tags=["b", "c"])
        e = store.recall("k")
        assert e.tags == ["a", "b", "c"]

    def test_remember_tags_none_leaves_existing(self, store):
        store.remember("k", "v", tags=["a"])
        store.remember("k", "v2")
        assert store.recall("k").tags == ["a"]

    def test_add_tag_returns_true_on_change(self, store):
        store.remember("k", "v")
        assert store.add_tag("k", "new") is True
        assert "new" in store.recall("k").tags
        assert store.add_tag("k", "new") is False  # already present

    def test_remove_tag(self, store):
        store.remember("k", "v", tags=["a", "b"])
        assert store.remove_tag("k", "a") is True
        assert store.recall("k").tags == ["b"]
        assert store.remove_tag("k", "a") is False  # not present

    def test_recall_all_filters_by_tag(self, store):
        store.remember("a", "x", tags=["work"])
        store.remember("b", "y", tags=["home"])
        store.remember("c", "z", tags=["work", "urgent"])
        keys = {e.key for e in store.recall_all(tag="work")}
        assert keys == {"a", "c"}

    def test_recall_all_tag_substring_safety(self, store):
        # "work" shouldn't match "homework" — substring on LIKE could
        # false-positive, so Python re-check is what guards it.
        store.remember("a", "x", tags=["homework"])
        assert store.recall_all(tag="work") == []

    def test_all_tags_counts_usage(self, store):
        store.remember("a", "x", tags=["work", "urgent"])
        store.remember("b", "y", tags=["work"])
        counts = store.all_tags()
        assert counts == {"work": 2, "urgent": 1}


class TestMemoryGraph:
    def test_co_retrieval_creates_links(self, store):
        store.remember("a", "alpha", "fact")
        store.remember("b", "beta", "fact")
        store.remember("c", "gamma", "fact")
        # Pull a + b together in a prompt block — both should now have
        # links to each other but not to c.
        store.to_prompt_block(query="alpha beta", limit=2)
        related_a = store.recall_related("a")
        related_b = store.recall_related("b")
        assert {key for entry, _ in related_a for key in [entry.key]} <= {"b", "c"}
        # The fallback-to-recent path might pull c too; what we
        # really want to assert is the bidirectional link a↔b exists.
        assert any(entry.key == "b" for entry, _ in related_a)
        assert any(entry.key == "a" for entry, _ in related_b)

    def test_repeat_co_retrieval_bumps_weight(self, store):
        store.remember("a", "x", "fact")
        store.remember("b", "y", "fact")
        store._bump_recall(["a", "b"])
        store._bump_recall(["a", "b"])
        store._bump_recall(["a", "b"])
        related = store.recall_related("a")
        b_weight = next(w for entry, w in related if entry.key == "b")
        assert b_weight == 3

    def test_forget_cascades_link_cleanup(self, store):
        store.remember("a", "x", "fact")
        store.remember("b", "y", "fact")
        store._bump_recall(["a", "b"])
        store.forget("b")
        # The link from a→b should be gone since b's row is gone.
        assert store.recall_related("a") == []

    def test_self_links_excluded(self, store):
        store.remember("solo", "only one", "fact")
        store._bump_recall(["solo"])
        # Single-key bump shouldn't create any links at all.
        assert store.recall_related("solo") == []

    def test_recall_related_ordered_by_weight(self, store):
        store.remember("a", "x", "fact")
        store.remember("b", "weak", "fact")
        store.remember("c", "strong", "fact")
        store._bump_recall(["a", "b"])             # a-b weight 1
        for _ in range(5):
            store._bump_recall(["a", "c"])          # a-c weight 5
        related = store.recall_related("a")
        keys_by_rank = [e.key for e, _ in related]
        assert keys_by_rank == ["c", "b"]


class TestGraphAugmentedRetrieval:
    """to_prompt_block(query=...) should pull in linked neighbors of
    BM25 hits so the agent gets semantically-adjacent memories even
    when the user's wording doesn't share lexical tokens with the
    target entry."""

    def test_neighbors_join_the_prompt_block(self, store):
        # Seed a graph: 'role' is heavily linked to 'editor'. A query
        # that hits 'role' lexically should pull 'editor' along.
        store.remember("role", "data scientist", "user")
        store.remember("editor", "neovim", "preference")
        store.remember("unrelated", "cloud setup notes", "fact")
        for _ in range(5):
            store._bump_recall(["role", "editor"])

        block = store.to_prompt_block(query="scientist", limit=8)
        # The seed BM25 hit ('role') brings 'editor' via graph
        # augmentation; the unrelated entry should not appear
        # (no lexical match and no graph link).
        assert "data scientist" in block
        assert "neovim" in block
        assert "cloud setup" not in block

    def test_limit_respected_with_graph_augmentation(self, store):
        # Make sure neighbor expansion doesn't blow past the limit.
        for i in range(10):
            store.remember(f"k{i}", f"shared word context-{i}", "fact")
        # Link all entries to k0 so it has many neighbors.
        for i in range(1, 10):
            store._bump_recall(["k0", f"k{i}"])

        block = store.to_prompt_block(query="shared word", limit=3)
        # The block has at most `limit` entries — count the bullets.
        bullets = block.count("\n- ")
        assert bullets <= 3


class TestMemoryEntry:
    def test_serialization_roundtrip(self):
        entry = MemoryEntry(key="test", content="value", memory_type="fact")
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.key == entry.key
        assert restored.content == entry.content
        assert restored.memory_type == entry.memory_type

    def test_str(self):
        entry = MemoryEntry(key="name", content="Kelsi", memory_type="user")
        assert "[user] name: Kelsi" in str(entry)

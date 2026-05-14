"""Tests for the optional embeddings layer.

The actual sentence-transformers model is not loaded in CI — these
tests cover:

* graceful degradation when the extra isn't installed
* cosine math correctness with hand-crafted vectors
* RRF fusion ordering vs BM25 alone
* embedding column persistence + None handling

Anything that needs a real model load belongs in a separate
slow-test target so the default CI run stays fast.
"""

from __future__ import annotations

import struct

import pytest

from towel.memory import embeddings as emb
from towel.memory.store import MemoryStore


# ── module-state isolation ────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_embedding_state(monkeypatch):
    """Reset the lazy-loaded model cache between tests."""
    monkeypatch.setattr(emb, "_model", None)
    monkeypatch.setattr(emb, "_model_load_failed", False)
    yield


# ── graceful degradation ──────────────────────────────────────────────


class TestUnavailable:
    def test_encode_returns_none_without_extra(self, monkeypatch):
        monkeypatch.setattr(emb, "is_available", lambda: False)
        assert emb.encode("anything") is None

    def test_encode_empty_text_returns_none(self):
        assert emb.encode("") is None

    def test_cosine_topk_handles_empty(self):
        assert emb.cosine_topk(b"", [("k", b"\x00")]) == []
        assert emb.cosine_topk(b"\x00", []) == []


# ── cosine math ───────────────────────────────────────────────────────


def _vec(values: list[float]) -> bytes:
    """Pack a list of floats as little-endian float32 bytes."""
    return struct.pack(f"<{len(values)}f", *values)


class TestCosineTopK:
    def test_orders_by_cosine_descending(self):
        np = pytest.importorskip("numpy")
        # 3-dim unit vectors; query is [1,0,0]. Closest should be
        # the candidate that aligns best with the x-axis.
        q = _vec([1.0, 0.0, 0.0])
        candidates = [
            ("perfect", _vec([1.0, 0.0, 0.0])),
            ("orthogonal", _vec([0.0, 1.0, 0.0])),
            ("opposite", _vec([-1.0, 0.0, 0.0])),
            ("close", _vec([0.9, 0.1, 0.0])),
        ]
        top = emb.cosine_topk(q, candidates, k=4)
        keys_by_rank = [k for k, _ in top]
        assert keys_by_rank[0] == "perfect"
        # "close" should beat "orthogonal" should beat "opposite".
        assert keys_by_rank.index("close") < keys_by_rank.index("orthogonal")
        assert keys_by_rank.index("orthogonal") < keys_by_rank.index("opposite")

    def test_skips_dimension_mismatch(self):
        pytest.importorskip("numpy")
        q = _vec([1.0, 0.0, 0.0])
        # One candidate has a different dimensionality — must be
        # skipped, not crash.
        ok = ("ok", _vec([1.0, 0.0, 0.0]))
        bad = ("bad", _vec([1.0, 0.0]))
        top = emb.cosine_topk(q, [ok, bad], k=2)
        assert [k for k, _ in top] == ["ok"]


# ── store integration ────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    return MemoryStore(store_dir=tmp_path)


class TestEmbeddingPersistence:
    def test_remember_stores_embedding_when_available(self, store, monkeypatch):
        # Stub encode so we don't load a real model.
        calls = []
        def fake_encode(text):
            calls.append(text)
            return _vec([1.0, 0.0, 0.0])
        monkeypatch.setattr(emb, "encode", fake_encode)
        store.remember("k", "some content")
        # Verify the row carries an embedding blob.
        import sqlite3
        con = sqlite3.connect(str(store._db_path))
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT embedding FROM memories WHERE key='k'").fetchone()
        con.close()
        assert row["embedding"] == _vec([1.0, 0.0, 0.0])
        # The hook was called with the content, not the key.
        assert calls == ["some content"]

    def test_remember_handles_no_embedding(self, store, monkeypatch):
        monkeypatch.setattr(emb, "encode", lambda text: None)
        store.remember("k", "v")  # must not raise
        # Round-trip still works.
        assert store.recall("k").content == "v"


class TestVectorSearch:
    def test_returns_empty_when_unavailable(self, store, monkeypatch):
        monkeypatch.setattr(emb, "is_available", lambda: False)
        store.remember("k", "v")
        assert store.vector_search("anything") == []

    def test_ranks_by_cosine(self, store, monkeypatch):
        # Hand-pick vectors per content so the test is deterministic
        # without loading a real model.
        canned = {
            "Python is great":  _vec([1.0, 0.0, 0.0]),
            "snake handling":   _vec([0.95, 0.31, 0.0]),  # close to python
            "stamp collecting": _vec([0.0, 0.0, 1.0]),
            "scientist":        _vec([0.1, 0.99, 0.0]),
        }
        def fake_encode(text):
            return canned.get(text)
        monkeypatch.setattr(emb, "encode", fake_encode)
        monkeypatch.setattr(emb, "is_available", lambda: True)

        for content in canned:
            store.remember(content[:10], content, "fact")

        # Query that resolves to ~[1,0,0] — should match python first.
        results = store.vector_search("Python is great", limit=2)
        assert results[0].content == "Python is great"
        # "snake handling" should beat "stamp collecting" / "scientist".
        assert results[1].content in {"snake handling"}


class TestReembedAll:
    def test_noop_without_extra(self, store, monkeypatch):
        monkeypatch.setattr(emb, "is_available", lambda: False)
        store.remember("k", "v")
        assert store.reembed_all() == 0

    def test_only_missing_skips_rows_with_existing_embedding(self, store, monkeypatch):
        canned = {"alpha": _vec([1.0, 0.0]), "beta": _vec([0.0, 1.0])}
        monkeypatch.setattr(emb, "encode", lambda t: canned.get(t))
        monkeypatch.setattr(emb, "is_available", lambda: True)
        # First write — embedding lands.
        store.remember("a", "alpha")
        # Drop embeddings on disk so we have a clear missing-vector
        # row to backfill.
        import sqlite3
        con = sqlite3.connect(str(store._db_path))
        con.execute("UPDATE memories SET embedding = NULL WHERE key = 'a'")
        con.commit()
        con.close()
        store.remember("b", "beta")  # embedding present
        # Backfill the missing one.
        assert store.reembed_all(only_missing=True) == 1

    def test_full_reembed_touches_every_row(self, store, monkeypatch):
        monkeypatch.setattr(emb, "encode", lambda t: _vec([1.0, 0.0]))
        monkeypatch.setattr(emb, "is_available", lambda: True)
        store.remember("a", "x")
        store.remember("b", "y")
        # Both rows already have embeddings. only_missing=False should
        # still rewrite both.
        assert store.reembed_all(only_missing=False) == 2


class TestFusedSearch:
    def test_falls_back_to_bm25_without_vectors(self, store, monkeypatch):
        # Without embeddings, fused_search returns BM25 ordering.
        monkeypatch.setattr(emb, "is_available", lambda: False)
        store.remember("a", "alpha beta gamma", "fact")
        store.remember("b", "delta epsilon", "fact")
        fused = store.fused_search("alpha")
        assert fused and fused[0].key == "a"

    def test_three_way_rrf_uses_graph_signal(self, store, monkeypatch):
        # No embeddings tier, no lexical overlap between query and
        # neighbor — but the graph link from the BM25 hit should be
        # enough to pull the neighbor into the fused result.
        monkeypatch.setattr(emb, "is_available", lambda: False)
        store.remember("anchor", "scientist data analysis", "fact")
        store.remember("buddy", "completely orthogonal content", "fact")
        # Strong graph link from anchor → buddy.
        for _ in range(5):
            store._bump_recall(["anchor", "buddy"])
        fused = store.fused_search("scientist", limit=5)
        keys = {e.key for e in fused}
        assert "anchor" in keys
        assert "buddy" in keys

    def test_combines_bm25_and_vector_via_rrf(self, store, monkeypatch):
        # Set up a corpus where BM25 ranks A first and vector ranks
        # B first; RRF should land both near the top with A slightly
        # ahead (it was rank 1 on BM25 vs rank 2 on vector for itself).
        canned = {
            "alpha keyword match": _vec([0.0, 1.0, 0.0]),
            "paraphrase target":   _vec([1.0, 0.0, 0.0]),
        }
        def fake_encode(text):
            # The query "alpha" resolves to vector that points at
            # paraphrase target ([1,0,0]).
            if text == "alpha":
                return _vec([1.0, 0.0, 0.0])
            return canned.get(text)
        monkeypatch.setattr(emb, "encode", fake_encode)
        monkeypatch.setattr(emb, "is_available", lambda: True)

        store.remember("kw", "alpha keyword match", "fact")
        store.remember("para", "paraphrase target", "fact")
        fused = store.fused_search("alpha", limit=2)
        keys = [e.key for e in fused]
        # Both should appear in the fused top-2 (BM25 picks "kw",
        # vector picks "para") — order may go either way depending
        # on RRF tie-break since both got rank 1 from one ranker.
        assert set(keys) == {"kw", "para"}

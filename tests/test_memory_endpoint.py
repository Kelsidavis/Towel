"""Tests for the /memory introspection endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.memory.store import MemoryStore
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore


class _FakeAgent:
    def __init__(self, memory: Any) -> None:
        self.memory = memory


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


@pytest.fixture
def memory(tmp_path):
    mem = MemoryStore(store_dir=tmp_path / "memory")
    mem.remember("favourite_color", "green", memory_type="preference")
    mem.remember("user_role", "data scientist", memory_type="user")
    mem.remember("project_status", "shipping next week", memory_type="project")
    return mem


def _gateway(store, agent: Any) -> GatewayServer:
    sessions = SessionManager(store=store)
    pin_store = SessionPinStore(path=store.store_dir / "session_pins.json")
    worker_state_store = WorkerStateStore(path=store.store_dir / "worker_state.json")
    return GatewayServer(
        config=TowelConfig(),
        agent=agent,
        sessions=sessions,
        pin_store=pin_store,
        worker_state_store=worker_state_store,
    )


class TestMemoryEndpoint:
    def test_lists_all_memories(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        keys = {m["key"] for m in data["memories"]}
        assert keys == {"favourite_color", "user_role", "project_status"}

    def test_filter_by_type(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory?type=user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["memories"][0]["key"] == "user_role"
        assert data["memories"][0]["type"] == "user"

    def test_substring_search(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory?q=shipping")
        data = resp.json()
        assert data["count"] == 1
        assert data["memories"][0]["key"] == "project_status"

    def test_search_and_type_compose(self, store, memory):
        # Search-then-type-filter should narrow further, not widen.
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory?q=green&type=preference")
        data = resp.json()
        assert data["count"] == 1
        assert data["memories"][0]["key"] == "favourite_color"

        # Mismatched filter should return nothing.
        resp = client.get("/memory?q=green&type=user")
        assert resp.json()["count"] == 0

    def test_limit_caps_response_and_flags_truncation(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory?limit=1")
        data = resp.json()
        assert data["count"] == 3        # full pre-limit total
        assert len(data["memories"]) == 1
        assert data["truncated"] is True

    def test_invalid_limit_rejected(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory?limit=not-a-number")
        assert resp.status_code == 400
        assert "limit" in resp.json()["error"]

    def test_rejects_overlong_q(self, store, memory):
        """A 1000-char q would walk the entire FTS index / substring
        fallback path with no upside — match the 256-char rule used
        on /search and other operator-facing query strings."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory?q=" + "a" * 1000)
        assert resp.status_code == 400
        assert "256" in resp.json()["error"]

    def test_rejects_control_chars_in_q(self, store, memory):
        """Null byte / embedded newline in `q` break log readability
        and the substring fallback would otherwise scan with the
        control char embedded — reject at the boundary."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        for encoded in ("%00", "hello%0aworld", "tab%09here"):
            resp = client.get(f"/memory?q={encoded}")
            assert resp.status_code == 400, f"accepted q={encoded!r}"
            assert "control" in resp.json()["error"].lower()

    def test_recalls_rejects_overlong_key(self, store, memory):
        """key_filter on /memory/recalls had no length / control-char
        guard — a 1000-char value or null byte slipped into the
        substring scan."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory/recalls?key=" + "a" * 1000)
        assert resp.status_code == 400
        assert "256" in resp.json()["error"]

    def test_recalls_rejects_control_chars_in_key(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        for encoded in ("%00", "with%0anewline"):
            resp = client.get(f"/memory/recalls?key={encoded}")
            assert resp.status_code == 400, f"accepted key={encoded!r}"
            assert "control" in resp.json()["error"].lower()

    def test_recalls_rejects_non_finite_hours(self, store, memory):
        """float('inf') / float('nan') / float('-inf') silently passed
        through to the SQL bound comparison and produced empty
        results indistinguishable from "no recent activity." Surface
        the bad input as 400."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        for bad in ("inf", "-inf", "nan", "Infinity"):
            resp = client.get(f"/memory/recalls?hours={bad}")
            assert resp.status_code == 400, f"accepted hours={bad!r}"
            assert "finite" in resp.json()["error"].lower()

    def test_recalls_rejects_negative_hours(self, store, memory):
        """hours=-1 would scan a future window (empty) — empty result
        looks identical to "no activity," misleading the operator."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory/recalls?hours=-1")
        assert resp.status_code == 400
        assert (
            "0" in resp.json()["error"]
            or "≥" in resp.json()["error"]
            or "non-negative" in resp.json()["error"].lower()
        )

    def test_rejects_overlong_tag(self, store, memory):
        """Same SQLITE_MAX_LIKE_PATTERN_LENGTH guard the /memory `q`
        rule applies: tag flows into a `tags LIKE %"{tag}"%` pattern
        in MemoryStore.recall_all that would trip the 50000-char
        SQLite limit on a megabyte tag and crash with 500."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory?tag=" + "a" * 1000)
        assert resp.status_code == 400
        assert "256" in resp.json()["error"]

    def test_rejects_control_chars_in_tag(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        for encoded in ("%00", "with%0anewline"):
            resp = client.get(f"/memory?tag={encoded}")
            assert resp.status_code == 400, f"accepted tag={encoded!r}"
            assert "control" in resp.json()["error"].lower()

    def test_returns_empty_when_agent_has_no_memory(self, store):
        class _BareAgent:
            pass

        gw = _gateway(store, _BareAgent())
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory")
        assert resp.status_code == 200
        assert resp.json() == {"memories": [], "count": 0}

    def test_delete_removes_entry(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())

        resp = client.delete("/memory/favourite_color")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "key": "favourite_color"}

        # The entry is gone from subsequent lists.
        remaining = client.get("/memory").json()
        assert "favourite_color" not in {m["key"] for m in remaining["memories"]}
        assert remaining["count"] == 2

    def test_delete_unknown_key_returns_404(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.delete("/memory/never-existed")
        assert resp.status_code == 404
        assert "never-existed" in resp.json()["error"]

    def test_patch_updates_content(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.patch(
            "/memory/user_role",
            json={"content": "senior data scientist"},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "senior data scientist"
        # And the change is durable.
        assert memory.recall("user_role").content == "senior data scientist"

    def test_patch_replaces_tags_wholesale(self, store, memory):
        memory.remember(
            "tagged", "x", memory_type="fact", tags=["old1", "old2"],
        )
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.patch("/memory/tagged", json={"tags": ["new"]})
        assert resp.status_code == 200
        # PATCH semantics REPLACE the tag list (not merge — that's
        # what add_tag is for).
        assert memory.recall("tagged").tags == ["new"]

    def test_patch_changes_type(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.patch(
            "/memory/favourite_color", json={"type": "user"}
        )
        assert resp.status_code == 200
        assert memory.recall("favourite_color").memory_type == "user"

    def test_patch_rejects_unknown_type(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.patch(
            "/memory/favourite_color", json={"type": "bogus"}
        )
        assert resp.status_code == 400

    def test_patch_unknown_key_returns_404(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.patch("/memory/never-existed", json={"content": "x"})
        assert resp.status_code == 404

    def test_patch_rejects_empty_content(self, store, memory):
        """POST requires non-empty content; PATCH must too. Without
        this, an accidental `{"content": ""}` silently destroys the
        memory's content with no way to recover."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        # Confirm there's a real entry to attempt destroying.
        assert memory.recall("favourite_color") is not None
        resp = client.patch("/memory/favourite_color", json={"content": ""})
        assert resp.status_code == 400
        assert "empty" in resp.json()["error"].lower()
        # And content must still be intact.
        assert memory.recall("favourite_color").content != ""

    def test_patch_rejects_whitespace_content(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.patch("/memory/favourite_color", json={"content": "   \t\n  "})
        assert resp.status_code == 400

    def test_post_creates_new_entry(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.post(
            "/memory",
            json={"key": "fresh", "content": "hello", "type": "fact"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["key"] == "fresh"
        assert body["content"] == "hello"
        assert memory.recall("fresh") is not None

    def test_post_rejects_duplicate_key(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.post(
            "/memory",
            json={"key": "user_role", "content": "x"},
        )
        assert resp.status_code == 409

    def test_post_requires_key_and_content(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        r1 = client.post("/memory", json={"content": "x"})
        r2 = client.post("/memory", json={"key": "k"})
        assert r1.status_code == 400
        assert r2.status_code == 400

    def test_post_rejects_overlong_key(self, store, memory):
        """Memory keys appear in URL paths and logs; a 1000-char key
        produces absurd URLs and breaks log readability. Cap at 256."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.post(
            "/memory",
            json={"key": "x" * 1000, "content": "y"},
        )
        assert resp.status_code == 400
        assert "256" in resp.json()["error"]

    def test_post_rejects_control_chars_in_key(self, store, memory):
        """Newlines, NULs etc. break URL routing and log parsing."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        for bad_key in ("a\nb", "a\tb", "a\x00b"):
            resp = client.post(
                "/memory",
                json={"key": bad_key, "content": "y"},
            )
            assert resp.status_code == 400, f"accepted bad key {bad_key!r}"
            assert "control" in resp.json()["error"].lower()

    def test_post_strips_leading_trailing_whitespace_from_key(self, store, memory):
        """A key like "trailing  " was stored as-is, so recall via the
        natural-looking "trailing" returned 404 — invisible whitespace
        made the entry effectively unrecallable without exact replay."""
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.post(
            "/memory",
            json={"key": "  spaced_key  ", "content": "x"},
        )
        assert resp.status_code == 201
        # The persisted key is the stripped version.
        assert memory.recall("spaced_key") is not None
        # And the as-stored key has no leading/trailing space.
        assert resp.json()["key"] == "spaced_key"

    def test_post_tags_and_scope_persist(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.post(
            "/memory",
            json={
                "key": "tagged_proj",
                "content": "x",
                "tags": ["a", "b"],
                "scope": "proj:foo",
            },
        )
        assert resp.status_code == 201
        e = memory.recall("tagged_proj")
        assert e.tags == ["a", "b"]
        assert e.scope == "proj:foo"

    def test_nudge_bumps_recall_count(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        before = memory.recall("user_role").recall_count
        resp = client.post("/memory/user_role/nudge")
        assert resp.status_code == 200
        after = memory.recall("user_role").recall_count
        assert after == before + 1

    def test_activity_returns_buckets(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.get("/memory/activity?hours=2&bucket_hours=1")
        assert resp.status_code == 200
        body = resp.json()
        assert "buckets" in body
        assert len(body["buckets"]) >= 2
        # The fixture writes via remember(...) which stamps created_at
        # at fixture-time, so at least one bucket should have count > 0.
        assert sum(b["count"] for b in body["buckets"]) >= 3

    def test_activity_rejects_bad_column(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.get("/memory/activity?column=evil")
        assert resp.status_code == 400

    def test_nudge_unknown_key_404(self, store, memory):
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.post("/memory/never-existed/nudge")
        assert resp.status_code == 404

    def test_patch_changes_scope(self, store, memory):
        memory.remember("rove", "x", scope="proj:a")
        gw = _gateway(store, _FakeAgent(memory))
        client = TestClient(gw._build_http_app())
        resp = client.patch("/memory/rove", json={"scope": ""})
        assert resp.status_code == 200
        assert memory.recall("rove").scope == ""

    def test_delete_returns_503_when_no_memory_backend(self, store):
        class _BareAgent:
            pass

        gw = _gateway(store, _BareAgent())
        client = TestClient(gw._build_http_app())
        resp = client.delete("/memory/anything")
        assert resp.status_code == 503

    def test_orders_newest_first(self, store, tmp_path):
        # recall_all returns dict-iteration order; the endpoint must sort
        # explicitly so the most-recently-updated entry leads.
        mem = MemoryStore(store_dir=tmp_path / "memory_ordered")
        mem.remember("oldest", "a")
        mem.remember("middle", "b")
        mem.remember("newest", "c")
        # Touch "oldest" again so it has the latest updated_at.
        mem.remember("oldest", "a — refreshed")

        gw = _gateway(store, _FakeAgent(mem))
        client = TestClient(gw._build_http_app())
        data = client.get("/memory").json()
        assert [m["key"] for m in data["memories"]] == ["oldest", "newest", "middle"]

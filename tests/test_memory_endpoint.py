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

    def test_returns_empty_when_agent_has_no_memory(self, store):
        class _BareAgent:
            pass

        gw = _gateway(store, _BareAgent())
        client = TestClient(gw._build_http_app())

        resp = client.get("/memory")
        assert resp.status_code == 200
        assert resp.json() == {"memories": [], "count": 0}

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

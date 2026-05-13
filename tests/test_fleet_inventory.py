"""Tests for the /fleet/inventory aggregate endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore


class _FakeAgent:
    pass


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


@pytest.fixture
def gateway(store):
    sessions = SessionManager(store=store)
    pin_store = SessionPinStore(path=store.store_dir / "session_pins.json")
    worker_state_store = WorkerStateStore(path=store.store_dir / "worker_state.json")
    return GatewayServer(
        config=TowelConfig(),
        agent=_FakeAgent(),
        sessions=sessions,
        pin_store=pin_store,
        worker_state_store=worker_state_store,
    )


def _register(gateway: GatewayServer, worker_id: str, **caps) -> None:
    gateway._workers.register(worker_id, ws=MagicMock(), capabilities=caps)


class TestFleetInventory:
    def test_aggregates_models_across_workers(self, gateway):
        _register(
            gateway, "mac-studio",
            available_models=["qwen3.6:27b", "haiku"],
            max_param_b_est=53.0,
        )
        _register(
            gateway, "rtx5090",
            available_models=["qwen3.6:27b", "Llama-3.3-70B"],
            max_param_b_est=70.0,
        )
        _register(
            gateway, "pi",
            available_models=["tinyllama"],
            max_param_b_est=3.0,
        )

        client = TestClient(gateway._build_http_app())
        resp = client.get("/fleet/inventory")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_workers"] == 3
        assert data["total_unique"] == 4
        assert data["fleet_max_param_b"] == 70.0

        by_name = {m["name"]: m for m in data["models"]}
        # qwen3.6:27b is on two workers; should sort first.
        assert data["models"][0]["name"] == "qwen3.6:27b"
        assert data["models"][0]["cached_count"] == 2
        assert by_name["qwen3.6:27b"]["workers"] == ["mac-studio", "rtx5090"]
        assert by_name["tinyllama"]["workers"] == ["pi"]

    def test_sorts_by_count_desc_then_name(self, gateway):
        _register(gateway, "a", available_models=["beta", "alpha"])
        _register(gateway, "b", available_models=["alpha"])
        _register(gateway, "c", available_models=["gamma"])
        client = TestClient(gateway._build_http_app())
        data = client.get("/fleet/inventory").json()
        # alpha (2 copies) first, then beta and gamma (1 each, alphabetical).
        names = [m["name"] for m in data["models"]]
        assert names == ["alpha", "beta", "gamma"]

    def test_empty_fleet_returns_empty_inventory(self, gateway):
        client = TestClient(gateway._build_http_app())
        data = client.get("/fleet/inventory").json()
        assert data["models"] == []
        assert data["total_unique"] == 0
        assert data["total_workers"] == 0
        assert data["fleet_max_param_b"] == 0

    def test_workers_with_no_inventory_dont_break_aggregation(self, gateway):
        _register(gateway, "old", available_models=None)  # legacy field-less worker
        _register(gateway, "new", available_models=["sonnet"])
        client = TestClient(gateway._build_http_app())
        data = client.get("/fleet/inventory").json()
        assert data["total_workers"] == 2
        assert data["total_unique"] == 1
        assert data["models"][0]["workers"] == ["new"]

    def test_garbage_entries_in_inventory_are_dropped(self, gateway):
        _register(
            gateway, "w",
            available_models=["", None, 42, "valid-model"],
        )
        client = TestClient(gateway._build_http_app())
        data = client.get("/fleet/inventory").json()
        names = [m["name"] for m in data["models"]]
        assert names == ["valid-model"]

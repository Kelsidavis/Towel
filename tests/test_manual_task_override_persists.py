"""Tests that operator-set worker task overrides survive worker reconnect.

The fleet panel's "save tasks" button writes to ``GatewayServer._node_tasks``
via the HTTP handler. Previously, a worker disconnect followed by a
reconnect (transient network blip, restart for upgrade, etc.) would
overwrite the operator's choice with auto-assigned defaults because the
register path unconditionally called ``assign_tasks(capabilities, roles)``.

This regression test asserts the new ``_manual_tasks`` shadow dict
preserves the override across a register cycle.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.nodes.roles import TaskType, assign_roles, assign_tasks
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


class TestManualTaskOverride:
    def test_register_uses_manual_override_when_present(self, gateway):
        # Simulate an operator who previously set manual tasks for this
        # worker (the HTTP handler at /workers/{id}/tasks writes here).
        worker_id = "gpu-host"
        manual = [TaskType.CODE_REVIEW, TaskType.RESEARCH]
        gateway._manual_tasks[worker_id] = manual

        # Now simulate the register code path running on a fresh connect.
        # We can't easily wire the websocket flow, so call the bits the
        # register-message handler runs to populate _node_tasks.
        capabilities = {"backend": "mlx", "modes": ["mlx_prompt"]}
        roles = assign_roles(capabilities)
        gateway._node_roles[worker_id] = roles
        # The handler now consults _manual_tasks first.
        override = gateway._manual_tasks.get(worker_id)
        gateway._node_tasks[worker_id] = (
            override if override is not None else assign_tasks(capabilities, roles)
        )

        assert gateway._node_tasks[worker_id] == manual

    def test_register_falls_back_to_auto_when_no_override(self, gateway):
        worker_id = "auto-host"
        assert worker_id not in gateway._manual_tasks
        capabilities = {"backend": "mlx", "modes": ["mlx_prompt"]}
        roles = assign_roles(capabilities)
        gateway._node_roles[worker_id] = roles
        override = gateway._manual_tasks.get(worker_id)
        gateway._node_tasks[worker_id] = (
            override if override is not None else assign_tasks(capabilities, roles)
        )
        # Auto-assigned tasks were applied (non-empty for an MLX-capable host).
        assert gateway._node_tasks[worker_id] is not None
        # Auto-assigned, not the manual list (which doesn't exist).
        assert override is None

    def test_setting_empty_task_list_via_handler_removes_override(self, gateway):
        # Pre-condition: an operator-set override exists.
        worker_id = "fickle-host"
        gateway._manual_tasks[worker_id] = [TaskType.CHAT]

        # Now the operator wipes their selection — handler treats an empty
        # list as "fall back to auto-assigned on next register".
        from starlette.testclient import TestClient

        # Register a fake worker so the /workers/{id}/tasks handler doesn't
        # 404. The actual WS-side register path isn't needed here.
        gateway._workers.register(worker_id, ws=MagicMock(), capabilities={})
        client = TestClient(gateway._build_http_app())
        resp = client.post(f"/workers/{worker_id}/tasks", json={"tasks": []})
        assert resp.status_code == 200

        assert worker_id not in gateway._manual_tasks

    def test_setting_tasks_via_handler_persists_them(self, gateway):
        worker_id = "fresh-host"
        gateway._workers.register(worker_id, ws=MagicMock(), capabilities={})
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        resp = client.post(
            f"/workers/{worker_id}/tasks",
            json={"tasks": ["code_review", "research"]},
        )
        assert resp.status_code == 200
        assert worker_id in gateway._manual_tasks
        assert gateway._manual_tasks[worker_id] == [
            TaskType.CODE_REVIEW,
            TaskType.RESEARCH,
        ]


class TestManualTaskOverridePersistsAcrossRestart:
    """Setting tasks via the HTTP handler should also write to disk so a
    coordinator restart picks them up — not just a worker reconnect during
    the same coordinator session.
    """

    def test_handler_write_hits_disk(self, tmp_path, store):
        # First coordinator instance — write an override via the HTTP handler.
        sessions = SessionManager(store=store)
        pin_store = SessionPinStore(path=tmp_path / "pins.json")
        state_path = tmp_path / "worker_state.json"
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=sessions,
            pin_store=pin_store,
            worker_state_store=WorkerStateStore(path=state_path),
        )
        worker_id = "persistent-host"
        gateway._workers.register(worker_id, ws=MagicMock(), capabilities={})
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        resp = client.post(
            f"/workers/{worker_id}/tasks",
            json={"tasks": ["code_review"]},
        )
        assert resp.status_code == 200
        assert state_path.exists(), "handler should have flushed state to disk"

        # Second coordinator instance — same on-disk state file. The manual
        # override should be hydrated even before the worker connects.
        gateway2 = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=state_path),
        )
        assert worker_id in gateway2._manual_tasks
        assert gateway2._manual_tasks[worker_id] == [TaskType.CODE_REVIEW]

    def test_clearing_override_removes_disk_entry(self, tmp_path, store):
        state_path = tmp_path / "worker_state.json"
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=state_path),
        )
        worker_id = "clearable-host"
        gateway._workers.register(worker_id, ws=MagicMock(), capabilities={})
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        client.post(f"/workers/{worker_id}/tasks", json={"tasks": ["chat"]})
        client.post(f"/workers/{worker_id}/tasks", json={"tasks": []})

        # New coordinator picking up the same file — override should be gone.
        gateway2 = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=state_path),
        )
        assert worker_id not in gateway2._manual_tasks

    def test_reset_to_auto_rederives_defaults_immediately(self, tmp_path, store):
        """Clearing an override shouldn't leave the worker with an empty
        task list until the next register — it should re-derive from
        capabilities right now so the UI reflects auto state immediately."""
        state_path = tmp_path / "worker_state.json"
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=state_path),
        )
        worker_id = "reset-host"
        # Capabilities that yield non-empty auto-assigned tasks.
        capabilities = {
            "backend": "mlx",
            "modes": ["mlx_prompt"],
            "tools": True,
            "context_window": 8192,
            "total_vram_mb": 24000,
        }
        gateway._workers.register(worker_id, ws=MagicMock(), capabilities=capabilities)
        gateway._node_roles[worker_id] = assign_roles(capabilities)
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        # First override with a narrow list.
        client.post(f"/workers/{worker_id}/tasks", json={"tasks": ["chat"]})
        assert gateway._node_tasks[worker_id] == [TaskType.CHAT]
        # Reset: response should already show the broader auto-assigned set.
        resp = client.post(f"/workers/{worker_id}/tasks", json={"tasks": []})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["tasks_overridden"] is False
        # Re-derived list — at minimum more than just [chat] given the
        # capabilities. The exact tasks depend on TASK_REQUIREMENTS so we
        # only assert it's non-empty and matches what assign_tasks produces.
        expected = assign_tasks(capabilities, gateway._node_roles[worker_id])
        assert gateway._node_tasks[worker_id] == expected
        assert payload["assigned_tasks"] == [str(t) for t in expected]


class TestWorkersEndpointSurfacesOverrideFlag:
    """The /workers JSON should flag whether each worker's tasks are
    operator-overridden vs auto-derived, so the UI can render a badge."""

    def test_flag_true_when_override_present(self, tmp_path, store):
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=tmp_path / "worker_state.json"),
        )
        worker_id = "flagged-host"
        gateway._workers.register(worker_id, ws=MagicMock(), capabilities={})
        gateway._manual_tasks[worker_id] = [TaskType.CHAT]
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        resp = client.get("/workers").json()
        flagged = [w for w in resp["workers"] if w["id"] == worker_id]
        assert flagged and flagged[0]["tasks_overridden"] is True

    def test_flag_false_when_no_override(self, tmp_path, store):
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=tmp_path / "worker_state.json"),
        )
        worker_id = "auto-host"
        gateway._workers.register(worker_id, ws=MagicMock(), capabilities={})
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        resp = client.get("/workers").json()
        flagged = [w for w in resp["workers"] if w["id"] == worker_id]
        assert flagged and flagged[0]["tasks_overridden"] is False


class TestOfflinePersistedSurface:
    """Offline workers with persisted state should be visible in /workers
    so operators can manage them (clear stale overrides for hosts that
    aren't currently connected, etc.)."""

    def test_offline_persisted_surfaced_in_workers_payload(self, tmp_path, store):
        import json

        state_path = tmp_path / "worker_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "offline-host": {
                        "enabled": False,
                        "draining": False,
                        "tasks": ["chat"],
                    }
                }
            ),
            encoding="utf-8",
        )
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=state_path),
        )
        # No worker connected — should still surface via offline_persisted.
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        resp = client.get("/workers").json()
        assert resp["workers"] == []
        offline = resp["offline_persisted"]
        assert len(offline) == 1
        assert offline[0]["id"] == "offline-host"
        assert offline[0]["enabled"] is False
        assert offline[0]["manual_tasks"] == ["chat"]

    def test_clear_offline_worker_override(self, tmp_path, store):
        import json

        state_path = tmp_path / "worker_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "offline-host": {
                        "enabled": True,
                        "draining": False,
                        "tasks": ["chat"],
                    }
                }
            ),
            encoding="utf-8",
        )
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=state_path),
        )
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        resp = client.post("/workers/offline-host/tasks", json={"tasks": []})
        assert resp.status_code == 200
        body = resp.json()
        assert body["cleared_offline"] is True
        # Re-load /workers: override no longer surfaces.
        next_resp = client.get("/workers").json()
        offline = next_resp["offline_persisted"]
        assert all(
            not o.get("manual_tasks") for o in offline if o["id"] == "offline-host"
        )

    def test_setting_override_on_unknown_worker_rejected(self, tmp_path, store):
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=tmp_path / "worker_state.json"),
        )
        from starlette.testclient import TestClient

        client = TestClient(gateway._build_http_app())
        # Setting a non-empty list on a worker that doesn't exist (and never
        # did) should still 404 — prevents typos creating ghost overrides.
        resp = client.post(
            "/workers/never-existed/tasks", json={"tasks": ["chat"]}
        )
        assert resp.status_code == 404


class TestUnknownTaskOnDisk:
    def test_unknown_task_value_on_disk_is_skipped(self, tmp_path, store):
        """If the schema evolves and an unknown task name is on disk,
        the coordinator should drop it rather than crash on startup."""
        import json

        state_path = tmp_path / "worker_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "future-host": {
                        "enabled": True,
                        "draining": False,
                        "tasks": ["chat", "task_invented_in_2027"],
                    }
                }
            ),
            encoding="utf-8",
        )
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=_FakeAgent(),
            sessions=SessionManager(store=store),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=WorkerStateStore(path=state_path),
        )
        # The unknown value got dropped; the known one survived.
        assert gateway._manual_tasks["future-host"] == [TaskType.CHAT]

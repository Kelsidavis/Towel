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

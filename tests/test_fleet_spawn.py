"""Tests for the coordinator-side POST /fleet/spawn endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
    config = TowelConfig()
    return GatewayServer(
        config=config,
        agent=_FakeAgent(),
        sessions=sessions,
        pin_store=pin_store,
        worker_state_store=worker_state_store,
    )


def _mock_post(status_code: int = 200, payload: Any = None) -> AsyncMock:
    """Create an AsyncMock that returns a fake httpx Response."""
    fake_resp = MagicMock()
    fake_resp.status_code = status_code
    fake_resp.is_success = 200 <= status_code < 300
    fake_resp.json.return_value = payload if payload is not None else {"ok": True}
    fake_resp.text = ""
    post = AsyncMock(return_value=fake_resp)
    return post


class TestFleetSpawn:
    def test_happy_path_forwards_and_returns_launcher_response(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post(
            payload={"ok": True, "pid": 4321, "argv": ["towel", "worker", "..."]}
        )
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/spawn",
                json={
                    "launcher_url": "http://gpu-box:18751",
                    "launcher_token": "secret",
                    "worker": {"backend": "ollama", "worker_id": "w-1"},
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["launcher_url"] == "http://gpu-box:18751"
        assert data["launcher_status"] == 200
        assert data["launcher_response"]["pid"] == 4321
        # Coordinator auto-fills the controller URL when omitted.
        assert data["controller_used"].startswith("ws://")

        # Confirm the launcher was called with the merged payload + bearer token.
        post.assert_awaited_once()
        call_args = post.call_args
        assert call_args.args[0] == "http://gpu-box:18751/launch"
        sent_body = call_args.kwargs["json"]
        assert sent_body["backend"] == "ollama"
        assert sent_body["worker_id"] == "w-1"
        assert "controller" in sent_body
        sent_headers = call_args.kwargs["headers"]
        assert sent_headers["Authorization"] == "Bearer secret"

    def test_caller_controller_is_preserved_when_explicitly_set(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post()
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/spawn",
                json={
                    "launcher_url": "http://other-host:18751",
                    "worker": {
                        "controller": "ws://load-balancer:18742",
                        "backend": "mlx",
                    },
                },
            )
        assert resp.status_code == 200
        assert resp.json()["controller_used"] == "ws://load-balancer:18742"
        sent_body = post.call_args.kwargs["json"]
        assert sent_body["controller"] == "ws://load-balancer:18742"

    def test_no_token_means_no_authorization_header(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post()
        with patch("httpx.AsyncClient.post", post):
            client.post(
                "/fleet/spawn",
                json={"launcher_url": "http://x:18751", "worker": {"backend": "mlx"}},
            )
        sent_headers = post.call_args.kwargs["headers"]
        assert "Authorization" not in sent_headers

    def test_missing_launcher_url_returns_400(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post("/fleet/spawn", json={"worker": {"backend": "mlx"}})
        assert resp.status_code == 400
        assert "launcher_url" in resp.json()["error"]

    def test_non_object_worker_returns_400(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/spawn",
            json={"launcher_url": "http://x", "worker": ["not", "an", "object"]},
        )
        assert resp.status_code == 400
        assert "worker" in resp.json()["error"]

    def test_invalid_json_returns_400(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/spawn",
            content="not-json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_unreachable_launcher_returns_502(self, gateway):
        import httpx

        client = TestClient(gateway._build_http_app())
        post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/spawn",
                json={"launcher_url": "http://dead:18751", "worker": {}},
            )
        assert resp.status_code == 502
        assert "unreachable" in resp.json()["error"]

    def test_launcher_4xx_returns_502_with_launcher_status_preserved(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post(
            status_code=401, payload={"error": "invalid token"}
        )
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/spawn",
                json={
                    "launcher_url": "http://x:18751",
                    "launcher_token": "wrong",
                    "worker": {"backend": "mlx"},
                },
            )
        # Coordinator marks the overall call as 502 (its upstream failed) but
        # preserves the launcher's status code and body so callers can see why.
        assert resp.status_code == 502
        data = resp.json()
        assert data["launcher_status"] == 401
        assert data["launcher_response"]["error"] == "invalid token"


class TestFleetUpgrade:
    """Coordinator-side /fleet/upgrade proxies an upgrade command to a launcher."""

    def test_default_strategy_is_pip_when_none_given(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post(
            payload={"ok": True, "strategy": "pip", "returncode": 0,
                     "stdout": "Successfully installed towel-0.42.0", "stderr": ""}
        )
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/upgrade",
                json={"launcher_url": "http://gpu-box:18751", "launcher_token": "x"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["launcher_response"]["strategy"] == "pip"
        sent_body = post.call_args.kwargs["json"]
        assert sent_body == {"strategy": "pip"}
        assert post.call_args.args[0] == "http://gpu-box:18751/upgrade"

    def test_strategy_forwarded(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post(payload={"ok": True, "strategy": "git-pull"})
        with patch("httpx.AsyncClient.post", post):
            client.post(
                "/fleet/upgrade",
                json={"launcher_url": "http://x:18751", "strategy": "git-pull"},
            )
        assert post.call_args.kwargs["json"] == {"strategy": "git-pull"}

    def test_custom_command_forwarded(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post(payload={"ok": True, "strategy": "custom"})
        with patch("httpx.AsyncClient.post", post):
            client.post(
                "/fleet/upgrade",
                json={
                    "launcher_url": "http://x:18751",
                    "command": ["sh", "-c", "echo hi"],
                },
            )
        assert post.call_args.kwargs["json"] == {"command": ["sh", "-c", "echo hi"]}

    def test_missing_launcher_url_returns_400(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post("/fleet/upgrade", json={"strategy": "pip"})
        assert resp.status_code == 400

    def test_launcher_unreachable_returns_502(self, gateway):
        import httpx

        client = TestClient(gateway._build_http_app())
        post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/upgrade",
                json={"launcher_url": "http://dead:18751"},
            )
        assert resp.status_code == 502

    def test_failed_upgrade_surfaces_as_502_with_response(self, gateway):
        client = TestClient(gateway._build_http_app())
        post = _mock_post(
            status_code=500,
            payload={"ok": False, "returncode": 1, "stderr": "pip not found"},
        )
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/upgrade",
                json={"launcher_url": "http://x:18751", "strategy": "pip"},
            )
        assert resp.status_code == 502
        assert resp.json()["launcher_response"]["ok"] is False


class TestFleetReplaceWorker:
    """Coordinator-side /fleet/replace-worker drains + shutdowns + respawns."""

    def _register_worker(self, gateway, worker_id: str):
        ws = MagicMock()
        ws.send = AsyncMock()
        gateway._workers.register(
            worker_id, ws=ws, capabilities={"backend": "mlx"},
        )
        return ws

    def test_happy_path_drains_shutdowns_and_respawns(self, gateway):
        client = TestClient(gateway._build_http_app())
        ws = self._register_worker(gateway, "old-w")
        post = _mock_post(payload={"ok": True, "pid": 999})
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/replace-worker",
                json={
                    "target_worker_id": "old-w",
                    "launcher_url": "http://host:18751",
                    "launcher_token": "secret",
                    "worker": {"backend": "mlx", "model": "new/model-id"},
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["replaced_worker_id"] == "old-w"
        assert data["shutdown_sent"] is True
        assert data["launcher_response"]["pid"] == 999

        # Worker was marked draining.
        assert gateway._workers.get("old-w").draining is True
        # WS shutdown was sent.
        ws.send.assert_awaited()
        sent = ws.send.await_args.args[0]
        assert '"type": "shutdown"' in sent

        # Launch payload reached the launcher with the new model.
        launcher_body = post.call_args.kwargs["json"]
        assert launcher_body["model"] == "new/model-id"
        assert launcher_body["backend"] == "mlx"
        # Coordinator auto-filled controller URL.
        assert launcher_body["controller"].startswith("ws://")

    def test_unknown_worker_returns_404(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/replace-worker",
            json={"target_worker_id": "ghost", "launcher_url": "http://x"},
        )
        assert resp.status_code == 404

    def test_proceeds_even_when_worker_ws_send_fails(self, gateway):
        """If the worker is already gone, the launcher call should still
        fire — we don't want a half-dead worker to block the replacement."""
        client = TestClient(gateway._build_http_app())
        ws = self._register_worker(gateway, "stale-w")
        ws.send = AsyncMock(side_effect=Exception("ws already closed"))
        post = _mock_post(payload={"ok": True, "pid": 1})
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/replace-worker",
                json={
                    "target_worker_id": "stale-w",
                    "launcher_url": "http://x",
                    "worker": {"backend": "mlx"},
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["shutdown_sent"] is False
        # Replacement still went through.
        post.assert_awaited_once()

    def test_launcher_unreachable_returns_502_with_drain_state(self, gateway):
        import httpx

        client = TestClient(gateway._build_http_app())
        self._register_worker(gateway, "w1")
        post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/replace-worker",
                json={
                    "target_worker_id": "w1",
                    "launcher_url": "http://dead:18751",
                    "worker": {"backend": "mlx"},
                },
            )
        assert resp.status_code == 502
        data = resp.json()
        # Caller can tell the drain succeeded even though the spawn failed.
        assert data["drained_worker_id"] == "w1"
        assert "shutdown_sent" in data

    def test_missing_target_worker_id_returns_400(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/replace-worker",
            json={"launcher_url": "http://x", "worker": {}},
        )
        assert resp.status_code == 400
        client = TestClient(gateway._build_http_app())
        post = _mock_post(
            status_code=401, payload={"error": "invalid token"}
        )
        with patch("httpx.AsyncClient.post", post):
            resp = client.post(
                "/fleet/spawn",
                json={
                    "launcher_url": "http://x:18751",
                    "launcher_token": "wrong",
                    "worker": {"backend": "mlx"},
                },
            )
        # Coordinator marks the overall call as 502 (its upstream failed) but
        # preserves the launcher's status code and body so callers can see why.
        assert resp.status_code == 502
        data = resp.json()
        assert data["launcher_status"] == 401
        assert data["launcher_response"]["error"] == "invalid token"

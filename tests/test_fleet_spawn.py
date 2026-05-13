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

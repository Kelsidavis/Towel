"""Tests for gateway worker coordination."""

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from towel.agent.conversation import Role
from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.gateway.workers import WorkerRegistry
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore


class DummyWS:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


@pytest.fixture
def gateway(store):
    config = TowelConfig()
    agent = AgentRuntime(config)
    sessions = SessionManager(store=store)
    pin_store = SessionPinStore(path=store.store_dir / "session_pins.json")
    worker_state_store = WorkerStateStore(path=store.store_dir / "worker_state.json")
    return GatewayServer(
        config=config,
        agent=agent,
        sessions=sessions,
        pin_store=pin_store,
        worker_state_store=worker_state_store,
    )


class TestWorkerRegistry:
    def test_prefers_sticky_idle_worker(self):
        registry = WorkerRegistry()
        registry.register("worker-a", DummyWS())
        registry.register("worker-b", DummyWS())

        picked = registry.acquire(preferred_id="worker-b")

        assert picked is not None
        assert picked.id == "worker-b"

    def test_skips_busy_preferred_worker(self):
        registry = WorkerRegistry()
        registry.register("worker-a", DummyWS())
        registry.register("worker-b", DummyWS())
        registry.assign("worker-b", "job-1", "session-1")

        picked = registry.acquire(preferred_id="worker-b")

        assert picked is not None
        assert picked.id == "worker-a"

    def test_prefers_best_capability_match(self):
        registry = WorkerRegistry()
        registry.register(
            "worker-a",
            DummyWS(),
            {"backend": "mlx", "model": "repo/model-a", "modes": ["mlx_prompt"], "tools": False},
        )
        registry.register(
            "worker-b",
            DummyWS(),
            {
                "backend": "claude",
                "model": "repo/model-a",
                "modes": ["anthropic_messages"],
                "tools": False,
            },
        )
        registry.register(
            "worker-c",
            DummyWS(),
            {"backend": "mlx", "model": "repo/model-a", "modes": ["mlx_prompt"], "tools": True},
        )

        picked = registry.acquire(
            requirements={
                "backend": "mlx",
                "model": "repo/model-a",
                "mode": "mlx_prompt",
                "tools": False,
            }
        )

        assert picked is not None
        assert picked.id == "worker-a"

    def test_returns_none_when_no_worker_matches_required_backend(self):
        registry = WorkerRegistry()
        registry.register(
            "worker-a",
            DummyWS(),
            {"backend": "claude", "model": "repo/model-a", "modes": ["anthropic_messages"]},
        )

        picked = registry.acquire(
            requirements={"backend": "mlx", "model": "repo/model-a", "mode": "mlx_prompt"}
        )

        assert picked is None

    def test_skips_disabled_and_draining_workers(self):
        registry = WorkerRegistry()
        registry.register(
            "worker-a",
            DummyWS(),
            {"backend": "mlx", "model": "repo/model-a", "modes": ["mlx_prompt"]},
        )
        registry.register(
            "worker-b",
            DummyWS(),
            {"backend": "mlx", "model": "repo/model-a", "modes": ["mlx_prompt"]},
        )
        registry.set_enabled("worker-a", False)
        registry.set_draining("worker-b", True)

        picked = registry.acquire(requirements={"backend": "mlx", "mode": "mlx_prompt"})

        assert picked is None


class TestGatewayWorkerVisibility:
    def test_health_reports_worker_stats(self, gateway):
        gateway._workers.register("worker-a", DummyWS(), {"backend": "mlx"})
        gateway._workers.register("worker-b", DummyWS(), {"backend": "claude"})
        gateway._workers.assign("worker-b", "job-1", "session-1")

        client = TestClient(gateway._build_http_app())
        data = client.get("/health").json()

        assert data["workers"] == {
            "total": 2,
            "busy": 1,
            "idle": 1,
            "enabled": 2,
            "draining": 0,
            "disabled": 0,
        }

    def test_sessions_endpoint_includes_assigned_worker(self, gateway):
        session = gateway.sessions.get_or_create("chat-1")
        session.conversation.add(Role.USER, "hello")
        gateway._session_workers["chat-1"] = "worker-a"

        client = TestClient(gateway._build_http_app())
        data = client.get("/sessions").json()

        assert data["sessions"][0]["worker_id"] == "worker-a"


class TestGatewayScheduling:
    def test_select_worker_uses_controller_runtime_requirements(self, gateway):
        gateway._workers.register(
            "worker-claude",
            DummyWS(),
            {
                "backend": "claude",
                "model": gateway.config.model.name,
                "modes": ["anthropic_messages"],
            },
        )
        gateway._workers.register(
            "worker-mlx",
            DummyWS(),
            {
                "backend": "mlx",
                "model": gateway.config.model.name,
                "modes": ["mlx_prompt"],
                "tools": False,
            },
        )

        picked = gateway._select_worker("session-1")

        assert picked is not None
        assert picked.id == "worker-mlx"

    def test_pin_session_worker_overrides_default_selection(self, gateway):
        gateway._workers.register(
            "worker-a",
            DummyWS(),
            {
                "backend": "mlx",
                "model": gateway.config.model.name,
                "modes": ["mlx_prompt"],
                "tools": False,
            },
        )
        gateway._workers.register(
            "worker-b",
            DummyWS(),
            {
                "backend": "mlx",
                "model": gateway.config.model.name,
                "modes": ["mlx_prompt"],
                "tools": False,
            },
        )

        assert gateway.pin_session_worker("session-1", "worker-b") is True

        picked = gateway._select_worker("session-1")

        assert picked is not None
        assert picked.id == "worker-b"


class TestRemoteExecution:
    @pytest.mark.asyncio
    async def test_step_remote_inference_updates_session_from_worker_result(self, gateway):
        session = gateway.sessions.get_or_create("remote-step")
        session.conversation.add(Role.USER, "hello from controller")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-a", worker_ws, {"tools": False})
        gateway.agent.build_inference_request = lambda conversation: {  # type: ignore[attr-defined]
            "mode": "mlx_prompt",
            "prompt": f"messages:{len(conversation.messages)}",
        }

        task = asyncio.create_task(gateway._step_remote_inference("remote-step", session, worker))
        await asyncio.sleep(0)

        assert len(worker_ws.sent) == 1
        run_msg = worker_ws.sent[0]
        assert run_msg["type"] == "infer"
        job_id = run_msg["job_id"]

        await gateway._job_queues[job_id].put(
            {
                "type": "job_done",
                "job_id": job_id,
                "result": {
                    "text": "remote answer",
                    "metadata": {"backend": "remote", "tokens": 3},
                },
            }
        )

        response = await task

        assert response.content == "remote answer"
        assert response.metadata["backend"] == "remote"
        assert session.conversation.last is not None
        assert session.conversation.last.content == "remote answer"
        assert not worker.busy

    @pytest.mark.asyncio
    async def test_stream_remote_inference_forwards_events_and_updates_session(self, gateway):
        session = gateway.sessions.get_or_create("remote-stream")
        session.conversation.add(Role.USER, "stream please")
        client_ws = DummyWS()
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-a", worker_ws, {"tools": False})
        gateway.agent.build_inference_request = lambda conversation: {  # type: ignore[attr-defined]
            "mode": "mlx_prompt",
            "prompt": f"messages:{len(conversation.messages)}",
        }

        task = asyncio.create_task(
            gateway._stream_remote_inference(client_ws, "remote-stream", session, worker)
        )
        await asyncio.sleep(0)

        run_msg = worker_ws.sent[0]
        assert run_msg["type"] == "infer"
        job_id = run_msg["job_id"]
        await gateway._job_queues[job_id].put(
            {
                "type": "job_event",
                "job_id": job_id,
                "event": {
                    "type": "token",
                    "session": "remote-stream",
                    "content": "hel",
                },
            }
        )
        await gateway._job_queues[job_id].put(
            {
                "type": "job_event",
                "job_id": job_id,
                "event": {
                    "type": "response_complete",
                    "session": "remote-stream",
                    "content": "hello there",
                    "metadata": {"backend": "remote"},
                },
            }
        )
        await gateway._job_queues[job_id].put(
            {
                "type": "job_done",
                "job_id": job_id,
                "result": {"text": "hello there", "metadata": {}},
            }
        )

        await task

        assert client_ws.sent[0]["type"] == "token"
        assert client_ws.sent[1]["type"] == "response_complete"
        assert session.conversation.last is not None
        assert session.conversation.last.content == "hello there"

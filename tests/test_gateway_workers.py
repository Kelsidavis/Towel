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

    def test_prefers_less_loaded_worker_when_capability_tied(self):
        """When two workers are otherwise identical, the live cpu_pressure
        from the latest heartbeat should break the tie toward the calmer one."""
        registry = WorkerRegistry()
        base = {"backend": "ollama", "modes": ["ollama_chat"]}
        registry.register("hot", DummyWS(), {**base, "live_resources": {"cpu_pressure": 0.95}})
        registry.register("cool", DummyWS(), {**base, "live_resources": {"cpu_pressure": 0.05}})

        picked = registry.acquire(requirements={"backend": "ollama", "mode": "ollama_chat"})

        assert picked is not None
        assert picked.id == "cool"

    def test_cpu_pressure_penalty_does_not_override_required_backend(self):
        """A worker on the wrong backend must never win, no matter how idle —
        the cpu_pressure penalty is bounded at -15 while a backend mismatch
        costs -100."""
        registry = WorkerRegistry()
        registry.register(
            "wrong_backend_idle",
            DummyWS(),
            {"backend": "claude", "modes": ["anthropic_messages"],
             "live_resources": {"cpu_pressure": 0.0}},
        )
        registry.register(
            "right_backend_busyish",
            DummyWS(),
            {"backend": "ollama", "modes": ["ollama_chat"],
             "live_resources": {"cpu_pressure": 0.9}},
        )

        picked = registry.acquire(requirements={"backend": "ollama", "mode": "ollama_chat"})

        assert picked is not None
        assert picked.id == "right_backend_busyish"

    def test_missing_live_resources_does_not_crash_scoring(self):
        """Old workers (or ones running on hosts where the detector returned
        an empty dict) won't have ``live_resources`` in their capabilities.
        The scorer must tolerate that."""
        registry = WorkerRegistry()
        registry.register(
            "no_telemetry",
            DummyWS(),
            {"backend": "ollama", "modes": ["ollama_chat"]},  # no live_resources key
        )
        picked = registry.acquire(requirements={"backend": "ollama", "mode": "ollama_chat"})
        assert picked is not None
        assert picked.id == "no_telemetry"

    def test_quality_tier_helper_buckets_workers_correctly(self):
        from towel.nodes.roles import worker_quality_tier

        assert worker_quality_tier({"backend": "claude"}) == "high"
        assert worker_quality_tier(
            {"total_vram_mb": 24000, "context_window": 131072}
        ) == "high"
        assert worker_quality_tier(
            {"total_vram_mb": 6000, "context_window": 32768}
        ) == "medium"
        assert worker_quality_tier({"context_window": 8192}) == "low"
        assert worker_quality_tier({}) == "low"

    def test_garbage_cpu_pressure_value_is_ignored(self):
        """A misbehaving worker reporting a non-numeric cpu_pressure shouldn't
        crash dispatch."""
        registry = WorkerRegistry()
        registry.register(
            "garbage",
            DummyWS(),
            {
                "backend": "ollama",
                "modes": ["ollama_chat"],
                "live_resources": {"cpu_pressure": "not-a-number"},
            },
        )
        picked = registry.acquire(requirements={"backend": "ollama", "mode": "ollama_chat"})
        assert picked is not None
        assert picked.id == "garbage"


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
            "stuck": 0,
        }

    def test_sessions_endpoint_includes_assigned_worker(self, gateway):
        session = gateway.sessions.get_or_create("chat-1")
        session.conversation.add(Role.USER, "hello")
        gateway._session_workers["chat-1"] = "worker-a"

        client = TestClient(gateway._build_http_app())
        data = client.get("/sessions").json()

        assert data["sessions"][0]["worker_id"] == "worker-a"


class TestGatewayScheduling:
    def test_dispatcher_picks_an_available_worker(self, gateway):
        """The legacy ``_select_worker`` method was removed when the
        Dispatcher landed. This test now exercises the public dispatcher
        contract: with two healthy workers and no session affinity, the
        dispatcher returns a worker rather than ``None``."""
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

        assert gateway._dispatcher is not None
        decision = gateway._dispatcher.select_for_session("session-1", intent="task")

        assert decision.worker is not None
        assert decision.worker.id in {"worker-claude", "worker-mlx"}

    def test_pin_session_worker_overrides_default_selection(self, gateway):
        """A session pin forces the dispatcher to return that specific
        worker regardless of which worker would otherwise score highest."""
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

        assert gateway._dispatcher is not None
        decision = gateway._dispatcher.select_for_session("session-1", intent="task")

        assert decision.worker is not None
        assert decision.worker.id == "worker-b"
        assert decision.reason == "pinned"


class TestInferenceTimeoutConfig:
    @pytest.mark.asyncio
    async def test_config_value_is_passed_to_wait_for(
        self, gateway, monkeypatch
    ):
        """The worker_inference_timeout config knob has to actually
        reach asyncio.wait_for, otherwise the runtime ignores it. We
        record every timeout passed during a successful inference and
        assert it matches the configured value."""
        # Bump the timeout so it's distinguishable from defaults.
        gateway.config.worker_inference_timeout = 777.0

        session = gateway.sessions.get_or_create("timeout-cfg")
        session.conversation.add(Role.USER, "hi")
        worker_ws = DummyWS()
        worker = gateway._workers.register("w", worker_ws, {"tools": False})
        gateway.agent.build_inference_request = lambda conversation: {  # type: ignore[attr-defined]
            "mode": "mlx_prompt", "prompt": "x",
        }

        recorded_timeouts: list[float] = []
        orig_wait_for = asyncio.wait_for

        async def spying_wait_for(awaitable, timeout):
            recorded_timeouts.append(timeout)
            return await orig_wait_for(awaitable, timeout)

        monkeypatch.setattr(asyncio, "wait_for", spying_wait_for)

        task = asyncio.create_task(
            gateway._step_remote_inference("timeout-cfg", session, worker)
        )
        await asyncio.sleep(0)
        # Find the job queue and inject a completion.
        job_id = worker_ws.sent[0]["job_id"]
        await gateway._job_queues[job_id].put({
            "type": "job_done",
            "job_id": job_id,
            "result": {"text": "ok", "metadata": {}},
        })
        await task
        # 777.0 must show up among the captured timeouts (the
        # _remote_generate path is the only caller that uses the
        # configured value; other wait_for calls use shorter
        # heartbeat / ack timeouts).
        assert 777.0 in recorded_timeouts


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
        # The remote-generate path sends type="run" (full conversation
        # transfer). ``infer`` is reserved for the lighter classifier path
        # in /_classify_on_worker and friends.
        assert run_msg["type"] == "run"
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
        # The remote-generate path sends type="run" (full conversation
        # transfer). ``infer`` is reserved for the lighter classifier path
        # in /_classify_on_worker and friends.
        assert run_msg["type"] == "run"
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

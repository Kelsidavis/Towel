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
    def test_step_remote_inference_has_loop_detection(self):
        """Source-level check that the agent-loop body contains the
        loop-detection logic — a worker stuck calling the same tool
        repeatedly would otherwise loop until MAX_TOOL_ITERATIONS =
        999 (~5h on a 20s-per-call worker). Full end-to-end test
        would need a real fake-worker process; this source check is
        sufficient to guard against accidental removal."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._step_remote_inference_inner)
        # Loop-detection fingerprint, threshold, and break path.
        assert "last_call_fingerprints" in src
        assert "LOOP_REPEAT_LIMIT" in src
        assert "loop_detected" in src
        # The message returned on loop detection should explain itself.
        assert "stuck" in src.lower() or "stopping" in src.lower()

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
    async def test_quick_remote_infer_timeout_surfaces_useful_error(self, gateway):
        """`asyncio.TimeoutError` stringifies to "" — without an explicit
        catch the caller (simple_ask) returns `{"error": ""}` HTTP 500
        with no indication the worker timed out. Convert it to a
        RuntimeError carrying the worker id and the timeout."""
        session = gateway.sessions.get_or_create("timeout-sess")
        session.conversation.add(Role.USER, "hi")
        worker_ws = DummyWS()
        worker = gateway._workers.register("slow-worker", worker_ws, {"tools": False})

        # Patch the wait_for timeout down so the test doesn't actually
        # block for 60s.
        import asyncio as _asyncio
        from unittest.mock import patch

        orig_wait_for = _asyncio.wait_for

        async def fast_timeout(awaitable, timeout):
            return await orig_wait_for(awaitable, 0.05)

        with patch("asyncio.wait_for", fast_timeout):
            with pytest.raises(RuntimeError) as exc:
                await gateway._quick_remote_infer("timeout-sess", session, worker)
        msg = str(exc.value)
        assert "slow-worker" in msg
        assert "60s" in msg or "did not respond" in msg
        # Worker assignment must be released even after the timeout.
        assert worker.busy is False

    @pytest.mark.asyncio
    async def test_remote_generate_injects_coordinator_memory(self, gateway):
        """The "run" payload must include a synthetic system message
        carrying coordinator-side memory. Workers have empty memory
        stores; without the injection they can't answer "what's my
        favorite number?" even though the answer lives on this host."""
        from unittest.mock import MagicMock

        # Plug a stub memory store onto the agent. We only care that
        # to_prompt_block(query=last_user) gets called and its return
        # value lands in the payload's conversation.messages.
        stub_memory = MagicMock()
        stub_memory.to_prompt_block.return_value = "USER_NAME=Kelsi\nFAV_NUMBER=42"
        gateway.agent.memory = stub_memory

        session = gateway.sessions.get_or_create("mem-inject")
        session.conversation.add(Role.USER, "what is my favorite number?")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-mem", worker_ws, {"tools": False})

        task = asyncio.create_task(
            gateway._step_remote_inference("mem-inject", session, worker)
        )
        await asyncio.sleep(0)

        # The payload should carry the memory as a SYSTEM message at the
        # head of the conversation. We don't need the inference to
        # complete here — only inspect what was sent.
        assert len(worker_ws.sent) == 1
        run_msg = worker_ws.sent[0]
        msgs = run_msg["conversation"]["messages"]
        assert msgs[0]["role"] == "system"
        assert "FAV_NUMBER=42" in msgs[0]["content"]
        assert msgs[0]["metadata"]["source"] == "coord_memory_injection"
        # Original user message is still present after the injection.
        assert any(m["role"] == "user" for m in msgs[1:])

        # Confirm the recall query was the user's last turn.
        stub_memory.to_prompt_block.assert_called_once()
        kwargs = stub_memory.to_prompt_block.call_args.kwargs
        assert kwargs["query"] == "what is my favorite number?"

        # Finally drain the task so we don't leak it.
        job_id = run_msg["job_id"]
        await gateway._job_queues[job_id].put({
            "type": "job_done", "job_id": job_id,
            "result": {"text": "42", "metadata": {}},
        })
        await task

        # And the original session.conversation must NOT have been
        # mutated by the injection.
        roles = [m.role.value for m in session.conversation.messages]
        assert "system" not in roles[:1]

    @pytest.mark.asyncio
    async def test_remote_generate_omits_memory_block_when_empty(self, gateway):
        """An empty memory corpus yields "" from to_prompt_block — the
        payload must NOT carry a stray empty system message in that case,
        because some worker runtimes treat any leading system message as
        an override of their default identity prompt."""
        from unittest.mock import MagicMock

        stub_memory = MagicMock()
        stub_memory.to_prompt_block.return_value = ""
        gateway.agent.memory = stub_memory

        session = gateway.sessions.get_or_create("mem-empty")
        session.conversation.add(Role.USER, "hi")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-empty", worker_ws, {"tools": False})

        task = asyncio.create_task(
            gateway._step_remote_inference("mem-empty", session, worker)
        )
        await asyncio.sleep(0)

        msgs = worker_ws.sent[0]["conversation"]["messages"]
        assert msgs[0]["role"] == "user"  # no leading synthetic system msg

        job_id = worker_ws.sent[0]["job_id"]
        await gateway._job_queues[job_id].put({
            "type": "job_done", "job_id": job_id,
            "result": {"text": "hello", "metadata": {}},
        })
        await task

    @pytest.mark.asyncio
    async def test_iter_remote_tokens_sends_cancel_on_early_exit(self, gateway):
        """When the SSE client disconnects (or the generator is
        otherwise cancelled), the coordinator must tell the worker
        to stop generating — otherwise the worker burns cycles
        producing tokens that nobody is reading."""
        session = gateway.sessions.get_or_create("cancel-sess")
        session.conversation.add(Role.USER, "say hi")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-c", worker_ws, {"tools": False})

        gen = gateway.iter_remote_tokens("cancel-sess", session, worker)
        # Drive the generator one step so it actually issues the
        # `run` to the worker and suspends on `queue.get()`.
        gen_task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0)
        await asyncio.sleep(0)  # second yield so the send hits the WS

        # Worker received the "run" message.
        assert worker_ws.sent, "worker received no messages"
        assert worker_ws.sent[0]["type"] == "run"
        job_id = worker_ws.sent[0]["job_id"]

        # Cancel the suspended task — finally block should fire.
        gen_task.cancel()
        try:
            await gen_task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass
        # Let the cancel propagate through any pending awaitables.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Worker must have received a cancel_job for that job_id.
        cancel_msgs = [m for m in worker_ws.sent if m.get("type") == "cancel_job"]
        assert cancel_msgs, (
            f"expected a cancel_job after early task cancel, got: "
            f"{[m.get('type') for m in worker_ws.sent]}"
        )
        assert cancel_msgs[0]["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_quick_remote_infer_stamps_timing_on_timeout(self, gateway):
        """Operators looking at /dispatch/recent for a failing primary
        need to see how long the failing attempt took. Without this
        stamp, the primary's decision shows total_ms=None when
        _quick_remote_infer raises — operators can't tell "timed out
        at 60s" apart from "errored instantly"."""
        session = gateway.sessions.get_or_create("timing-stamp")
        session.conversation.add(Role.USER, "hi")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-ts", worker_ws, {"tools": False})

        # Force fast timeout so the test doesn't actually wait.
        gateway.config.chat_fast_timeout = 0.05

        # Trigger a real dispatch so the decision exists in history.
        await gateway._route_by_role("hi", "timing-stamp")
        assert gateway._dispatcher is not None
        before_decision = gateway._dispatcher.last_decision_for_session("timing-stamp")
        assert before_decision is not None
        assert before_decision.total_ms is None

        with pytest.raises(RuntimeError):
            await gateway._quick_remote_infer("timing-stamp", session, worker)

        # The decision now has a total_ms reflecting the timeout.
        decision = gateway._dispatcher.last_decision_for_session("timing-stamp")
        assert decision is not None
        assert decision.total_ms is not None
        # Should be roughly the timeout duration (50ms) — allow generous range.
        assert decision.total_ms > 10
        assert decision.total_ms < 5000

    @pytest.mark.asyncio
    async def test_quick_remote_infer_sends_cancel_on_timeout(self, gateway):
        """`_quick_remote_infer` is the chat-fast path. A worker
        timeout used to leave the worker still generating in the
        background — same waste as the streaming paths. Verify the
        cancel_job fires when the wait_for times out."""
        session = gateway.sessions.get_or_create("qri-timeout")
        session.conversation.add(Role.USER, "hi")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-qto", worker_ws, {"tools": False})

        # Patch chat_fast_timeout so we don't actually wait 60s.
        gateway.config.chat_fast_timeout = 0.05

        with pytest.raises(RuntimeError):
            await gateway._quick_remote_infer("qri-timeout", session, worker)

        # First message is the `infer`, then the `cancel_job`.
        types = [m.get("type") for m in worker_ws.sent]
        assert "infer" in types
        assert "cancel_job" in types

    @pytest.mark.asyncio
    async def test_iter_remote_tokens_no_cancel_on_normal_completion(self, gateway):
        """When the worker emits job_done normally, we should NOT
        send a redundant cancel_job."""
        session = gateway.sessions.get_or_create("clean-sess")
        session.conversation.add(Role.USER, "hi")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-d", worker_ws, {"tools": False})

        gen = gateway.iter_remote_tokens("clean-sess", session, worker)
        gen_task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0)

        job_id = worker_ws.sent[0]["job_id"]
        # Send job_done to trigger clean exit.
        await gateway._job_queues[job_id].put({"type": "job_done", "job_id": job_id})
        try:
            await gen_task
        except StopAsyncIteration:
            pass

        # No cancel_job sent.
        assert not any(m.get("type") == "cancel_job" for m in worker_ws.sent)

    @pytest.mark.asyncio
    async def test_iter_remote_tokens_also_injects_memory(self, gateway):
        """The SSE streaming path used by /v1/chat/completions has the
        same memory-blindness gap — make sure the injection covers it
        too. We don't run the full generator to completion; we only
        peek at the first thing the worker sees."""
        from unittest.mock import MagicMock

        stub_memory = MagicMock()
        stub_memory.to_prompt_block.return_value = "FAV=42"
        gateway.agent.memory = stub_memory

        session = gateway.sessions.get_or_create("stream-mem")
        session.conversation.add(Role.USER, "what is my favorite number?")
        worker_ws = DummyWS()
        worker = gateway._workers.register("worker-stream", worker_ws, {"tools": False})

        gen = gateway.iter_remote_tokens("stream-mem", session, worker)
        # Pull one tick of the generator so the WS send fires.
        gen_task = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0)

        assert worker_ws.sent  # WS send fired
        run_msg = worker_ws.sent[0]
        assert run_msg["stream"] is True
        msgs = run_msg["conversation"]["messages"]
        assert msgs[0]["role"] == "system"
        assert "FAV=42" in msgs[0]["content"]

        # Drain: feed a job_done so the generator can complete cleanly.
        job_id = run_msg["job_id"]
        await gateway._job_queues[job_id].put({"type": "job_done", "job_id": job_id})
        # Discard the StopAsyncIteration the next yield will raise.
        try:
            await gen_task
        except StopAsyncIteration:
            pass

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

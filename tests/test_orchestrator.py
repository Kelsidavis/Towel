"""Tests for the multi-agent orchestrator."""

import asyncio

from towel.agent.orchestrator import (
    ROLE_PROMPTS,
    AgentTask,
    Orchestrator,
    OrchestratorResult,
)


class TestAgentTask:
    def test_defaults(self):
        t = AgentTask(role="coder", prompt="write hello world")
        assert t.status == "pending"
        assert t.result == ""
        assert t.depends_on == []

    def test_to_dict(self):
        t = AgentTask(
            role="reviewer",
            prompt="check code",
            status="completed",
            elapsed=2.5,
            result="looks good",
        )
        d = t.to_dict()
        assert d["role"] == "reviewer"
        assert d["status"] == "completed"
        assert d["result_length"] == 10


class TestOrchestratorResult:
    def test_success_all_completed(self):
        tasks = [
            AgentTask(role="coder", prompt="x", status="completed", result="code"),
            AgentTask(role="reviewer", prompt="y", status="completed", result="ok"),
        ]
        r = OrchestratorResult(tasks=tasks)
        assert r.success

    def test_failure_if_any_failed(self):
        tasks = [
            AgentTask(role="coder", prompt="x", status="completed", result="code"),
            AgentTask(role="reviewer", prompt="y", status="failed", result="error"),
        ]
        r = OrchestratorResult(tasks=tasks)
        assert not r.success

    def test_summary(self):
        tasks = [
            AgentTask(role="coder", prompt="write", status="completed", elapsed=1.2, result="done"),
        ]
        r = OrchestratorResult(tasks=tasks, total_elapsed=1.5)
        s = r.summary()
        assert "1 tasks" in s
        assert "coder" in s
        assert "completed" in s


class TestRolePrompts:
    def test_all_roles_have_prompts(self):
        expected = {
            "coder",
            "researcher",
            "reviewer",
            "writer",
            "architect",
            "tester",
            "debugger",
            "default",
        }
        assert expected.issubset(set(ROLE_PROMPTS.keys()))

    def test_prompts_are_nonempty(self):
        for role, prompt in ROLE_PROMPTS.items():
            assert len(prompt) > 20, f"Prompt for {role} is too short"


class TestOrchestrator:
    def test_instantiation(self):
        from towel.config import TowelConfig

        config = TowelConfig()
        orch = Orchestrator(config)
        assert orch is not None

    def test_task_dependency_tracking(self):
        tasks = [
            AgentTask(role="architect", prompt="design"),
            AgentTask(role="coder", prompt="implement", depends_on=[0]),
            AgentTask(role="reviewer", prompt="review", depends_on=[1]),
        ]
        assert tasks[1].depends_on == [0]
        assert tasks[2].depends_on == [1]


class _RecordingDispatcher:
    """In-process RoleDispatcher used to verify the orchestrator's
    role-routing contract without touching real workers."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def dispatch_role_task(
        self,
        role: str,
        role_system: str,
        prompt: str,
        *,
        session_id: str,
        max_tokens: int,
        temperature: float,
        with_tools: bool,
    ) -> str:
        self.calls.append({
            "role": role,
            "role_system": role_system,
            "prompt": prompt,
            "session_id": session_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "with_tools": with_tools,
        })
        return f"[{role} result for: {prompt[:40]}]"


class TestOrchestratorWithDispatcher:
    """End-to-end: confirm that `dispatcher`, when set, replaces the
    local AgentRuntime path. These tests pin the contract the gateway
    must satisfy (see RoleDispatcher Protocol)."""

    def test_dispatcher_invoked_per_task(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="architect", prompt="design API"),
            AgentTask(role="coder", prompt="write impl"),
        ]
        result = asyncio.run(orch.run("Build a thing", tasks))
        assert result.success
        assert len(dispatcher.calls) == 2
        assert dispatcher.calls[0]["role"] == "architect"
        assert dispatcher.calls[1]["role"] == "coder"
        # Each subtask must get its own session id so role affinities
        # don't bleed between them.
        assert dispatcher.calls[0]["session_id"] != dispatcher.calls[1]["session_id"]
        # The dispatcher receives the role's system identity verbatim
        # — the gateway needs this to set identity_override.
        assert "software engineer" in dispatcher.calls[1]["role_system"].lower() \
            or "code" in dispatcher.calls[1]["role_system"].lower()

    def test_dispatcher_receives_dependency_context(self):
        """When a task depends on a prior task, its prompt must
        include the prior task's result — that's how piecemeal
        coordination actually shares state across workers."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="architect", prompt="design"),
            AgentTask(role="coder", prompt="implement", depends_on=[0]),
        ]
        asyncio.run(orch.run("Build a thing", tasks))
        # The second call's prompt must mention the first task's result
        # (the recording dispatcher echoes the role+prompt as its result).
        second_prompt = dispatcher.calls[1]["prompt"]
        assert "Result from architect" in second_prompt
        # The architect role's result starts with "[architect result for:"
        # — its exact suffix depends on prompt truncation, so just check
        # the role tag round-tripped into the next subtask's prompt.
        assert "[architect result for:" in second_prompt

    def test_dispatcher_run_parallel_independent_sessions(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="file a"),
            AgentTask(role="coder", prompt="file b"),
            AgentTask(role="coder", prompt="file c"),
        ]
        asyncio.run(orch.run_parallel("Build three files", tasks))
        assert all(t.status == "completed" for t in tasks)
        # Three distinct session_ids — parallel subtasks must not
        # share a session or the dispatcher's session-pinning code
        # serializes them onto one worker.
        sids = {c["session_id"] for c in dispatcher.calls}
        assert len(sids) == 3

    def test_with_tools_flows_through_to_dispatcher(self):
        """A subtask declared with_tools=True must hand that down to
        the dispatcher — without this, "coder" subtasks can never call
        write_file etc., which makes piecemeal artifact building
        impossible regardless of how good the planning is."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="writer", prompt="explain x"),               # no tools
            AgentTask(role="coder", prompt="write x.py", with_tools=True),
        ]
        asyncio.run(orch.run("g", tasks))
        assert dispatcher.calls[0]["with_tools"] is False
        assert dispatcher.calls[1]["with_tools"] is True

    def test_dispatcher_error_propagates_as_failed_task(self):
        from towel.config import TowelConfig

        class _BrokenDispatcher:
            async def dispatch_role_task(self, *args, **kwargs) -> str:  # noqa: ARG002
                raise RuntimeError("worker timed out")

        # max_attempts=1 disables the default retry so the test is
        # checking the failure-propagation path, not the retry path.
        orch = Orchestrator(
            TowelConfig(), dispatcher=_BrokenDispatcher(), max_attempts=1,
        )
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run("g", tasks))
        assert not result.success
        assert tasks[0].status == "failed"
        assert "worker timed out" in tasks[0].result
        assert tasks[0].attempts == 1

    def test_retry_recovers_when_second_attempt_succeeds(self):
        """When a subtask fails once then succeeds, the orchestrator
        marks the task completed and records the attempt count.
        This is the codex-style "primary worker emitted empty text →
        alt worker answered" pattern."""
        from towel.config import TowelConfig

        attempts = {"count": 0}

        class _FlakyDispatcher:
            async def dispatch_role_task(self, *args, **kwargs) -> str:  # noqa: ARG002
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("primary returned empty")
                return "real answer on retry"

        orch = Orchestrator(TowelConfig(), dispatcher=_FlakyDispatcher())
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert tasks[0].status == "completed"
        assert tasks[0].result == "real answer on retry"
        assert tasks[0].attempts == 2

    def test_retry_gives_up_after_max_attempts(self):
        from towel.config import TowelConfig

        attempt_log: list[int] = []

        class _AlwaysFails:
            async def dispatch_role_task(self, *args, **kwargs) -> str:  # noqa: ARG002
                attempt_log.append(1)
                raise RuntimeError(f"fail #{len(attempt_log)}")

        orch = Orchestrator(
            TowelConfig(), dispatcher=_AlwaysFails(), max_attempts=3,
        )
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run("g", tasks))
        assert not result.success
        # Tried exactly max_attempts times.
        assert len(attempt_log) == 3
        assert tasks[0].attempts == 3
        # Final error message reflects the last failure.
        assert "fail #3" in tasks[0].result

    def test_retry_max_attempts_floor_is_one(self):
        """max_attempts<=0 should clamp to 1 — orchestrator must always
        try at least once per subtask, never zero-attempts."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher, max_attempts=0)
        tasks = [AgentTask(role="coder", prompt="x")]
        asyncio.run(orch.run("g", tasks))
        assert len(dispatcher.calls) == 1
        assert tasks[0].attempts == 1

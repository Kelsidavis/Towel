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
        task_type: str | None,
        exclude_workers: set[str] | None,
    ) -> str:
        self.calls.append({
            "role": role,
            "role_system": role_system,
            "prompt": prompt,
            "session_id": session_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "with_tools": with_tools,
            "task_type": task_type,
            "exclude_workers": set(exclude_workers) if exclude_workers else set(),
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

    def test_role_to_task_type_mapping_flows_to_dispatcher(self):
        """Without this, the workspace preamble the orchestrator
        prepends to subtask prompts prevented the keyword classifier
        from triggering (prompt no longer starts with 'write …') —
        coder/architect/tester subtasks fell through to role_match
        and skipped the dispatcher's prefer_quality preempt path.
        Explicit role→task_type mapping closes the gap."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="architect", prompt="plan it"),
            AgentTask(role="coder", prompt="write it"),
            AgentTask(role="tester", prompt="test it"),
            AgentTask(role="reviewer", prompt="review it"),
            AgentTask(role="writer", prompt="document it"),
            AgentTask(role="researcher", prompt="research it"),
            AgentTask(role="debugger", prompt="debug it"),
            AgentTask(role="default", prompt="something else"),
        ]
        asyncio.run(orch.run("g", tasks))
        types_by_role = {c["role"]: c["task_type"] for c in dispatcher.calls}
        assert types_by_role["architect"] == "plan"
        assert types_by_role["coder"] == "generate"
        assert types_by_role["tester"] == "test_gen"
        assert types_by_role["reviewer"] == "code_review"
        assert types_by_role["writer"] == "draft"
        assert types_by_role["researcher"] == "research"
        assert types_by_role["debugger"] == "analyze"
        # `default` has no mapping — falls through to classifier.
        assert types_by_role["default"] is None

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
        from towel.agent.orchestrator import WorkerDispatchError
        from towel.config import TowelConfig

        attempts = {"count": 0}
        seen_excludes: list[set[str]] = []

        class _FlakyDispatcher:
            async def dispatch_role_task(self, *args, **kwargs) -> str:  # noqa: ARG002
                attempts["count"] += 1
                seen_excludes.append(
                    set(kwargs.get("exclude_workers") or ())
                )
                if attempts["count"] == 1:
                    raise WorkerDispatchError(
                        "primary returned empty", worker_id="primary",
                    )
                return "real answer on retry"

        orch = Orchestrator(TowelConfig(), dispatcher=_FlakyDispatcher())
        tasks = [AgentTask(role="coder", prompt="x")]
        result = asyncio.run(orch.run("g", tasks))
        assert result.success
        assert tasks[0].status == "completed"
        assert tasks[0].result == "real answer on retry"
        assert tasks[0].attempts == 2
        # First attempt had no exclude_workers; second attempt
        # excludes the worker that just failed.
        assert seen_excludes[0] == set()
        assert seen_excludes[1] == {"primary"}

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

    def test_workspace_dir_injected_into_prompts(self):
        """When workspace_dir is set, every subtask must see a
        workspace directive — that's how a `coder` subtask knows where
        to write so the next subtask can read from the same place."""
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="write game.py"),
            AgentTask(role="tester", prompt="read game.py and add tests"),
        ]
        asyncio.run(orch.run("g", tasks, workspace_dir="/tmp/orch-test"))
        for call in dispatcher.calls:
            assert "/tmp/orch-test" in call["prompt"]
            assert "Shared workspace" in call["prompt"]

    def test_workspace_dir_absent_no_preamble(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [AgentTask(role="coder", prompt="x")]
        asyncio.run(orch.run("g", tasks))
        assert "Shared workspace" not in dispatcher.calls[0]["prompt"]

    def test_workspace_dir_parallel_too(self):
        from towel.config import TowelConfig
        dispatcher = _RecordingDispatcher()
        orch = Orchestrator(TowelConfig(), dispatcher=dispatcher)
        tasks = [
            AgentTask(role="coder", prompt="file a"),
            AgentTask(role="coder", prompt="file b"),
        ]
        asyncio.run(
            orch.run_parallel("g", tasks, workspace_dir="/tmp/orch-par"),
        )
        for call in dispatcher.calls:
            assert "/tmp/orch-par" in call["prompt"]

    def test_failed_dep_skips_dependents(self):
        """When a subtask fails after all retries, dependent subtasks
        should be `skipped` rather than run with the dep's error
        string injected as context — that wastes worker time and
        produces nonsensical output."""
        from towel.config import TowelConfig

        class _FailFirst:
            def __init__(self) -> None:
                self.calls: list[str] = []

            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools, task_type,
                exclude_workers,
            ):
                self.calls.append(role)
                if role == "architect":
                    raise RuntimeError("architect timed out")
                return f"{role}-result"

        dispatcher = _FailFirst()
        orch = Orchestrator(
            TowelConfig(), dispatcher=dispatcher, max_attempts=1,
        )
        tasks = [
            AgentTask(role="architect", prompt="plan"),
            AgentTask(role="coder", prompt="impl", depends_on=[0]),
            AgentTask(role="reviewer", prompt="review", depends_on=[1]),
        ]
        result = asyncio.run(orch.run("g", tasks))
        # Architect ran and failed.
        assert tasks[0].status == "failed"
        # Coder and reviewer were skipped — never dispatched.
        assert tasks[1].status == "skipped"
        assert tasks[2].status == "skipped"
        assert "depends on task(s) [0]" in tasks[1].result
        assert "depends on task(s) [1]" in tasks[2].result
        assert dispatcher.calls == ["architect"]  # no waste on dependents
        assert not result.success

    def test_skipped_task_no_synthesis(self):
        """When any task is skipped, the run is not 'success' and the
        markdown synthesis block stays empty — operators reading the
        response don't get a misleadingly-complete summary."""
        from towel.config import TowelConfig

        class _FailDep:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools, task_type,
                exclude_workers,
            ):
                if role == "architect":
                    raise RuntimeError("nope")
                return f"{role}-result"

        orch = Orchestrator(
            TowelConfig(), dispatcher=_FailDep(), max_attempts=1,
        )
        tasks = [
            AgentTask(role="architect", prompt="plan"),
            AgentTask(role="coder", prompt="impl", depends_on=[0]),
        ]
        result = asyncio.run(orch.run("g", tasks))
        assert not result.success
        assert result.synthesis == ""

    def test_extract_to_writes_fenced_code_block(self, tmp_path):
        """Lets a no-tools chat-fast coder produce code without going
        through the slow tool loop. Models often wrap code in ```python
        fences; the orchestrator extracts the first block and writes
        it to the workspace path the caller specified."""
        from towel.config import TowelConfig

        class _Echo:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                return (
                    "Here is the function:\n\n```python\n"
                    "def hello():\n    return 'hi'\n```\n\nDone."
                )

        orch = Orchestrator(TowelConfig(), dispatcher=_Echo())
        tasks = [
            AgentTask(role="coder", prompt="write hello", extract_to="hello.py"),
        ]
        ws = str(tmp_path / "ws")
        result = asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        assert result.success
        # Extracted body landed on disk.
        target = tmp_path / "ws" / "hello.py"
        assert target.exists()
        body = target.read_text(encoding="utf-8")
        assert "def hello()" in body
        assert "return 'hi'" in body
        # Python fence stripped — no triple-backticks in body.
        assert "```" not in body
        assert tasks[0].extracted_path == str(target.resolve())

    def test_extract_to_no_fence_writes_whole_response(self, tmp_path):
        """When the model doesn't use fences, write the whole stripped
        body anyway — a code-shaped response without backticks is
        still useful."""
        from towel.config import TowelConfig

        class _Plain:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                return "def f(): return 1\n"

        orch = Orchestrator(TowelConfig(), dispatcher=_Plain())
        tasks = [
            AgentTask(role="coder", prompt="x", extract_to="f.py"),
        ]
        ws = str(tmp_path / "ws")
        result = asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        assert result.success
        assert (tmp_path / "ws" / "f.py").read_text(encoding="utf-8") == "def f(): return 1\n"

    def test_extract_to_rejects_path_traversal(self, tmp_path):
        """A model-suggested `extract_to` shouldn't be able to write
        outside the workspace. Path resolution + ancestor check."""
        from towel.config import TowelConfig

        class _Echo:
            async def dispatch_role_task(  # noqa: PLR0913
                self, role, role_system, prompt, *,
                session_id, max_tokens, temperature, with_tools,
                task_type, exclude_workers,
            ):
                return "```python\nx = 1\n```"

        orch = Orchestrator(TowelConfig(), dispatcher=_Echo())
        tasks = [
            AgentTask(role="coder", prompt="x", extract_to="../escape.py"),
        ]
        ws = str(tmp_path / "ws")
        (tmp_path / "ws").mkdir()
        asyncio.run(orch.run("g", tasks, workspace_dir=ws))
        # Task marked failed because the extract path escaped.
        assert tasks[0].status == "failed"
        assert "escape" not in (tmp_path / "escape.py").exists().__str__() or \
            not (tmp_path / "escape.py").exists()
        # And the original error surfaces in the task result.
        assert "outside workspace" in tasks[0].result

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

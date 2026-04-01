"""Tests for the multi-agent orchestrator."""

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

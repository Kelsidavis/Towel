"""Tests for autonomous agents."""
import pytest
from towel.agent.agents import (
    AutonomousAgent, create_agent, delete_agent, list_agents,
    get_agent, log_agent_action, AGENTS_FILE,
)


class TestAutonomousAgent:
    def test_roundtrip(self):
        a = AutonomousAgent(name="test", goal="do stuff", tools=["git_status"])
        d = a.to_dict()
        a2 = AutonomousAgent.from_dict(d)
        assert a2.name == "test"
        assert a2.goal == "do stuff"
        assert a2.tools == ["git_status"]


class TestAgentStorage:
    @pytest.fixture(autouse=True)
    def tmp_storage(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.agent.agents.AGENTS_FILE", tmp_path / "agents.json")

    def test_create_and_list(self):
        create_agent("bot", "monitor things")
        agents = list_agents()
        assert len(agents) == 1
        assert agents[0].name == "bot"

    def test_get(self):
        create_agent("x", "goal")
        assert get_agent("x") is not None
        assert get_agent("y") is None

    def test_delete(self):
        create_agent("temp", "temp goal")
        assert delete_agent("temp")
        assert len(list_agents()) == 0

    def test_log_action(self):
        create_agent("logger", "log stuff")
        log_agent_action("logger", "check", "all good")
        a = get_agent("logger")
        assert a is not None
        assert len(a.logs) == 1
        assert a.total_runs == 1

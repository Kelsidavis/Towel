"""Tests for autonomous agents."""

import pytest

from towel.agent.agents import (
    AutonomousAgent,
    create_agent,
    delete_agent,
    get_agent,
    list_agents,
    log_agent_action,
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

    def test_corrupt_file_backed_up_on_load(self, tmp_path):
        """If _load_agents returns [] on corruption, the next create
        would overwrite the corrupt file with just the new agent —
        every prior agent gone. Rename the bad file aside instead.
        Same pattern persistence stores got (5512834, 98d1c68,
        8a86987)."""
        from towel.agent import agents as _agents
        # Write a corrupt file in the test's monkeypatched location.
        bad = _agents.AGENTS_FILE
        bad.write_text("{ not valid json")

        # create_agent triggers _load_agents → fails → backs up →
        # writes new file with one entry.
        create_agent("after-corruption", "test")

        # The bad bytes are saved aside under a corrupted-* sibling.
        backups = list(bad.parent.glob(f"{bad.name}.corrupted-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "{ not valid json"
        # The fresh file has just the new agent.
        agents = list_agents()
        assert len(agents) == 1
        assert agents[0].name == "after-corruption"

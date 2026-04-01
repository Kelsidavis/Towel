"""Tests for agent creation, deletion, and cloning."""

import pytest

from towel.cli.agent_mgr import (
    clone_agent,
    create_agent,
    delete_agent,
    load_user_agents,
)
from towel.config import TowelConfig


@pytest.fixture(autouse=True)
def isolate_agents(tmp_path, monkeypatch):
    """Redirect agents.toml to a temp dir."""
    agents_file = tmp_path / "agents.toml"
    monkeypatch.setattr("towel.cli.agent_mgr.AGENTS_FILE", agents_file)
    monkeypatch.setattr("towel.config.TOWEL_HOME", tmp_path)


class TestCreateAgent:
    def test_create_basic(self):
        profile = create_agent(
            name="tester",
            model_name="mlx-community/test-model",
            identity="You are a tester.",
            description="For testing",
        )
        assert profile.model.name == "mlx-community/test-model"
        assert "tester" in profile.identity

    def test_persists(self):
        create_agent(name="persist", model_name="m", identity="i")
        agents = load_user_agents()
        assert "persist" in agents
        assert agents["persist"]["model"]["name"] == "m"

    def test_custom_params(self):
        profile = create_agent(
            name="hot",
            model_name="m",
            identity="i",
            temperature=0.95,
            context_window=32768,
        )
        assert profile.model.temperature == 0.95
        assert profile.model.context_window == 32768

    def test_overwrite_existing(self):
        create_agent(name="x", model_name="old", identity="old")
        create_agent(name="x", model_name="new", identity="new")
        agents = load_user_agents()
        assert agents["x"]["model"]["name"] == "new"


class TestDeleteAgent:
    def test_delete(self):
        create_agent(name="doomed", model_name="m", identity="i")
        assert delete_agent("doomed")
        assert load_user_agents().get("doomed") is None

    def test_delete_nonexistent(self):
        assert not delete_agent("nope")


class TestCloneAgent:
    def test_clone_builtin(self):
        config = TowelConfig()
        profile = clone_agent("coder", "my-coder", config)
        assert profile is not None
        assert "Qwen" in profile.model.name or "coder" in profile.identity.lower()
        agents = load_user_agents()
        assert "my-coder" in agents

    def test_clone_nonexistent(self):
        config = TowelConfig()
        assert clone_agent("nope", "new", config) is None

    def test_clone_user_agent(self):
        create_agent(name="src", model_name="m", identity="original")
        config = TowelConfig()
        profile = clone_agent("src", "dst", config)
        assert profile is not None


class TestCLICommands:
    def test_agents_list(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agents"])
        assert result.exit_code == 0
        assert "coder" in result.output

    def test_agents_create(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "agents",
                "create",
                "mybot",
                "-m",
                "mlx-community/test",
                "-i",
                "You are mybot.",
                "-d",
                "My custom bot",
            ],
        )
        assert result.exit_code == 0
        assert "Created" in result.output
        assert "mybot" in result.output

    def test_agents_clone(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agents", "clone", "coder", "my-coder"])
        assert result.exit_code == 0
        assert "Cloned" in result.output

    def test_agents_delete(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        # Create then delete
        runner.invoke(cli, ["agents", "create", "temp", "-m", "m", "-i", "i"])
        result = runner.invoke(cli, ["agents", "delete", "temp"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_cannot_delete_builtin(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agents", "delete", "coder"])
        assert "Cannot delete" in result.output

    def test_agents_help(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["agents", "--help"])
        assert "create" in result.output
        assert "clone" in result.output
        assert "delete" in result.output

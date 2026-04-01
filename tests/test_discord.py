"""Tests for Discord channel."""


class TestDiscordChannel:
    def test_instantiation(self):
        from towel.channels.discord import DiscordChannel

        ch = DiscordChannel(token="fake-token", prefix="!ai")
        assert ch.name == "discord"
        assert ch.prefix == "!ai"
        assert ch.token == "fake-token"

    def test_cli_registered(self):
        from towel.cli.main import cli

        assert "discord" in [c.name for c in cli.commands.values()]

    def test_help(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["discord", "--help"])
        assert result.exit_code == 0
        assert "discord" in result.output.lower()
        assert "token" in result.output.lower()

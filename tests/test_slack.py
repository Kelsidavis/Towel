"""Tests for Slack channel."""


class TestSlackChannel:
    def test_instantiation(self):
        from towel.channels.slack import SlackChannel

        ch = SlackChannel(bot_token="xoxb-fake", app_token="xapp-fake")
        assert ch.name == "slack"

    def test_cli_registered(self):
        from towel.cli.main import cli

        assert "slack" in [c.name for c in cli.commands.values()]

    def test_help(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        result = CliRunner().invoke(cli, ["slack", "--help"])
        assert result.exit_code == 0
        assert "slack" in result.output.lower()
        assert "bot-token" in result.output.lower()

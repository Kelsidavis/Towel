"""Tests for Webhook channel."""


class TestWebhookChannel:
    def test_instantiation(self):
        from towel.channels.webhook import WebhookChannel

        ch = WebhookChannel(port=9999, token="secret")
        assert ch.name == "webhook"
        assert ch.port == 9999
        assert ch.token == "secret"

    def test_default_port(self):
        from towel.channels.webhook import WebhookChannel

        ch = WebhookChannel()
        assert ch.port == 18750

    def test_cli_registered(self):
        from towel.cli.main import cli

        assert "webhook" in [c.name for c in cli.commands.values()]

    def test_help(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        result = CliRunner().invoke(cli, ["webhook", "--help"])
        assert result.exit_code == 0
        assert "webhook" in result.output.lower()

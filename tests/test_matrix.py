"""Tests for Matrix channel."""


class TestMatrixChannel:
    def test_instantiation(self):
        from towel.channels.matrix import MatrixChannel

        ch = MatrixChannel(homeserver="https://matrix.org", access_token="fake")
        assert ch.name == "matrix"

    def test_cli_registered(self):
        from towel.cli.main import cli

        assert "matrix" in [c.name for c in cli.commands.values()]

    def test_help(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        result = CliRunner().invoke(cli, ["matrix", "--help"])
        assert result.exit_code == 0
        assert "matrix" in result.output.lower()

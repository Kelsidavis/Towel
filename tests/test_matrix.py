"""Tests for Matrix channel."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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

    @pytest.mark.asyncio
    async def test_listen_returns_on_auth_http_error(self):
        """An HTTP error on /whoami should log and return, not crash."""
        from towel.channels.matrix import MatrixChannel

        ch = MatrixChannel(homeserver="https://matrix.example", access_token="bad")

        fake_resp = MagicMock()
        fake_resp.status_code = 401
        fake_resp.text = '{"errcode":"M_UNKNOWN_TOKEN"}'

        client = AsyncMock()
        client.get = AsyncMock(return_value=fake_resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            await ch.listen()

        assert ch._user_id is None

    @pytest.mark.asyncio
    async def test_listen_returns_on_initial_sync_http_error(self):
        """An HTTP error on initial /sync should log and return."""
        from towel.channels.matrix import MatrixChannel

        ch = MatrixChannel(homeserver="https://matrix.example", access_token="ok")

        whoami_resp = MagicMock()
        whoami_resp.status_code = 200
        whoami_resp.json.return_value = {"user_id": "@bot:example"}

        sync_resp = MagicMock()
        sync_resp.status_code = 502
        sync_resp.text = "Bad Gateway"

        call_count = 0

        async def _fake_get(url, **kw):
            nonlocal call_count
            call_count += 1
            if "whoami" in url:
                return whoami_resp
            return sync_resp

        client = AsyncMock()
        client.get = _fake_get
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            await ch.listen()

        assert ch._user_id == "@bot:example"
        assert ch._next_batch == ""

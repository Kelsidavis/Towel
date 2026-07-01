"""Tests for Discord channel."""

import json

import pytest


class _FakeDiscordWS:
    """Fake Discord gateway socket: one recv() for Hello, then iterates frames."""

    def __init__(self, frames):
        self._hello = frames[0]
        self._rest = frames[1:]

    async def recv(self):
        return self._hello

    async def send(self, data):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._rest:
            raise StopAsyncIteration
        return self._rest.pop(0)


class _FakeConnect:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


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

    @pytest.mark.asyncio
    async def test_malformed_frame_does_not_crash_listen(self, monkeypatch):
        """A malformed gateway frame should be logged and skipped, not
        propagate and kill the whole bot connection."""
        import websockets

        from towel.channels.discord import DiscordChannel

        ch = DiscordChannel(token="fake-token")

        async def _noop_heartbeat(ws):
            return

        monkeypatch.setattr(ch, "_heartbeat", _noop_heartbeat)

        seen_events = []

        async def _fake_dispatch(event, data):
            seen_events.append(event)

        monkeypatch.setattr(ch, "_handle_dispatch", _fake_dispatch)

        hello = json.dumps({"op": 10, "d": {"heartbeat_interval": 41250}})
        bad_frame = "{not valid json"
        good_frame = json.dumps(
            {
                "op": 0,
                "t": "READY",
                "d": {"user": {"id": "1", "username": "b", "discriminator": "0"}},
            }
        )
        ws = _FakeDiscordWS([hello, bad_frame, good_frame])
        monkeypatch.setattr(websockets, "connect", lambda url: _FakeConnect(ws))

        await ch.listen()

        assert seen_events == ["READY"]

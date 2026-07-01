"""Tests for Slack channel."""

import json
from unittest.mock import MagicMock, patch

import pytest


class _FakeSlackWS:
    def __init__(self, frames):
        self._frames = list(frames)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)

    async def send(self, data):
        pass


class _FakeConnect:
    def __init__(self, ws):
        self._ws = ws

    async def __aenter__(self):
        return self._ws

    async def __aexit__(self, *exc):
        return False


def _fake_response(payload, status_code=200):
    fake_resp = MagicMock()
    fake_resp.json.return_value = payload
    fake_resp.status_code = status_code
    return fake_resp


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

    @pytest.mark.asyncio
    async def test_malformed_frame_does_not_crash_listen(self, monkeypatch):
        """A malformed Socket Mode frame should be logged and skipped,
        not propagate and kill the whole bot connection."""
        import websockets

        from towel.channels.slack import SlackChannel

        ch = SlackChannel(bot_token="xoxb-fake", app_token="xapp-fake")

        seen_events = []

        async def _fake_handle_event(event):
            seen_events.append(event)

        monkeypatch.setattr(ch, "_handle_event", _fake_handle_event)

        bad_frame = "{not valid json"
        good_frame = json.dumps(
            {"type": "events_api", "payload": {"event": {"type": "message", "text": "hi"}}}
        )
        disconnect_frame = json.dumps({"type": "disconnect"})
        ws = _FakeSlackWS([bad_frame, good_frame, disconnect_frame])

        responses = [
            _fake_response({"ok": True, "url": "wss://fake"}),
            _fake_response({"ok": True, "user": "bot", "user_id": "U1"}),
        ]

        with patch("httpx.AsyncClient.post", side_effect=responses):
            monkeypatch.setattr(websockets, "connect", lambda url: _FakeConnect(ws))
            await ch.listen()

        assert seen_events == [{"type": "message", "text": "hi"}]

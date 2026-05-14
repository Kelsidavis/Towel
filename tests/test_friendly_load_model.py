"""Tests for the friendly model-load helper used by user-facing CLI commands."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from towel.cli.main import _load_model_with_friendly_error


class TestFriendlyLoadModel:
    def test_returns_true_on_success(self):
        agent = MagicMock()
        agent.load_model = AsyncMock(return_value=None)
        ok = asyncio.run(_load_model_with_friendly_error(agent))
        assert ok is True
        agent.load_model.assert_awaited_once()

    def test_returns_false_when_load_raises(self, capsys):
        agent = MagicMock()
        agent.load_model = AsyncMock(side_effect=RuntimeError("bad model"))
        ok = asyncio.run(_load_model_with_friendly_error(agent))
        assert ok is False
        # The friendly panel should mention the exception text and the
        # three remediation pointers.
        out = capsys.readouterr().out
        assert "bad model" in out
        assert "Failed to load model" in out
        assert "towel setup" in out
        assert "towel doctor" in out

    def test_handles_oserror_too(self, capsys):
        agent = MagicMock()
        agent.load_model = AsyncMock(side_effect=OSError("network unreachable"))
        ok = asyncio.run(_load_model_with_friendly_error(agent))
        assert ok is False
        out = capsys.readouterr().out
        assert "network unreachable" in out

    def test_does_not_swallow_keyboardinterrupt(self):
        # Ctrl-C during model load should propagate so the operator can
        # actually interrupt a slow download.
        agent = MagicMock()
        agent.load_model = AsyncMock(side_effect=KeyboardInterrupt)
        try:
            asyncio.run(_load_model_with_friendly_error(agent))
        except KeyboardInterrupt:
            return
        # The helper catches ``Exception``, not ``BaseException``, so
        # ``KeyboardInterrupt`` (which subclasses BaseException) should
        # propagate. If it didn't, fail explicitly.
        raise AssertionError("KeyboardInterrupt should propagate, but didn't")

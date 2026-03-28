"""Tests for chat slash commands."""

from unittest.mock import MagicMock

import pytest

from towel.agent.conversation import Conversation, Role
from towel.cli.slash import handle_slash, SlashContext
from towel.config import TowelConfig
from towel.memory.store import MemoryStore


@pytest.fixture
def ctx(tmp_path):
    config = TowelConfig()
    conv = Conversation(id="test-session", channel="cli")
    conv.add(Role.USER, "hello")
    conv.add(Role.ASSISTANT, "hi there")
    memory = MemoryStore(store_dir=tmp_path / "memory")
    agent = MagicMock()
    agent.config = config
    store = MagicMock()
    return SlashContext(config=config, conv=conv, agent=agent, memory=memory, store=store)


class TestSlashDispatch:
    def test_non_slash_returns_none(self, ctx):
        assert handle_slash("hello world", ctx) is None

    def test_slash_command_returns_true(self, ctx):
        assert handle_slash("/help", ctx) is True

    def test_unknown_command(self, ctx):
        assert handle_slash("/nonexistent", ctx) is True  # still consumed


class TestHelp:
    def test_help(self, ctx):
        # Should not raise
        handle_slash("/help", ctx)


class TestInfo:
    def test_info(self, ctx):
        handle_slash("/info", ctx)


class TestClear:
    def test_clear_empties_conversation(self, ctx):
        assert len(ctx.conv) == 2
        handle_slash("/clear", ctx)
        assert len(ctx.conv) == 0


class TestAgent:
    def test_show_current(self, ctx):
        handle_slash("/agent", ctx)

    def test_switch_to_builtin(self, ctx):
        handle_slash("/agent coder", ctx)
        assert ctx.current_agent_name == "coder"
        assert "coder" in ctx.config.model.name.lower() or "coder" in ctx.config.identity.lower()

    def test_switch_unknown(self, ctx):
        handle_slash("/agent nonexistent", ctx)
        assert ctx.current_agent_name is None  # unchanged


class TestAgents:
    def test_list(self, ctx):
        handle_slash("/agents", ctx)


class TestMemoryCommands:
    def test_show_empty(self, ctx):
        handle_slash("/memory", ctx)

    def test_remember_and_show(self, ctx):
        handle_slash("/remember name Kelsi", ctx)
        entry = ctx.memory.recall("name")
        assert entry is not None
        assert entry.content == "Kelsi"

    def test_remember_needs_args(self, ctx):
        handle_slash("/remember", ctx)  # should not crash

    def test_forget(self, ctx):
        ctx.memory.remember("temp", "value")
        handle_slash("/forget temp", ctx)
        assert ctx.memory.recall("temp") is None

    def test_forget_nonexistent(self, ctx):
        handle_slash("/forget nope", ctx)  # should not crash


class TestExport:
    def test_export_empty(self, ctx):
        ctx.conv.messages.clear()
        handle_slash("/export", ctx)  # prints "nothing to export"

    def test_export_to_stdout(self, ctx):
        handle_slash("/export", ctx)  # should not crash

    def test_export_to_file(self, ctx, tmp_path):
        f = tmp_path / "out.md"
        handle_slash(f"/export {f}", ctx)
        assert f.exists()
        content = f.read_text()
        assert "hello" in content
        assert "Towel" in content


class TestStats:
    def test_stats_empty(self, ctx):
        ctx.conv.messages.clear()
        handle_slash("/stats", ctx)  # should not crash

    def test_stats_basic(self, ctx):
        handle_slash("/stats", ctx)  # should not crash with 2 messages

    def test_stats_with_metadata(self, ctx):
        ctx.conv.add(Role.ASSISTANT, "Generated text here", metadata={"tps": 42.5, "tokens": 100})
        handle_slash("/stats", ctx)  # should show token stats

    def test_stats_cost_comparison(self, ctx):
        ctx.conv.add(Role.ASSISTANT, "Answer", metadata={"tps": 30.0, "tokens": 5000})
        # Should run without errors and show cost comparison
        handle_slash("/stats", ctx)


class TestExportHtml:
    def test_export_html_file(self, ctx, tmp_path):
        f = tmp_path / "out.html"
        handle_slash(f"/export {f}", ctx)
        assert f.exists()
        content = f.read_text()
        assert "<!DOCTYPE html>" in content
        assert "hello" in content


class TestSystem:
    def test_show_current(self, ctx):
        handle_slash("/system", ctx)

    def test_override(self, ctx):
        handle_slash("/system You are a pirate.", ctx)
        assert ctx.config.identity == "You are a pirate."

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


class TestUndo:
    def test_undo_removes_exchange(self, ctx):
        # ctx has: user("hello"), assistant("hi there")
        assert len(ctx.conv) == 2
        handle_slash("/undo", ctx)
        assert len(ctx.conv) == 0  # both removed

    def test_undo_with_tool_messages(self, ctx):
        ctx.conv.add(Role.USER, "read file")
        ctx.conv.add(Role.TOOL, "[read_file] contents")
        ctx.conv.add(Role.ASSISTANT, "here are the contents")
        assert len(ctx.conv) == 5
        handle_slash("/undo", ctx)
        # Should remove: assistant + tool + user = 3, leaving original 2
        assert len(ctx.conv) == 2

    def test_undo_empty(self, ctx):
        ctx.conv.messages.clear()
        handle_slash("/undo", ctx)  # should not crash


class TestRetry:
    def test_retry_returns_false(self, ctx):
        # ctx has: user("hello"), assistant("hi there")
        result = handle_slash("/retry", ctx)
        assert result is False  # signals "run agent step"
        # Assistant message removed, user message kept
        assert len(ctx.conv) == 1
        assert ctx.conv.messages[0].content == "hello"

    def test_retry_empty(self, ctx):
        ctx.conv.messages.clear()
        result = handle_slash("/retry", ctx)
        assert result is True  # consumed, nothing to do

    def test_retry_no_assistant(self, ctx):
        ctx.conv.messages.clear()
        ctx.conv.add(Role.USER, "hello")
        result = handle_slash("/retry", ctx)
        assert result is True  # no assistant to remove


class TestCompact:
    def test_compact_reduces_messages(self, ctx):
        # Add more messages to have something to compact
        for i in range(10):
            ctx.conv.add(Role.USER, f"Question {i}")
            ctx.conv.add(Role.ASSISTANT, f"Answer {i} with some detail and explanation")
        total_before = len(ctx.conv)
        handle_slash("/compact 4", ctx)
        # Should have: 1 summary + 4 recent
        assert len(ctx.conv) < total_before
        assert len(ctx.conv) == 5  # summary + 4 kept

    def test_compact_preserves_recent(self, ctx):
        ctx.conv.add(Role.USER, "important question")
        ctx.conv.add(Role.ASSISTANT, "important answer")
        handle_slash("/compact 4", ctx)
        # Last messages should be unchanged
        contents = [m.content for m in ctx.conv.messages]
        assert "important answer" in contents

    def test_compact_preserves_pinned(self, ctx):
        ctx.conv.messages[1].pinned = True  # pin "hi there"
        for i in range(8):
            ctx.conv.add(Role.USER, f"Q{i}")
            ctx.conv.add(Role.ASSISTANT, f"A{i}")
        handle_slash("/compact 4", ctx)
        pinned = [m for m in ctx.conv.messages if m.pinned]
        assert len(pinned) == 1
        assert pinned[0].content == "hi there"

    def test_compact_too_few_messages(self, ctx):
        # Only 2 messages, compact 4 should do nothing
        handle_slash("/compact 4", ctx)
        assert len(ctx.conv) == 2

    def test_compact_summary_is_system(self, ctx):
        for i in range(10):
            ctx.conv.add(Role.USER, f"Q{i}")
            ctx.conv.add(Role.ASSISTANT, f"A{i}")
        handle_slash("/compact 2", ctx)
        assert ctx.conv.messages[0].role == Role.SYSTEM
        assert "Compacted" in ctx.conv.messages[0].content


class TestDiff:
    def test_diff_identical(self, ctx, tmp_path):
        from towel.persistence.store import ConversationStore
        from towel.agent.conversation import Conversation
        store = ConversationStore(store_dir=tmp_path / "convs")
        ctx.store = store

        # Save a copy as "other"
        other = Conversation(id="other-1", channel="cli")
        other.add(Role.USER, "hello")
        other.add(Role.ASSISTANT, "hi there")
        store.save(other)

        handle_slash("/diff other-1", ctx)  # should say identical

    def test_diff_divergent(self, ctx, tmp_path):
        from towel.persistence.store import ConversationStore
        from towel.agent.conversation import Conversation
        store = ConversationStore(store_dir=tmp_path / "convs")
        ctx.store = store

        # Same start, different continuation
        other = Conversation(id="other-2", channel="cli")
        other.add(Role.USER, "hello")
        other.add(Role.ASSISTANT, "hi there")
        other.add(Role.USER, "different question")
        other.add(Role.ASSISTANT, "different answer")
        store.save(other)

        ctx.conv.add(Role.USER, "my question")
        ctx.conv.add(Role.ASSISTANT, "my answer")

        handle_slash("/diff other-2", ctx)  # should show divergence

    def test_diff_not_found(self, ctx):
        handle_slash("/diff nonexistent", ctx)  # should print not found

    def test_diff_no_arg(self, ctx):
        handle_slash("/diff", ctx)  # should print usage


class TestGrep:
    def test_grep_finds_match(self, ctx):
        ctx.conv.add(Role.ASSISTANT, "The answer is 42, obviously.")
        result = handle_slash("/grep 42", ctx)
        assert result is True  # consumed

    def test_grep_no_match(self, ctx):
        handle_slash("/grep xyznonexistent", ctx)  # should print "no matches"

    def test_grep_empty_query(self, ctx):
        handle_slash("/grep", ctx)  # should print usage

    def test_grep_empty_conversation(self, ctx):
        ctx.conv.messages.clear()
        handle_slash("/grep hello", ctx)  # should print "no messages"

    def test_grep_case_insensitive(self, ctx):
        ctx.conv.add(Role.ASSISTANT, "Python is great")
        # Should match regardless of case
        handle_slash("/grep PYTHON", ctx)  # should not crash, finds match


class TestPin:
    def test_pin_last_assistant(self, ctx):
        handle_slash("/pin", ctx)
        # Last assistant message should be pinned
        asst_msgs = [m for m in ctx.conv.messages if m.role == Role.ASSISTANT]
        assert asst_msgs[-1].pinned is True

    def test_pin_toggle(self, ctx):
        handle_slash("/pin", ctx)
        handle_slash("/pin", ctx)
        asst_msgs = [m for m in ctx.conv.messages if m.role == Role.ASSISTANT]
        assert asst_msgs[-1].pinned is False  # toggled off

    def test_pin_by_id(self, ctx):
        msg_id = ctx.conv.messages[0].id  # user message
        handle_slash(f"/pin {msg_id}", ctx)
        assert ctx.conv.messages[0].pinned is True

    def test_pin_unknown_id(self, ctx):
        handle_slash("/pin nonexistent123", ctx)  # should not crash

    def test_pins_empty(self, ctx):
        handle_slash("/pins", ctx)  # should not crash

    def test_pins_shows_pinned(self, ctx):
        ctx.conv.messages[1].pinned = True
        handle_slash("/pins", ctx)  # should show the pinned message

    def test_pinned_serialization(self):
        from towel.agent.conversation import Message, Role
        msg = Message(role=Role.ASSISTANT, content="important", pinned=True)
        d = msg.to_dict()
        assert d["pinned"] is True
        restored = Message.from_dict(d)
        assert restored.pinned is True

    def test_unpinned_not_in_dict(self):
        from towel.agent.conversation import Message, Role
        msg = Message(role=Role.ASSISTANT, content="normal")
        d = msg.to_dict()
        assert "pinned" not in d  # only serialized when True


class TestSave:
    def test_save_code_block(self, ctx, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ctx.conv.add(Role.ASSISTANT, "Here:\n```python\nprint('hello')\n```")
        handle_slash(f"/save {tmp_path / 'out.py'}", ctx)
        result = (tmp_path / "out.py").read_text()
        assert "print('hello')" in result

    def test_save_second_block(self, ctx, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ctx.conv.add(Role.ASSISTANT, "A:\n```py\nfirst\n```\nB:\n```py\nsecond\n```")
        handle_slash(f"/save {tmp_path / 'out.py'} 2", ctx)
        result = (tmp_path / "out.py").read_text()
        assert "second" in result
        assert "first" not in result

    def test_save_no_blocks(self, ctx):
        ctx.conv.add(Role.ASSISTANT, "Just text, no code.")
        handle_slash("/save out.py", ctx)  # should print "no code blocks"

    def test_save_no_args(self, ctx):
        handle_slash("/save", ctx)  # should print usage

    def test_save_invalid_block_index(self, ctx, tmp_path):
        ctx.conv.add(Role.ASSISTANT, "One:\n```py\nonly\n```")
        handle_slash(f"/save {tmp_path / 'out.py'} 5", ctx)  # block 5 doesn't exist
        assert not (tmp_path / "out.py").exists()


class TestCopy:
    def test_copy_no_response(self, ctx):
        ctx.conv.messages.clear()
        ctx.conv.add(Role.USER, "hello")
        handle_slash("/copy", ctx)  # should not crash, no assistant msg

    def test_copy_extracts_code_blocks(self, ctx):
        """Test code extraction logic without actually touching clipboard."""
        import re
        ctx.conv.add(Role.ASSISTANT, "Here:\n```python\nprint('hi')\n```\nand\n```js\nalert(1)\n```")
        last = ctx.conv.messages[-1]
        blocks = re.findall(r"```\w*\n(.*?)```", last.content, re.DOTALL)
        assert len(blocks) == 2
        assert "print('hi')" in blocks[0]
        assert "alert(1)" in blocks[1]

    def test_copy_no_code_blocks(self, ctx):
        ctx.conv.add(Role.ASSISTANT, "Just plain text, no code here.")
        # /copy code should print "no code blocks" message
        handle_slash("/copy code", ctx)  # should not crash


class TestTags:
    def test_add_tag(self, ctx):
        handle_slash("/tag project-alpha", ctx)
        assert "project-alpha" in ctx.conv.tags

    def test_add_duplicate_tag(self, ctx):
        handle_slash("/tag work", ctx)
        handle_slash("/tag work", ctx)
        assert ctx.conv.tags.count("work") == 1

    def test_remove_tag(self, ctx):
        ctx.conv.tags.append("old")
        handle_slash("/tag -old", ctx)
        assert "old" not in ctx.conv.tags

    def test_remove_nonexistent(self, ctx):
        handle_slash("/tag -nope", ctx)  # should not crash

    def test_show_tags(self, ctx):
        ctx.conv.tags = ["a", "b"]
        handle_slash("/tags", ctx)  # should not crash

    def test_show_no_tags(self, ctx):
        handle_slash("/tags", ctx)  # should not crash

    def test_tags_lowercase(self, ctx):
        handle_slash("/tag UPPERCASE", ctx)
        assert "uppercase" in ctx.conv.tags


class TestFork:
    def test_fork_creates_branch(self, ctx):
        old_id = ctx.conv.id
        handle_slash("/fork", ctx)
        # ID should change
        assert ctx.conv.id != old_id
        # Messages preserved
        assert len(ctx.conv) == 2
        # Original was saved
        ctx.store.save.assert_called()

    def test_fork_with_custom_title(self, ctx):
        handle_slash("/fork My exploration branch", ctx)
        assert ctx.conv.title == "My exploration branch"

    def test_fork_empty(self, ctx):
        ctx.conv.messages.clear()
        old_id = ctx.conv.id
        handle_slash("/fork", ctx)
        # Should not fork, ID unchanged
        assert ctx.conv.id == old_id

    def test_fork_preserves_message_content(self, ctx):
        handle_slash("/fork", ctx)
        assert ctx.conv.messages[0].content == "hello"
        assert ctx.conv.messages[1].content == "hi there"


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


class TestSnippets:
    def test_create_snippet(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        handle_slash("/snippet header # My Project", ctx)
        from towel.cli.snippets import get_snippet
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        assert get_snippet("header") == "# My Project"

    def test_snippet_newline_escape(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        handle_slash("/snippet sig Best regards\\nKelsi", ctx)
        from towel.cli.snippets import get_snippet
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        assert get_snippet("sig") == "Best regards\nKelsi"

    def test_remove_snippet(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        from towel.cli.snippets import set_snippet
        set_snippet("temp", "temporary")
        handle_slash("/snippet -temp", ctx)
        from towel.cli.snippets import get_snippet
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        assert get_snippet("temp") is None

    def test_list_snippets(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        handle_slash("/snippet test Hello world", ctx)
        handle_slash("/snippets", ctx)  # should not crash

    def test_use_snippet(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        from towel.cli.snippets import set_snippet
        set_snippet("greet", "Hello, please help with")
        result = handle_slash("/s greet my code", ctx)
        assert result is False  # signal agent step
        last_user = [m for m in ctx.conv.messages if m.role == Role.USER][-1]
        assert "Hello, please help with" in last_user.content
        assert "my code" in last_user.content

    def test_use_snippet_not_found(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.snippets.SNIPPETS_FILE", tmp_path / "snippets.json")
        result = handle_slash("/s nonexistent", ctx)
        assert result is True  # consumed as error


class TestAliases:
    def test_create_alias(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        handle_slash("/alias review Review this code carefully", ctx)
        from towel.cli.aliases import get_alias
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        assert get_alias("review") == "Review this code carefully"

    def test_list_aliases(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        handle_slash("/alias test A test alias", ctx)
        handle_slash("/aliases", ctx)  # should not crash

    def test_remove_alias(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        handle_slash("/alias temp Temporary alias", ctx)
        handle_slash("/unalias temp", ctx)
        from towel.cli.aliases import get_alias
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        assert get_alias("temp") is None

    def test_alias_expansion(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        from towel.cli.aliases import set_alias
        set_alias("greet", "Say hello to")
        result = handle_slash("/greet the world", ctx)
        assert result is False  # should signal agent step
        # Last user message should contain the expanded alias
        last_user = [m for m in ctx.conv.messages if m.role == Role.USER][-1]
        assert "Say hello to" in last_user.content
        assert "the world" in last_user.content

    def test_alias_no_args(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        from towel.cli.aliases import set_alias
        set_alias("status", "What is the current project status?")
        result = handle_slash("/status", ctx)
        assert result is False
        last_user = [m for m in ctx.conv.messages if m.role == Role.USER][-1]
        assert "current project status" in last_user.content

    def test_unknown_still_errors(self, ctx, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.aliases.ALIASES_FILE", tmp_path / "aliases.json")
        result = handle_slash("/totallyunknown", ctx)
        assert result is True  # consumed as unknown command


class TestSystem:
    def test_show_current(self, ctx):
        handle_slash("/system", ctx)

    def test_override(self, ctx):
        handle_slash("/system You are a pirate.", ctx)
        assert ctx.config.identity == "You are a pirate."

"""Tests for conversation export."""

from datetime import datetime, timezone

import pytest

from towel.agent.conversation import Conversation, Message, Role
from towel.persistence.export import export_markdown, export_text, export_json, export_html


@pytest.fixture
def conv():
    c = Conversation(
        id="test-export-123",
        channel="cli",
        created_at=datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc),
    )
    c.add(Role.USER, "How do I make pancakes?")
    c.add(Role.ASSISTANT, "Here's a simple pancake recipe:\n\n1. Mix flour, eggs, and milk\n2. Heat a pan\n3. Pour batter and flip")
    c.add(Role.USER, "What about toppings?")
    c.add(Role.ASSISTANT, "Try maple syrup, blueberries, or whipped cream.")
    return c


@pytest.fixture
def conv_with_tools():
    c = Conversation(id="tools-456", channel="webchat")
    c.add(Role.USER, "Read my config file")
    c.add(Role.TOOL, "[read_file] host = 127.0.0.1\nport = 8080")
    c.add(Role.ASSISTANT, "Your config shows host 127.0.0.1 on port 8080.")
    return c


class TestMarkdownExport:
    def test_header(self, conv):
        md = export_markdown(conv)
        assert "# How do I make pancakes?" in md
        assert "`test-export-123`" in md
        assert "cli" in md
        assert "2026-03-15" in md

    def test_user_messages(self, conv):
        md = export_markdown(conv)
        assert "### You" in md
        assert "How do I make pancakes?" in md
        assert "What about toppings?" in md

    def test_assistant_messages(self, conv):
        md = export_markdown(conv)
        assert "### Towel" in md
        assert "maple syrup" in md

    def test_tool_messages_in_details(self, conv_with_tools):
        md = export_markdown(conv_with_tools)
        assert "<details>" in md
        assert "read_file" in md
        assert "host = 127.0.0.1" in md
        assert "</details>" in md

    def test_footer(self, conv):
        md = export_markdown(conv)
        assert "Don't Panic" in md

    def test_metadata_flag(self, conv):
        without = export_markdown(conv, include_metadata=False)
        with_meta = export_markdown(conv, include_metadata=True)
        # Metadata version should have timestamps
        assert ":" in with_meta and len(with_meta) > len(without)

    def test_empty_conversation(self):
        c = Conversation(id="empty")
        md = export_markdown(c)
        assert "(empty)" in md


class TestTextExport:
    def test_header(self, conv):
        txt = export_text(conv)
        assert "test-export-123" in txt
        assert "cli" in txt

    def test_user_messages(self, conv):
        txt = export_text(conv)
        assert "[you]" in txt
        assert "pancakes" in txt

    def test_assistant_messages(self, conv):
        txt = export_text(conv)
        assert "[towel]" in txt

    def test_tool_truncated(self, conv_with_tools):
        txt = export_text(conv_with_tools)
        assert "[tool]" in txt

    def test_separator(self, conv):
        txt = export_text(conv)
        assert "=" * 60 in txt


class TestJsonExport:
    def test_valid_json(self, conv):
        import json
        result = export_json(conv)
        data = json.loads(result)
        assert data["id"] == "test-export-123"

    def test_messages_preserved(self, conv):
        import json
        data = json.loads(export_json(conv))
        assert len(data["messages"]) == 4

    def test_roundtrip(self, conv):
        import json
        exported = export_json(conv)
        restored = Conversation.from_dict(json.loads(exported))
        assert restored.id == conv.id
        assert len(restored) == len(conv)

    def test_compact_mode(self, conv):
        compact = export_json(conv, pretty=False)
        pretty = export_json(conv, pretty=True)
        assert len(compact) < len(pretty)


class TestHtmlExport:
    def test_valid_html(self, conv):
        result = export_html(conv)
        assert "<!DOCTYPE html>" in result
        assert "</html>" in result
        assert "<style>" in result

    def test_title(self, conv):
        result = export_html(conv)
        assert "<title>" in result
        assert "pancakes" in result.lower()

    def test_user_messages(self, conv):
        result = export_html(conv)
        assert "You" in result
        assert "pancakes" in result

    def test_assistant_messages(self, conv):
        result = export_html(conv)
        assert "Towel" in result
        assert "maple syrup" in result

    def test_tool_messages(self, conv_with_tools):
        result = export_html(conv_with_tools)
        assert "read_file" in result
        assert "127.0.0.1" in result

    def test_code_blocks(self):
        c = Conversation(id="code-test")
        c.add(Role.ASSISTANT, "Here is code:\n```python\nprint('hello')\n```")
        result = export_html(c)
        assert "<pre><code>" in result
        assert "print" in result

    def test_html_escaping(self):
        c = Conversation(id="escape-test")
        c.add(Role.USER, "What does <script>alert('xss')</script> do?")
        result = export_html(c)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_footer(self, conv):
        result = export_html(conv)
        assert "Don't Panic" in result

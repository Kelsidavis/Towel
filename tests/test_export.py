"""Tests for conversation export."""

from datetime import UTC, datetime

import pytest

from towel.agent.conversation import Conversation, Role
from towel.persistence.export import export_html, export_json, export_markdown, export_text


@pytest.fixture
def conv():
    c = Conversation(
        id="test-export-123",
        channel="cli",
        created_at=datetime(2026, 3, 15, 10, 30, 0, tzinfo=UTC),
    )
    c.add(Role.USER, "How do I make pancakes?")
    c.add(
        Role.ASSISTANT,
        "Here's a simple pancake recipe:\n\n"
        "1. Mix flour, eggs, and milk\n"
        "2. Heat a pan\n3. Pour batter and flip",
    )
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

    def test_header_prefers_explicit_title(self):
        """An operator who renamed the conversation via
        /conversations/<id>/rename should see that title in the
        exported file instead of the first user message — previously
        the export always used the auto-derived summary."""
        c = Conversation(id="titled-export", channel="cli")
        c.title = "Recipe Notes"
        c.add(Role.USER, "How do I make pancakes?")
        md = export_markdown(c)
        # Explicit title wins.
        assert "# Recipe Notes" in md
        # The auto-summary is no longer the header.
        assert "# How do I make pancakes?" not in md

    def test_html_header_prefers_explicit_title(self):
        """Parity with markdown — HTML title + h1 also honor the
        operator-set title."""
        c = Conversation(id="titled-html", channel="cli")
        c.title = "Recipe Notes"
        c.add(Role.USER, "How do I make pancakes?")
        from towel.persistence.export import export_html
        html_out = export_html(c)
        assert "<title>Recipe Notes</title>" in html_out
        assert "<h1>Recipe Notes</h1>" in html_out

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

    def test_ensemble_metadata_surfaces(self):
        """An exported transcript should make it obvious when
        multi-worker collaboration ran on a turn. Without this, the
        markdown looked identical for a single-worker answer and an
        ensemble-arbitrated answer."""
        c = Conversation(id="ens-export", channel="api")
        c.add(Role.USER, "What's 2+2?")
        c.add(
            Role.ASSISTANT,
            "4",
            ensemble=True,
            ensemble_arbitration="consensus",
            ensemble_contributions=[
                {"worker_id": "a"}, {"worker_id": "b"}, {"worker_id": "c"},
            ],
        )
        md = export_markdown(c, include_metadata=True)
        assert "ensemble:consensus" in md
        assert "3 workers" in md

    def test_verify_metadata_surfaces(self):
        c = Conversation(id="ver-export", channel="api")
        c.add(Role.USER, "Capital of Germany?")
        c.add(
            Role.ASSISTANT,
            "Berlin",
            verified_by="big-worker",
            verifier_corrected=True,
        )
        md = export_markdown(c, include_metadata=True)
        assert "verified-by:big-worker" in md
        assert "(corrected)" in md

    def test_html_ensemble_metadata_surfaces(self):
        """Parity with markdown export: HTML should also expose
        collaboration metadata so a saved HTML transcript shows when
        multi-worker arbitration ran."""
        c = Conversation(id="html-ens", channel="api")
        c.add(Role.USER, "Q?")
        c.add(
            Role.ASSISTANT,
            "Answer.",
            ensemble=True,
            ensemble_arbitration="synthesis",
            ensemble_contributions=[
                {"worker_id": "a"}, {"worker_id": "b"},
            ],
        )
        html = export_html(c, include_metadata=True)
        assert "ensemble:synthesis" in html
        assert "2 workers" in html

    def test_html_verify_metadata_surfaces(self):
        c = Conversation(id="html-ver", channel="api")
        c.add(Role.USER, "Q?")
        c.add(
            Role.ASSISTANT,
            "A",
            verified_by="vrf",
            verifier_corrected=False,
        )
        html = export_html(c, include_metadata=True)
        assert "verified-by:vrf" in html
        # No (corrected) tag when verifier_corrected is False.
        assert "(corrected)" not in html


class TestTextExportTitle:
    def test_text_header_includes_title_and_session_id(self):
        """The plain-text export header now leads with the operator-
        set title (or auto-summary when no title is set) so a
        reader skimming saved txt files sees the name first. The
        session id still appears for cross-referencing."""
        from towel.persistence.export import export_text

        c = Conversation(id="text-titled", channel="cli")
        c.title = "My Deploy Notes"
        c.add(Role.USER, "first message")
        out = export_text(c)
        assert "Conversation: My Deploy Notes" in out
        assert "Session: text-titled" in out

    def test_text_header_falls_back_to_summary_without_title(self):
        """No explicit title → use auto-derived summary, matching
        the markdown/html exports."""
        from towel.persistence.export import export_text

        c = Conversation(id="text-untitled", channel="cli")
        c.add(Role.USER, "what is towel")
        out = export_text(c)
        # display_title falls back to summary (first user message),
        # so the heading reflects the actual content.
        assert "Conversation: what is towel" in out


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

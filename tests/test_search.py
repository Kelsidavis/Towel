"""Tests for conversation search."""

import pytest

from towel.agent.conversation import Conversation, Role
from towel.persistence.store import ConversationStore, _extract_snippet

import re


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


def _save_conv(store, conv_id, messages):
    """Helper to create and save a conversation with given messages."""
    conv = Conversation(id=conv_id, channel="cli")
    for role, content in messages:
        conv.add(role, content)
    store.save(conv)
    return conv


class TestSearch:
    def test_basic_search(self, store):
        _save_conv(store, "c1", [
            (Role.USER, "how do I make pancakes"),
            (Role.ASSISTANT, "here is a pancake recipe"),
        ])
        _save_conv(store, "c2", [
            (Role.USER, "tell me about waffles"),
            (Role.ASSISTANT, "waffles are great"),
        ])

        results = store.search("pancake")
        assert len(results) == 1
        assert results[0].conversation_id == "c1"
        assert len(results[0].matches) >= 1

    def test_case_insensitive(self, store):
        _save_conv(store, "c1", [(Role.USER, "Hello WORLD")])
        results = store.search("hello world")
        assert len(results) == 1

    def test_no_results(self, store):
        _save_conv(store, "c1", [(Role.USER, "hello")])
        results = store.search("xyzzy")
        assert len(results) == 0

    def test_multiple_matches_in_one_conversation(self, store):
        _save_conv(store, "c1", [
            (Role.USER, "tell me about Python"),
            (Role.ASSISTANT, "Python is a programming language"),
            (Role.USER, "what about Python 3.12"),
        ])
        results = store.search("Python")
        assert len(results) == 1
        assert len(results[0].matches) == 3

    def test_search_across_conversations(self, store):
        _save_conv(store, "c1", [(Role.USER, "deploy the API")])
        _save_conv(store, "c2", [(Role.USER, "fix the API bug")])
        _save_conv(store, "c3", [(Role.USER, "unrelated stuff")])

        results = store.search("API")
        assert len(results) == 2
        ids = {r.conversation_id for r in results}
        assert ids == {"c1", "c2"}

    def test_role_filter(self, store):
        _save_conv(store, "c1", [
            (Role.USER, "tell me about cats"),
            (Role.ASSISTANT, "cats are wonderful animals"),
        ])
        # Search only user messages
        results = store.search("cats", role_filter=Role.USER)
        assert len(results) == 1
        assert all(m.role == "user" for m in results[0].matches)

        # Search only assistant messages
        results = store.search("cats", role_filter=Role.ASSISTANT)
        assert len(results) == 1
        assert all(m.role == "assistant" for m in results[0].matches)

    def test_regex_search(self, store):
        _save_conv(store, "c1", [(Role.USER, "error code 404 happened")])
        _save_conv(store, "c2", [(Role.USER, "error code 500 happened")])
        _save_conv(store, "c3", [(Role.USER, "everything is fine")])

        results = store.search(r"error code \d+", regex=True)
        assert len(results) == 2

    def test_invalid_regex(self, store):
        results = store.search("[invalid regex", regex=True)
        assert len(results) == 0

    def test_limit(self, store):
        for i in range(10):
            _save_conv(store, f"c{i}", [(Role.USER, "common search term")])

        results = store.search("common", limit=3)
        assert len(results) == 3

    def test_sorted_by_match_count(self, store):
        _save_conv(store, "few", [
            (Role.USER, "one mention of towel"),
        ])
        _save_conv(store, "many", [
            (Role.USER, "towel towel towel"),
            (Role.ASSISTANT, "here is your towel"),
            (Role.USER, "another towel please"),
        ])

        results = store.search("towel")
        assert len(results) == 2
        # "many" should come first (more matches)
        assert results[0].conversation_id == "many"

    def test_search_result_has_snippets(self, store):
        _save_conv(store, "c1", [
            (Role.USER, "I need help with the deployment pipeline that keeps failing"),
        ])
        results = store.search("deployment")
        assert len(results) == 1
        assert "deployment" in results[0].matches[0].snippet.lower()

    def test_corrupt_file_skipped(self, store, tmp_path):
        _save_conv(store, "good", [(Role.USER, "findme")])
        (tmp_path / "bad.json").write_text("{invalid json")

        results = store.search("findme")
        assert len(results) == 1


class TestExtractSnippet:
    def test_short_text(self):
        pattern = re.compile("hello", re.IGNORECASE)
        snippet = _extract_snippet("hello world", pattern)
        assert "hello" in snippet

    def test_long_text_with_context(self):
        text = "x" * 200 + " KEY_WORD " + "y" * 200
        pattern = re.compile("KEY_WORD", re.IGNORECASE)
        snippet = _extract_snippet(text, pattern, context_chars=40)
        assert "KEY_WORD" in snippet
        assert snippet.startswith("...")
        assert snippet.endswith("...")
        assert len(snippet) < len(text)

    def test_no_match_returns_start(self):
        pattern = re.compile("missing", re.IGNORECASE)
        text = "some text here"
        snippet = _extract_snippet(text, pattern)
        assert snippet == "some text here"

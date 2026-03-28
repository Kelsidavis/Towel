"""Tests for conversation persistence."""

import json
from datetime import datetime, timezone

import pytest

from towel.agent.conversation import Conversation, Message, Role
from towel.persistence.store import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


def _make_conversation(conv_id: str = "test-123", messages: int = 3) -> Conversation:
    conv = Conversation(id=conv_id, channel="cli")
    for i in range(messages):
        if i % 2 == 0:
            conv.add(Role.USER, f"Message {i}")
        else:
            conv.add(Role.ASSISTANT, f"Response {i}")
    return conv


class TestConversationSerialization:
    def test_message_roundtrip(self):
        msg = Message(role=Role.USER, content="hello", metadata={"key": "val"})
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.id == msg.id
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.metadata == msg.metadata

    def test_conversation_roundtrip(self):
        conv = _make_conversation()
        d = conv.to_dict()
        restored = Conversation.from_dict(d)
        assert restored.id == conv.id
        assert restored.channel == conv.channel
        assert len(restored) == len(conv)
        assert restored.messages[0].content == conv.messages[0].content

    def test_summary_property(self):
        conv = Conversation()
        assert conv.summary == "(empty)"
        conv.add(Role.USER, "What is the meaning of life?")
        assert conv.summary == "What is the meaning of life?"

    def test_summary_truncation(self):
        conv = Conversation()
        conv.add(Role.USER, "x" * 200)
        assert len(conv.summary) == 83  # 80 + "..."
        assert conv.summary.endswith("...")


class TestConversationStore:
    def test_save_and_load(self, store):
        conv = _make_conversation()
        store.save(conv)
        loaded = store.load(conv.id)
        assert loaded is not None
        assert loaded.id == conv.id
        assert len(loaded) == len(conv)

    def test_load_nonexistent(self, store):
        assert store.load("nope") is None

    def test_exists(self, store):
        conv = _make_conversation()
        assert not store.exists(conv.id)
        store.save(conv)
        assert store.exists(conv.id)

    def test_delete(self, store):
        conv = _make_conversation()
        store.save(conv)
        assert store.delete(conv.id)
        assert not store.exists(conv.id)
        assert not store.delete(conv.id)  # already gone

    def test_list_conversations(self, store):
        for i in range(5):
            store.save(_make_conversation(f"conv-{i}"))
        summaries = store.list_conversations()
        assert len(summaries) == 5
        # Check summary fields
        s = summaries[0]
        assert s.id.startswith("conv-")
        assert s.message_count == 3
        assert s.channel == "cli"

    def test_list_respects_limit(self, store):
        for i in range(10):
            store.save(_make_conversation(f"conv-{i}"))
        assert len(store.list_conversations(limit=3)) == 3

    def test_len(self, store):
        assert store.count == 0
        store.save(_make_conversation("a"))
        store.save(_make_conversation("b"))
        assert store.count == 2

    def test_overwrite_existing(self, store):
        conv = _make_conversation()
        store.save(conv)
        conv.add(Role.USER, "one more")
        store.save(conv)
        loaded = store.load(conv.id)
        assert loaded is not None
        assert len(loaded) == 4  # original 3 + 1

    def test_path_traversal_sanitized(self, store):
        conv = _make_conversation("../../etc/passwd")
        path = store.save(conv)
        assert ".." not in str(path)
        assert path.parent == store.store_dir

    def test_corrupt_file_skipped(self, store, tmp_path):
        (tmp_path / "bad.json").write_text("not valid json {{{")
        summaries = store.list_conversations()
        assert len(summaries) == 0


class TestSessionManagerPersistence:
    def test_save_and_resume(self, store):
        from towel.gateway.sessions import SessionManager

        sm = SessionManager(store=store)
        session = sm.get_or_create("persistent-1")
        session.conversation.add(Role.USER, "hello")
        session.conversation.add(Role.ASSISTANT, "hi there")
        sm.save("persistent-1")

        # New session manager (simulating restart)
        sm2 = SessionManager(store=store)
        session2 = sm2.get_or_create("persistent-1")
        assert len(session2.conversation) == 2
        assert session2.conversation.messages[0].content == "hello"

    def test_save_all(self, store):
        from towel.gateway.sessions import SessionManager

        sm = SessionManager(store=store)
        sm.get_or_create("a").conversation.add(Role.USER, "msg a")
        sm.get_or_create("b").conversation.add(Role.USER, "msg b")
        saved = sm.save_all()
        assert saved == 2
        assert store.count == 2

"""Tests for incremental context synchronization."""

from towel.agent.conversation import Conversation, Message, Role
from towel.gateway.context_sync import (
    ContextSyncManager,
    ConversationDelta,
    apply_delta,
    compute_response_delta,
)


def _conv_with_messages(count: int) -> Conversation:
    conv = Conversation(id="test-conv")
    for i in range(count):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        conv.add(role, f"Message {i}")
    return conv


class TestContextSyncManager:
    def test_first_sync_is_full(self):
        mgr = ContextSyncManager()
        conv = _conv_with_messages(5)

        delta = mgr.compute_delta("w1", "s1", conv)

        assert delta.is_full_sync is True
        assert len(delta.new_messages) == 5

    def test_incremental_after_cursor_advance(self):
        mgr = ContextSyncManager()
        conv = _conv_with_messages(3)

        # First sync: full
        delta = mgr.compute_delta("w1", "s1", conv)
        assert delta.is_full_sync is True

        # Advance cursor
        mgr.advance_cursor("w1", "s1", conv)

        # Add more messages
        conv.add(Role.USER, "New question")
        conv.add(Role.ASSISTANT, "New answer")

        # Second sync: incremental
        delta = mgr.compute_delta("w1", "s1", conv)
        assert delta.is_full_sync is False
        assert len(delta.new_messages) == 2
        assert delta.new_messages[0]["content"] == "New question"

    def test_falls_back_to_full_sync_after_compaction(self):
        mgr = ContextSyncManager()
        conv = _conv_with_messages(5)

        mgr.advance_cursor("w1", "s1", conv)

        # Simulate compaction: remove old messages
        conv.messages = conv.messages[3:]

        delta = mgr.compute_delta("w1", "s1", conv)
        assert delta.is_full_sync is True

    def test_empty_delta_when_no_changes(self):
        mgr = ContextSyncManager()
        conv = _conv_with_messages(3)

        mgr.advance_cursor("w1", "s1", conv)

        delta = mgr.compute_delta("w1", "s1", conv)
        assert delta.is_full_sync is False
        assert delta.is_empty is True

    def test_clear_worker(self):
        mgr = ContextSyncManager()
        conv = _conv_with_messages(3)

        mgr.advance_cursor("w1", "s1", conv)
        mgr.advance_cursor("w1", "s2", conv)
        mgr.advance_cursor("w2", "s1", conv)

        cleared = mgr.clear_worker("w1")
        assert cleared == 2
        assert mgr.get_cursor("w1", "s1") is None
        assert mgr.get_cursor("w2", "s1") is not None

    def test_clear_session(self):
        mgr = ContextSyncManager()
        conv = _conv_with_messages(3)

        mgr.advance_cursor("w1", "s1", conv)
        mgr.advance_cursor("w2", "s1", conv)
        mgr.advance_cursor("w1", "s2", conv)

        cleared = mgr.clear_session("s1")
        assert cleared == 2
        assert mgr.get_cursor("w1", "s2") is not None

    def test_different_workers_track_independently(self):
        mgr = ContextSyncManager()
        conv = _conv_with_messages(3)

        # Worker 1 sees full conversation
        mgr.advance_cursor("w1", "s1", conv)

        # Worker 2 has never seen it
        delta_w2 = mgr.compute_delta("w2", "s1", conv)
        assert delta_w2.is_full_sync is True

        # Worker 1 sees incremental
        conv.add(Role.USER, "New message")
        delta_w1 = mgr.compute_delta("w1", "s1", conv)
        assert delta_w1.is_full_sync is False
        assert len(delta_w1.new_messages) == 1


class TestApplyDelta:
    def test_full_sync_replaces_all(self):
        conv = _conv_with_messages(3)
        new_messages = [
            Message(role=Role.USER, content="Fresh start").to_dict(),
            Message(role=Role.ASSISTANT, content="Hello").to_dict(),
        ]
        delta = ConversationDelta(
            session_id="s1",
            new_messages=new_messages,
            is_full_sync=True,
            conversation_metadata={"title": "New title"},
        )

        apply_delta(conv, delta)

        assert len(conv.messages) == 2
        assert conv.messages[0].content == "Fresh start"
        assert conv.title == "New title"

    def test_incremental_appends(self):
        conv = _conv_with_messages(3)
        original_count = len(conv.messages)
        new_msg = Message(role=Role.USER, content="Appended").to_dict()
        delta = ConversationDelta(
            session_id="s1",
            new_messages=[new_msg],
            is_full_sync=False,
        )

        apply_delta(conv, delta)

        assert len(conv.messages) == original_count + 1
        assert conv.messages[-1].content == "Appended"

    def test_remove_messages(self):
        conv = _conv_with_messages(5)
        remove_id = conv.messages[1].id
        delta = ConversationDelta(
            session_id="s1",
            removed_message_ids=[remove_id],
            is_full_sync=False,
        )

        apply_delta(conv, delta)

        assert len(conv.messages) == 4
        assert all(m.id != remove_id for m in conv.messages)

    def test_update_messages(self):
        conv = _conv_with_messages(3)
        target = conv.messages[1]
        updated = target.to_dict()
        updated["content"] = "Updated content"
        delta = ConversationDelta(
            session_id="s1",
            updated_messages=[updated],
            is_full_sync=False,
        )

        apply_delta(conv, delta)

        assert conv.messages[1].content == "Updated content"


class TestComputeResponseDelta:
    def test_captures_new_messages(self):
        conv = _conv_with_messages(3)
        before = len(conv.messages)

        conv.add(Role.ASSISTANT, "New response")
        conv.add(Role.TOOL, "Tool result")

        delta = compute_response_delta(before, conv)
        assert len(delta.new_messages) == 2
        assert delta.new_messages[0]["content"] == "New response"
        assert delta.is_full_sync is False

    def test_empty_when_no_new_messages(self):
        conv = _conv_with_messages(3)
        before = len(conv.messages)

        delta = compute_response_delta(before, conv)
        assert len(delta.new_messages) == 0


class TestConversationDeltaSerialization:
    def test_roundtrip(self):
        delta = ConversationDelta(
            session_id="s1",
            new_messages=[{"id": "m1", "role": "user", "content": "hello"}],
            removed_message_ids=["m0"],
            is_full_sync=False,
            base_message_id="m_base",
        )

        d = delta.to_dict()
        restored = ConversationDelta.from_dict(d)

        assert restored.session_id == "s1"
        assert len(restored.new_messages) == 1
        assert restored.removed_message_ids == ["m0"]
        assert restored.is_full_sync is False
        assert restored.base_message_id == "m_base"

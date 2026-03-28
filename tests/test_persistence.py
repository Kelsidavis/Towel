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

    def test_tags_roundtrip(self):
        conv = _make_conversation()
        conv.tags = ["project", "important"]
        d = conv.to_dict()
        restored = Conversation.from_dict(d)
        assert restored.tags == ["project", "important"]

    def test_tags_absent_defaults_empty(self):
        data = {"id": "x", "channel": "cli", "created_at": datetime.now(timezone.utc).isoformat(), "messages": []}
        conv = Conversation.from_dict(data)
        assert conv.tags == []

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


class TestImportExportRoundtrip:
    """Test that export → import produces identical conversations."""

    def test_json_export_import(self, store, tmp_path):
        from towel.persistence.export import export_json

        conv = _make_conversation("roundtrip-1")
        conv.tags = ["test", "roundtrip"]
        store.save(conv)

        # Export
        exported = export_json(conv)
        export_file = tmp_path / "export.json"
        export_file.write_text(exported, encoding="utf-8")

        # Delete from store
        store.delete("roundtrip-1")
        assert not store.exists("roundtrip-1")

        # Import
        imported_data = json.loads(export_file.read_text(encoding="utf-8"))
        imported_conv = Conversation.from_dict(imported_data)
        store.save(imported_conv)

        assert store.exists("roundtrip-1")
        loaded = store.load("roundtrip-1")
        assert loaded is not None
        assert len(loaded) == 3
        assert loaded.tags == ["test", "roundtrip"]

    def test_import_skips_duplicates(self, store):
        conv = _make_conversation("dup-test")
        store.save(conv)
        assert store.count == 1
        # Saving again with same ID just overwrites, doesn't duplicate
        store.save(conv)
        assert store.count == 1

    def test_import_array_format(self, tmp_path):
        """Test importing an array of conversations from a single file."""
        import_store = ConversationStore(store_dir=tmp_path / "import_test")
        convs = [_make_conversation(f"batch-{i}").to_dict() for i in range(3)]
        batch_file = tmp_path / "batch.json"
        batch_file.write_text(json.dumps(convs), encoding="utf-8")

        data = json.loads(batch_file.read_text())
        for item in data:
            c = Conversation.from_dict(item)
            import_store.save(c)

        assert import_store.count == 3


class TestLogTimeline:
    """Test the data that powers towel log."""

    def test_conversations_sorted_by_recent_activity(self, store):
        from datetime import timedelta

        old = _make_conversation("old")
        old.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        store.save(old)

        new = _make_conversation("new")
        store.save(new)

        # List should have newest first
        summaries = store.list_conversations()
        assert summaries[0].id == "new"

    def test_filter_by_date(self, store):
        from datetime import timedelta

        old = _make_conversation("ancient")
        old.created_at = datetime(2020, 6, 15, tzinfo=timezone.utc)
        store.save(old)

        recent = _make_conversation("fresh")
        store.save(recent)

        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        # Only fresh should pass the filter
        results = []
        for path in store.store_dir.glob("*.json"):
            data = json.loads(path.read_text())
            conv = Conversation.from_dict(data)
            if conv.created_at >= cutoff:
                results.append(conv.id)

        assert "fresh" in results
        assert "ancient" not in results

    def test_tags_in_log_data(self, store):
        conv = _make_conversation("tagged")
        conv.tags = ["work", "project-x"]
        store.save(conv)

        data = json.loads(store._path_for("tagged").read_text())
        assert data["tags"] == ["work", "project-x"]


class TestGarbageCollection:
    """Test that old conversations can be identified and deleted."""

    def test_old_conversation_detected(self, store):
        old = _make_conversation("old-conv")
        old.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        store.save(old)

        recent = _make_conversation("new-conv")
        store.save(recent)

        # Simulate what gc does: find files older than cutoff
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)

        old_ids = []
        for path in store.store_dir.glob("*.json"):
            data = json.loads(path.read_text())
            conv = Conversation.from_dict(data)
            if conv.created_at < cutoff:
                old_ids.append(conv.id)

        assert "old-conv" in old_ids
        assert "new-conv" not in old_ids

    def test_delete_old_conversations(self, store):
        old = _make_conversation("old-conv")
        old.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
        store.save(old)
        recent = _make_conversation("new-conv")
        store.save(recent)

        assert store.count == 2
        store.delete("old-conv")
        assert store.count == 1
        assert store.exists("new-conv")
        assert not store.exists("old-conv")

    def test_all_recent_nothing_to_delete(self, store):
        for i in range(3):
            store.save(_make_conversation(f"recent-{i}"))

        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        old_count = 0
        for path in store.store_dir.glob("*.json"):
            data = json.loads(path.read_text())
            conv = Conversation.from_dict(data)
            if conv.created_at < cutoff:
                old_count += 1

        assert old_count == 0


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

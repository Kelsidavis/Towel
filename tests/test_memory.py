"""Tests for the persistent memory system."""

import pytest

from towel.memory.store import MemoryEntry, MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(store_dir=tmp_path)


class TestMemoryStore:
    def test_remember_and_recall(self, store):
        store.remember("user_name", "Kelsi", memory_type="user")
        entry = store.recall("user_name")
        assert entry is not None
        assert entry.content == "Kelsi"
        assert entry.memory_type == "user"

    def test_remember_updates_existing(self, store):
        store.remember("lang", "Python")
        store.remember("lang", "Rust")
        entry = store.recall("lang")
        assert entry is not None
        assert entry.content == "Rust"
        assert store.count == 1

    def test_forget(self, store):
        store.remember("temp", "delete me")
        assert store.forget("temp")
        assert store.recall("temp") is None
        assert not store.forget("temp")  # already gone

    def test_recall_nonexistent(self, store):
        assert store.recall("nope") is None

    def test_recall_all(self, store):
        store.remember("a", "1", memory_type="fact")
        store.remember("b", "2", memory_type="user")
        store.remember("c", "3", memory_type="fact")

        all_entries = store.recall_all()
        assert len(all_entries) == 3

        facts = store.recall_all(memory_type="fact")
        assert len(facts) == 2

        users = store.recall_all(memory_type="user")
        assert len(users) == 1

    def test_search(self, store):
        store.remember("favorite_language", "Python is great")
        store.remember("favorite_food", "Pizza")
        store.remember("project_deadline", "March 2026")

        results = store.search("favorite")
        assert len(results) == 2

        results = store.search("python")
        assert len(results) == 1

    def test_count(self, store):
        assert store.count == 0
        store.remember("a", "1")
        store.remember("b", "2")
        assert store.count == 2

    def test_persistence_across_instances(self, tmp_path):
        store1 = MemoryStore(store_dir=tmp_path)
        store1.remember("persistent", "I survive restarts")

        store2 = MemoryStore(store_dir=tmp_path)
        entry = store2.recall("persistent")
        assert entry is not None
        assert entry.content == "I survive restarts"

    def test_to_prompt_block_empty(self, store):
        assert store.to_prompt_block() == ""

    def test_to_prompt_block_with_entries(self, store):
        store.remember("name", "Kelsi", memory_type="user")
        store.remember("style", "concise", memory_type="preference")
        store.remember("project", "Towel v0.2", memory_type="project")

        block = store.to_prompt_block()
        assert "Your Memory" in block
        assert "Kelsi" in block
        assert "concise" in block
        assert "Towel v0.2" in block
        assert "remember" in block.lower()

    def test_to_prompt_block_grouped_by_type(self, store):
        store.remember("a", "1", memory_type="user")
        store.remember("b", "2", memory_type="fact")
        block = store.to_prompt_block()
        assert "**User:**" in block
        assert "**Fact:**" in block


class TestCorruptionRecovery:
    """The memory file is the agent's long-term brain — a JSON parse
    failure shouldn't silently destroy every entry on the next save."""

    def test_corrupted_file_is_backed_up_before_reset(self, tmp_path):
        index = tmp_path / "memories.json"
        index.write_text("{not valid json", encoding="utf-8")
        store = MemoryStore(store_dir=tmp_path)
        # The first read should treat the file as empty but move the bad
        # file aside so it isn't clobbered by the next save.
        assert store.count == 0
        backups = list(tmp_path.glob("memories.json.corrupted-*"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "{not valid json"
        # And the canonical path is now absent (so the next save creates
        # a fresh, valid file from scratch).
        assert not index.exists()

    def test_save_after_corruption_writes_fresh_file(self, tmp_path):
        (tmp_path / "memories.json").write_text("definitely not json", encoding="utf-8")
        store = MemoryStore(store_dir=tmp_path)
        store.remember("survivor", "I made it", memory_type="fact")
        # The new save produced a valid file...
        assert (tmp_path / "memories.json").exists()
        # ...and the corrupted backup is still there for the operator.
        backups = list(tmp_path.glob("memories.json.corrupted-*"))
        assert len(backups) == 1

    def test_save_is_atomic(self, tmp_path):
        """Writing to a tmp sibling and renaming means a kill mid-write
        leaves the original file intact rather than truncated. Verify
        the tmp file doesn't linger after a clean save."""
        store = MemoryStore(store_dir=tmp_path)
        store.remember("k", "v")
        # No tmp file left lying around.
        assert not (tmp_path / "memories.json.tmp").exists()
        # The real file is parseable JSON.
        import json

        data = json.loads((tmp_path / "memories.json").read_text())
        assert "k" in data


class TestMemoryEntry:
    def test_serialization_roundtrip(self):
        entry = MemoryEntry(key="test", content="value", memory_type="fact")
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.key == entry.key
        assert restored.content == entry.content
        assert restored.memory_type == entry.memory_type

    def test_str(self):
        entry = MemoryEntry(key="name", content="Kelsi", memory_type="user")
        assert "[user] name: Kelsi" in str(entry)

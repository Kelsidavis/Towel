"""Tests for shared cluster memory synchronization."""

from towel.memory.cluster import ClusterMemorySync, MemoryMutation
from towel.memory.store import MemoryStore


class TestClusterMemorySync:
    def test_remember_records_mutation(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=True)

        entry = sync.remember("user_name", "Alice", "user")
        assert entry.content == "Alice"
        assert sync.version == 1

        pending = sync.drain_pending()
        assert len(pending) == 1
        assert pending[0].action == "remember"
        assert pending[0].key == "user_name"

    def test_forget_records_mutation(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=True)

        sync.remember("temp_fact", "something")
        sync.drain_pending()

        removed = sync.forget("temp_fact")
        assert removed is True
        assert sync.version == 2

        pending = sync.drain_pending()
        assert len(pending) == 1
        assert pending[0].action == "forget"

    def test_apply_mutation_remember(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=False)

        mutation = MemoryMutation(
            action="remember",
            key="remote_fact",
            content="from controller",
            memory_type="fact",
            origin_worker_id="controller",
        )

        applied = sync.apply_mutation(mutation)
        assert applied is True
        assert store.recall("remote_fact") is not None
        assert store.recall("remote_fact").content == "from controller"

    def test_apply_mutation_forget(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=False)
        store.remember("to_delete", "temporary")

        mutation = MemoryMutation(
            action="forget",
            key="to_delete",
            origin_worker_id="controller",
        )
        applied = sync.apply_mutation(mutation)
        assert applied is True
        assert store.recall("to_delete") is None

    def test_snapshot_and_restore(self, tmp_path):
        # Controller side
        controller_store = MemoryStore(store_dir=tmp_path / "controller")
        controller_sync = ClusterMemorySync(controller_store, is_controller=True)
        controller_sync.remember("fact_1", "value_1", "fact")
        controller_sync.remember("user_pref", "dark mode", "preference")
        controller_sync.drain_pending()

        snapshot = controller_sync.snapshot()
        assert snapshot["version"] == 2
        assert len(snapshot["memories"]) == 2

        # Worker side
        worker_store = MemoryStore(store_dir=tmp_path / "worker")
        worker_sync = ClusterMemorySync(worker_store, is_controller=False)

        count = worker_sync.apply_snapshot(snapshot)
        assert count == 2
        assert worker_store.recall("fact_1") is not None
        assert worker_store.recall("user_pref") is not None

    def test_apply_mutations_batch(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=False)

        mutations = [
            MemoryMutation(action="remember", key="a", content="1").to_dict(),
            MemoryMutation(action="remember", key="b", content="2").to_dict(),
            MemoryMutation(action="remember", key="c", content="3").to_dict(),
        ]

        applied = sync.apply_mutations(mutations)
        assert applied == 3
        assert store.count == 3

    def test_apply_mutations_survives_bad_items(self, tmp_path):
        """A worker that ships a malformed memory_sync (non-dict item,
        missing action/key) must not kill the whole batch — and
        crucially must not raise out to the WS handler that called us,
        which would disconnect the worker. Skip the bad items, apply
        the good ones."""
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=False)

        mutations = [
            42,  # not a dict at all
            "string-mutation",  # also not a dict
            None,  # null slipped through
            {"action": "remember"},  # missing key
            {},  # empty dict — missing both action and key
            MemoryMutation(action="remember", key="ok-key", content="x").to_dict(),
            MemoryMutation(action="remember", key="ok-key2", content="y").to_dict(),
        ]

        applied = sync.apply_mutations(mutations)
        # Only the two well-formed mutations applied.
        assert applied == 2
        assert store.recall("ok-key") is not None
        assert store.recall("ok-key2") is not None

    def test_build_sync_message(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=True)

        sync.remember("key1", "val1", origin_worker_id="w1")

        msg = sync.build_sync_message(target_worker_id="w2")
        assert msg["type"] == "memory_sync"
        assert len(msg["mutations"]) == 1

    def test_build_sync_message_excludes_origin(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=True)

        sync.remember("key1", "val1", origin_worker_id="w1")

        # Should not send back to the origin
        msg = sync.build_sync_message(target_worker_id="w1")
        assert msg["type"] == "memory_sync"
        assert len(msg["mutations"]) == 0

    def test_build_sync_message_empty_when_no_pending(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=True)

        msg = sync.build_sync_message()
        assert msg == {}

    def test_build_snapshot_message(self, tmp_path):
        store = MemoryStore(store_dir=tmp_path / "mem")
        sync = ClusterMemorySync(store, is_controller=True)
        sync.remember("fact", "data")
        sync.drain_pending()

        msg = sync.build_snapshot_message()
        assert msg["type"] == "memory_snapshot"
        assert len(msg["memories"]) == 1


class TestMemoryMutation:
    def test_roundtrip(self):
        m = MemoryMutation(
            action="remember",
            key="test",
            content="value",
            memory_type="fact",
            origin_worker_id="w1",
        )
        d = m.to_dict()
        restored = MemoryMutation.from_dict(d)
        assert restored.action == "remember"
        assert restored.key == "test"
        assert restored.content == "value"
        assert restored.origin_worker_id == "w1"

    def test_carries_source_tags_scope(self):
        m = MemoryMutation(
            action="remember",
            key="k",
            content="v",
            memory_type="fact",
            source="auto_capture:role",
            tags=["work", "urgent"],
            scope="proj:demo",
        )
        d = m.to_dict()
        restored = MemoryMutation.from_dict(d)
        assert restored.source == "auto_capture:role"
        assert restored.tags == ["work", "urgent"]
        assert restored.scope == "proj:demo"

    def test_from_dict_tolerates_missing_new_fields(self):
        # A peer running pre-tags/scope code would send only the
        # older fields; the parser must default cleanly.
        d = {
            "action": "remember",
            "key": "k",
            "content": "v",
            "memory_type": "fact",
        }
        restored = MemoryMutation.from_dict(d)
        assert restored.tags == []
        assert restored.scope == ""
        assert restored.source == ""


class TestClusterCarriesAllFields:
    """End-to-end: tags + scope ride from one node through a mutation
    and out the other side intact."""

    def test_remember_propagates_tags_and_scope(self, tmp_path):
        from towel.memory.cluster import ClusterMemorySync
        from towel.memory.store import MemoryStore

        # Controller side (where the write originates).
        ctl_store = MemoryStore(store_dir=tmp_path / "ctl")
        ctl = ClusterMemorySync(store=ctl_store, is_controller=True)
        ctl.remember(
            "role", "engineer", "user",
            source="auto_capture:role",
            tags=["public"],
            scope="proj:alpha",
        )
        pending = ctl.drain_pending()
        assert len(pending) == 1
        m = pending[0]
        assert m.source == "auto_capture:role"
        assert m.tags == ["public"]
        assert m.scope == "proj:alpha"

        # Worker side applies the mutation.
        wk_store = MemoryStore(store_dir=tmp_path / "wk")
        wk = ClusterMemorySync(store=wk_store, is_controller=False)
        wk.apply_mutation(m)
        # The worker's persisted entry must carry every field.
        e = wk_store.recall("role")
        assert e is not None
        assert e.source == "auto_capture:role"
        assert e.tags == ["public"]
        assert e.scope == "proj:alpha"

    def test_snapshot_round_trip_preserves_fields(self, tmp_path):
        from towel.memory.cluster import ClusterMemorySync
        from towel.memory.store import MemoryStore

        ctl_store = MemoryStore(store_dir=tmp_path / "ctl")
        ctl_store.remember(
            "k", "v", "fact",
            source="manual",
            tags=["x", "y"],
            scope="proj:beta",
        )
        ctl = ClusterMemorySync(store=ctl_store, is_controller=True)
        snap = ctl.build_snapshot_message()

        wk_store = MemoryStore(store_dir=tmp_path / "wk")
        wk = ClusterMemorySync(store=wk_store, is_controller=False)
        wk.apply_snapshot(snap)

        e = wk_store.recall("k")
        assert e.tags == ["x", "y"]
        assert e.scope == "proj:beta"
        assert e.source == "manual"

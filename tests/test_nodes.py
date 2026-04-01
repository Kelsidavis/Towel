"""Tests for node capability tracking and cluster scheduling."""

from towel.nodes.capability import ContextSlot, NodeCapability, NodeResources
from towel.nodes.tracker import NodeTracker


class TestNodeResources:
    def test_vram_free(self):
        r = NodeResources(vram_total_mb=16384, vram_used_mb=8000)
        assert r.vram_free_mb == 8384

    def test_vram_utilization(self):
        r = NodeResources(vram_total_mb=16384, vram_used_mb=8192)
        assert r.vram_utilization == 0.5

    def test_zero_vram_utilization(self):
        r = NodeResources(vram_total_mb=0, vram_used_mb=0)
        assert r.vram_utilization == 0.0

    def test_roundtrip_serialization(self):
        r = NodeResources(
            hostname="mac-studio",
            vram_total_mb=65536,
            vram_used_mb=32000,
            ram_total_mb=131072,
            ram_used_mb=64000,
            cpu_count=12,
        )
        d = r.to_dict()
        r2 = NodeResources.from_dict(d)
        assert r2.hostname == "mac-studio"
        assert r2.vram_total_mb == 65536
        assert r2.cpu_count == 12


class TestContextSlot:
    def test_tokens_free(self):
        slot = ContextSlot(session_id="s1", tokens_used=2000, context_window=8192)
        assert slot.tokens_free == 6192

    def test_utilization(self):
        slot = ContextSlot(session_id="s1", tokens_used=4096, context_window=8192)
        assert slot.utilization == 0.5

    def test_zero_context_window(self):
        slot = ContextSlot(session_id="s1", tokens_used=100, context_window=0)
        assert slot.utilization == 0.0
        assert slot.tokens_free == 0


class TestNodeCapability:
    def test_context_pressure_empty(self):
        node = NodeCapability(worker_id="w1", context_window=8192)
        assert node.context_pressure == 0.0
        assert node.active_sessions == 0

    def test_context_pressure_with_sessions(self):
        node = NodeCapability(worker_id="w1", context_window=8192)
        node.add_context_slot("s1", tokens_used=4096)
        assert node.context_pressure == 0.5
        assert node.active_sessions == 1

    def test_add_and_remove_context_slot(self):
        node = NodeCapability(worker_id="w1", context_window=8192)
        node.add_context_slot("s1", tokens_used=1000)
        node.add_context_slot("s2", tokens_used=2000)
        assert node.active_sessions == 2
        assert node.total_context_tokens_used == 3000

        removed = node.remove_context_slot("s1")
        assert removed is True
        assert node.active_sessions == 1
        assert node.total_context_tokens_used == 2000

    def test_update_context_slot(self):
        node = NodeCapability(worker_id="w1", context_window=8192)
        node.add_context_slot("s1", tokens_used=1000)
        updated = node.update_context_slot("s1", 5000)
        assert updated is True
        assert node.total_context_tokens_used == 5000

    def test_can_fit_conversation(self):
        node = NodeCapability(worker_id="w1", context_window=8192)
        assert node.can_fit_conversation(4000) is True
        assert node.can_fit_conversation(10000) is False

    def test_can_fit_unknown_capacity(self):
        node = NodeCapability(worker_id="w1", context_window=0)
        assert node.can_fit_conversation(99999) is True

    def test_from_worker_capabilities(self):
        caps = {
            "hostname": "mac-mini",
            "backend": "mlx",
            "model": "qwen/72b",
            "modes": ["mlx_prompt"],
            "context_window": 32768,
            "max_tokens": 4096,
            "tools": True,
            "resources": {
                "hostname": "mac-mini",
                "vram_total_mb": 65536,
                "vram_used_mb": 40000,
            },
        }
        node = NodeCapability.from_worker_capabilities("w1", caps)
        assert node.backend == "mlx"
        assert node.model == "qwen/72b"
        assert node.context_window == 32768
        assert node.resources.vram_total_mb == 65536
        assert node.tools is True


class TestNodeTracker:
    def test_register_and_get(self):
        tracker = NodeTracker()
        node = tracker.register("w1", {"backend": "mlx", "context_window": 8192})
        assert node.worker_id == "w1"
        assert tracker.get("w1") is not None
        assert len(tracker) == 1

    def test_unregister(self):
        tracker = NodeTracker()
        tracker.register("w1", {"backend": "mlx"})
        removed = tracker.unregister("w1")
        assert removed is not None
        assert len(tracker) == 0

    def test_context_slot_lifecycle(self):
        tracker = NodeTracker()
        tracker.register("w1", {"backend": "mlx", "context_window": 8192})

        slot = tracker.open_context_slot("w1", "session-1", tokens_used=2000)
        assert slot is not None
        assert slot.tokens_used == 2000

        tracker.update_context_usage("w1", "session-1", 5000)
        node = tracker.get("w1")
        assert node is not None
        assert node.total_context_tokens_used == 5000

        tracker.close_context_slot("w1", "session-1")
        assert node.active_sessions == 0

    def test_open_context_slot_deduplicates(self):
        tracker = NodeTracker()
        tracker.register("w1", {"backend": "mlx", "context_window": 8192})
        tracker.open_context_slot("w1", "session-1", tokens_used=1000)
        tracker.open_context_slot("w1", "session-1", tokens_used=2000)
        node = tracker.get("w1")
        assert node is not None
        assert node.active_sessions == 1
        assert node.total_context_tokens_used == 2000

    def test_least_loaded_node(self):
        tracker = NodeTracker()
        tracker.register("w1", {"backend": "mlx", "context_window": 8192})
        tracker.register("w2", {"backend": "mlx", "context_window": 8192})

        tracker.open_context_slot("w1", "s1", tokens_used=6000)
        tracker.open_context_slot("w2", "s2", tokens_used=1000)

        least = tracker.least_loaded_node(backend="mlx")
        assert least is not None
        assert least.worker_id == "w2"

    def test_nodes_with_capacity(self):
        tracker = NodeTracker()
        tracker.register("w1", {"backend": "mlx", "context_window": 4096})
        tracker.register("w2", {"backend": "mlx", "context_window": 32768})

        can_fit = tracker.nodes_with_capacity(8000)
        assert len(can_fit) == 1
        assert can_fit[0].worker_id == "w2"

    def test_cluster_stats(self):
        tracker = NodeTracker()
        tracker.register("w1", {
            "backend": "mlx",
            "context_window": 8192,
            "resources": {"vram_total_mb": 16384, "vram_used_mb": 8000},
        })
        tracker.register("w2", {
            "backend": "mlx",
            "context_window": 8192,
            "resources": {"vram_total_mb": 32768, "vram_used_mb": 10000},
        })
        tracker.open_context_slot("w1", "s1", tokens_used=4096)

        stats = tracker.cluster_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_vram_mb"] == 16384 + 32768
        assert stats["used_vram_mb"] == 8000 + 10000
        assert stats["total_context_tokens"] == 4096
        assert stats["active_sessions"] == 1

    def test_register_preserves_context_slots(self):
        tracker = NodeTracker()
        tracker.register("w1", {"backend": "mlx", "context_window": 8192})
        tracker.open_context_slot("w1", "s1", tokens_used=1000)

        # Re-register (heartbeat with full caps) should keep slots
        tracker.register("w1", {"backend": "mlx", "context_window": 8192})
        node = tracker.get("w1")
        assert node is not None
        assert node.active_sessions == 1

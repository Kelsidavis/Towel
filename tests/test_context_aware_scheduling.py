"""Tests for context-aware worker scheduling."""

import json

from towel.gateway.workers import WorkerRegistry
from towel.nodes.tracker import NodeTracker


class DummyWS:
    def __init__(self):
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))


class TestContextAwareScoring:
    def test_prefers_lower_context_pressure(self):
        registry = WorkerRegistry()
        tracker = NodeTracker()

        registry.register("w1", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})
        registry.register("w2", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})

        tracker.register("w1", {"backend": "mlx", "context_window": 8192})
        tracker.register("w2", {"backend": "mlx", "context_window": 8192})

        # w1 is heavily loaded, w2 is nearly empty
        tracker.open_context_slot("w1", "s_old", tokens_used=7000)
        tracker.open_context_slot("w2", "s_old2", tokens_used=500)

        picked = registry.acquire(
            requirements={"backend": "mlx", "mode": "mlx_prompt"},
            node_tracker=tracker,
        )

        assert picked is not None
        assert picked.id == "w2"

    def test_penalizes_insufficient_capacity(self):
        registry = WorkerRegistry()
        tracker = NodeTracker()

        registry.register("w1", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})
        registry.register("w2", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})

        tracker.register("w1", {"backend": "mlx", "context_window": 2048})
        tracker.register("w2", {"backend": "mlx", "context_window": 32768})

        picked = registry.acquire(
            requirements={
                "backend": "mlx",
                "mode": "mlx_prompt",
                "estimated_tokens": 8000,
            },
            node_tracker=tracker,
        )

        assert picked is not None
        assert picked.id == "w2"

    def test_context_locality_bonus(self):
        registry = WorkerRegistry()
        tracker = NodeTracker()

        registry.register("w1", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})
        registry.register("w2", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})

        tracker.register("w1", {"backend": "mlx", "context_window": 8192})
        tracker.register("w2", {"backend": "mlx", "context_window": 8192})

        # w1 already has session-1's context loaded
        tracker.open_context_slot("w1", "session-1", tokens_used=2000)

        picked = registry.acquire(
            requirements={
                "backend": "mlx",
                "mode": "mlx_prompt",
                "session_id": "session-1",
            },
            node_tracker=tracker,
        )

        assert picked is not None
        assert picked.id == "w1"

    def test_falls_back_without_node_tracker(self):
        registry = WorkerRegistry()
        registry.register("w1", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})
        registry.register("w2", DummyWS(), {"backend": "mlx", "modes": ["mlx_prompt"]})

        picked = registry.acquire(
            requirements={"backend": "mlx", "mode": "mlx_prompt"},
            node_tracker=None,
        )

        assert picked is not None

    def test_least_loaded_without_requirements(self):
        registry = WorkerRegistry()
        tracker = NodeTracker()

        registry.register("w1", DummyWS())
        registry.register("w2", DummyWS())

        tracker.register("w1", {"context_window": 8192})
        tracker.register("w2", {"context_window": 8192})
        tracker.open_context_slot("w1", "s1", tokens_used=6000)

        picked = registry.acquire(node_tracker=tracker)

        assert picked is not None
        assert picked.id == "w2"

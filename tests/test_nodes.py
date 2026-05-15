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

    def test_add_context_slot_caps_oversized_estimate(self):
        """A coordinator-side token estimate larger than the worker's
        context window must be capped — the worker can't physically
        load more than its window, and an uncapped estimate poisons
        context_pressure (saw pressure=1.0 from a single 1MB-probe
        slot on the live coordinator)."""
        node = NodeCapability(worker_id="w1", context_window=8192)
        slot = node.add_context_slot("s1", tokens_used=250_000)
        assert slot.tokens_used == 8192
        # One oversized slot can't single-handedly push pressure
        # above the equivalent of one full window.
        assert node.context_pressure == 1.0
        # And total_context_tokens_used reflects the cap.
        assert node.total_context_tokens_used == 8192

    def test_update_context_slot_caps_oversized_estimate(self):
        """Same cap applies on update — otherwise a later turn with
        a huge user message could re-poison a previously-fine slot."""
        node = NodeCapability(worker_id="w1", context_window=8192)
        node.add_context_slot("s1", tokens_used=1000)
        node.update_context_slot("s1", 250_000)
        assert node.context_slots[0].tokens_used == 8192

    def test_total_tokens_caps_per_slot_on_read(self):
        """Defense in depth: a slot can outlive the cap-on-write fix
        when it was created before the cap was added (the slot
        persists across worker reconnects via
        NodeTracker.register's `node.context_slots = existing.context_slots`).
        Observed live as a stale 250k-token slot on an 8k worker
        permanently pinning context_pressure to 1.0. The read-side
        cap fixes the in-memory state without forcing a restart."""
        node = NodeCapability(worker_id="w1", context_window=8192)
        # Bypass the cap-on-write path entirely — simulate the
        # historical bug where a slot landed with raw bogus tokens.
        from towel.nodes.capability import ContextSlot
        node.context_slots.append(
            ContextSlot(
                session_id="legacy-bad",
                tokens_used=250_000,
                context_window=8192,
            )
        )
        # The raw slot still carries the bogus number — we don't
        # mutate persisted state, only the aggregate read.
        assert node.context_slots[0].tokens_used == 250_000
        # But the aggregate caps the contribution per slot.
        assert node.total_context_tokens_used == 8192
        # And pressure correctly reads as full, not 32x full.
        assert node.context_pressure == 1.0

    def test_total_tokens_caps_handles_multiple_bad_slots(self):
        """Multiple legacy bad slots still cap correctly — each
        contributes at most context_window, so N bad slots → N×window
        (which then clamps to pressure=1.0 via the existing min(1.0)
        in context_pressure)."""
        from towel.nodes.capability import ContextSlot
        node = NodeCapability(worker_id="w1", context_window=8192)
        for i in range(3):
            node.context_slots.append(
                ContextSlot(
                    session_id=f"bad-{i}", tokens_used=250_000,
                    context_window=8192,
                )
            )
        # 3 slots × 8192 cap each = 24576 — well over a single
        # window, but each individual slot couldn't single-handedly
        # poison the aggregate.
        assert node.total_context_tokens_used == 24576
        assert node.context_pressure == 1.0

    def test_total_tokens_unknown_window_skips_per_slot_cap(self):
        """When context_window is 0/unknown, the per-slot cap can't
        be computed — return the raw sum so callers aren't surprised
        by silent data drops on workers that didn't advertise a
        window."""
        from towel.nodes.capability import ContextSlot
        node = NodeCapability(worker_id="w1", context_window=0)
        node.context_slots.append(
            ContextSlot(session_id="s", tokens_used=99_999, context_window=0)
        )
        assert node.total_context_tokens_used == 99_999

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

    def test_from_worker_capabilities_hoists_top_level_vram(self):
        """Live workers report VRAM at the top level (`total_vram_mb`)
        and inside a `gpus` array, NOT inside `resources`. Without the
        hoist, `/cluster/nodes` reported vram_total_mb=0 for every
        worker even when a GPU was present."""
        caps = {
            "hostname": "SparklesMint",
            "backend": "llama",
            "total_vram_mb": 16303,
            "gpus": [{"name": "NVIDIA GeForce RTX 5080", "vram_mb": 16303}],
            "resources": {
                "hostname": "SparklesMint",
                "cpu_count": 32,
                "ram_total_mb": 128709,
                "ram_available_mb": 117307,
            },
        }
        node = NodeCapability.from_worker_capabilities("w-spark", caps)
        assert node.resources.vram_total_mb == 16303
        # ram_used must be derived from total - available since workers
        # report psutil's `ram_available_mb`, not `ram_used_mb`.
        assert node.resources.ram_used_mb == 128709 - 117307

    def test_from_worker_capabilities_sums_gpus_when_total_absent(self):
        """Older workers may report only per-GPU `vram_mb` without a
        top-level total. Sum the array so the cluster view is still
        accurate."""
        caps = {
            "hostname": "dual-gpu",
            "backend": "llama",
            "gpus": [
                {"name": "A", "vram_mb": 8000},
                {"name": "B", "vram_mb": 8000},
            ],
            "resources": {"hostname": "dual-gpu"},
        }
        node = NodeCapability.from_worker_capabilities("w-dual", caps)
        assert node.resources.vram_total_mb == 16000

    def test_from_worker_capabilities_resources_value_wins(self):
        """If a future worker fixes the omission and DOES set
        vram_total_mb inside resources, that value must win over the
        hoist — we shouldn't double-count."""
        caps = {
            "total_vram_mb": 999999,  # ignored
            "resources": {
                "hostname": "self-reporting",
                "vram_total_mb": 8000,
            },
        }
        node = NodeCapability.from_worker_capabilities("w-self", caps)
        assert node.resources.vram_total_mb == 8000

    def test_ram_used_uses_live_value_not_stale_resources(self):
        """`resources.ram_available_mb` is captured once at worker
        startup and never updates. `live_resources.ram_available_mb`
        refreshes every 15s heartbeat. Deriving ram_used from the
        stale value made /cluster/nodes show "RAM used at startup"
        for the whole session — useless for monitoring."""
        caps = {
            "resources": {
                "hostname": "host",
                "ram_total_mb": 100_000,
                "ram_available_mb": 90_000,   # stale: register-time value
            },
            "live_resources": {
                "ram_available_mb": 60_000,    # fresh: current value
            },
        }
        node = NodeCapability.from_worker_capabilities("w", caps)
        # Used = total - LIVE available = 100k - 60k = 40k.
        # If the stale value were used we'd see 10k instead.
        assert node.resources.ram_used_mb == 40_000

    def test_ram_used_falls_back_to_resources_when_no_live(self):
        """When live_resources doesn't carry ram_available_mb (older
        worker without live-reporting code), fall back to the stale
        resources value rather than reporting zero."""
        caps = {
            "resources": {
                "hostname": "host",
                "ram_total_mb": 100_000,
                "ram_available_mb": 90_000,
            },
            "live_resources": {"load_avg_1min": 0.5},
        }
        node = NodeCapability.from_worker_capabilities("w", caps)
        assert node.resources.ram_used_mb == 10_000


class TestAssignRolesDefensiveCoercion:
    """A worker registering with malformed capabilities (non-list
    ``gpus``, non-int ``context_window``, etc.) used to crash
    ``assign_roles`` inside ``sum(g.get(...) ...)`` with
    AttributeError. The exception propagated up through the WS
    register handler and tore down the connection — the worker
    reconnected, hit the same crash, and looped forever.

    These tests pin the defensive coercion so a buggy worker build
    can't take the coordinator's register path down with it."""

    def test_non_list_gpus_does_not_crash(self):
        from towel.nodes.roles import assign_roles
        # String gpus (e.g. a worker that serialized incorrectly):
        # iteration yields chars; .get() on a char would AttributeError.
        roles = assign_roles({"backend": "llama", "gpus": "not-a-list"})
        # No crash; gpus is treated as empty.
        assert roles  # at least CLASSIFIER + GENERAL for llama

    def test_dict_gpus_does_not_crash(self):
        from towel.nodes.roles import assign_roles
        # Dict gpus: iteration yields keys (strings); same crash class.
        roles = assign_roles({"backend": "llama", "gpus": {"a": "b"}})
        assert roles

    def test_none_gpus_does_not_crash(self):
        from towel.nodes.roles import assign_roles
        # Explicit None for gpus — the .get() default of [] would
        # only apply if the key was MISSING; an explicit None bypasses
        # it and breaks `sum(g.get(...) ...)`.
        roles = assign_roles({"backend": "llama", "gpus": None})
        assert roles

    def test_non_dict_gpu_entries_skipped(self):
        from towel.nodes.roles import assign_roles
        # Mixed list with a non-dict entry — the bad entry's .get()
        # raises. Defensive filter drops it instead.
        roles = assign_roles({
            "backend": "llama",
            "gpus": [{"vram_mb": 8000}, "garbage", 42],
        })
        # Valid entry survived, total_vram_mb is computed from it.
        from towel.nodes.roles import NodeRole
        assert NodeRole.INFERENCE in roles

    def test_non_numeric_context_window_treated_as_zero(self):
        from towel.nodes.roles import assign_roles
        # A worker reporting context_window as a string (e.g. "8192"
        # instead of the int) used to compare oddly in `context_window
        # >= 32768` — string vs int compare raises TypeError in
        # Python 3. Coerce non-numeric to 0.
        roles = assign_roles({"backend": "llama", "context_window": "huge"})
        # No crash; treats as 0 context.
        assert roles

    def test_non_string_backend_treated_as_empty(self):
        from towel.nodes.roles import assign_roles
        # A worker registering with backend=42 (typo, buggy serialiser)
        # used to compare oddly against "claude" / "llama" / etc. and
        # silently miss every role check. Coerce to empty string so
        # the role assignment falls through to defaults.
        roles = assign_roles({"backend": 42})
        from towel.nodes.roles import NodeRole
        assert NodeRole.GENERAL in roles

    def test_assign_tasks_also_coerces_defensively(self):
        from towel.nodes.roles import NodeRole, assign_tasks
        # Same defensive shape on assign_tasks — it shares the
        # capabilities dereferencing pattern and would crash the same
        # way on the same inputs.
        tasks = assign_tasks(
            {"backend": "llama", "gpus": "not-a-list", "context_window": None},
            [NodeRole.CLASSIFIER, NodeRole.GENERAL],
        )
        # No crash; returns whatever tasks the roles + (now-zeroed)
        # vram + (zeroed) context_window qualify for.
        assert isinstance(tasks, list)

    def test_worker_quality_tier_handles_non_numeric_vram(self):
        """``int("huge") → ValueError`` would crash quality_tier
        deep inside the doctor probe and fleet panel renders. The
        ``_safe_int`` helper returns 0 for non-numeric inputs, so a
        garbage capability field downgrades the worker to ``low``
        instead of taking out the whole render."""
        from towel.nodes.roles import worker_quality_tier
        assert worker_quality_tier({"total_vram_mb": "huge"}) == "low"
        assert worker_quality_tier({"total_vram_mb": [1, 2]}) == "low"
        assert worker_quality_tier({"total_vram_mb": {"v": 1}}) == "low"
        assert worker_quality_tier({"context_window": "huge"}) == "low"
        # Sanity: valid numeric inputs still work.
        assert worker_quality_tier({"total_vram_mb": 10000}) == "high"
        assert worker_quality_tier({"total_vram_mb": 10000.5}) == "high"

    def test_resources_from_worker_caps_handles_non_numeric_vram(self):
        """``resources_from_worker_caps`` is called from
        ``NodeCapability.from_worker_capabilities`` (register path)
        AND from the NodeTracker heartbeat update. A worker reporting
        ``total_vram_mb: "huge"`` would crash ``int(top_vram)`` and
        500 every /cluster/nodes render. Defensive coercion at the
        boundary keeps the cluster view alive even when one worker's
        capability blob is garbage."""
        from towel.nodes.capability import resources_from_worker_caps
        # Top-level garbage vram falls through to the per-GPU sum.
        r = resources_from_worker_caps({"total_vram_mb": "huge", "gpus": []})
        assert r.vram_total_mb == 0
        # Per-GPU garbage vram coerces to 0 inside the comprehension.
        r = resources_from_worker_caps({
            "gpus": [{"vram_mb": "huge"}, {"vram_mb": 8000}],
        })
        assert r.vram_total_mb == 8000

    def test_resources_from_worker_caps_handles_non_numeric_ram(self):
        """Same defensive shape on the RAM computation — a worker
        reporting ``ram_total_mb`` or ``ram_available_mb`` as a
        garbage value used to crash inside the ``int(...) - int(...)``
        subtraction."""
        from towel.nodes.capability import resources_from_worker_caps
        r = resources_from_worker_caps({
            "resources": {"ram_total_mb": "huge", "ram_available_mb": 5000},
        })
        # garbage total → 0 - 5000 = -5000, then max(0, ...) = 0.
        # No crash.
        assert r.ram_used_mb == 0

    def test_node_meets_task_requirements_handles_non_numeric(self):
        """Same defensive shape on the dispatcher's gate check —
        a non-numeric ``total_vram_mb`` used to crash inside
        ``int(...)`` and 500 the user's routing request."""
        from towel.nodes.roles import TaskType, node_meets_task_requirements
        # CODE_REVIEW has min_vram_mb=4000. A garbage vram value
        # coerces to 0, so the gate correctly says "doesn't meet".
        node = {"capabilities": {"total_vram_mb": "huge", "context_window": 32768}}
        assert node_meets_task_requirements(node, TaskType.CODE_REVIEW) is False
        # Valid numeric still works.
        node_ok = {"capabilities": {"total_vram_mb": 10000, "context_window": 32768}}
        assert node_meets_task_requirements(node_ok, TaskType.CODE_REVIEW) is True


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

    def test_heartbeat_preserves_top_level_vram(self):
        """Workers heartbeat every 15s with the same top-level-vram
        capability shape they used at register. The heartbeat path
        previously called NodeResources.from_dict on the resources
        sub-dict alone, which lacks total_vram_mb — so each heartbeat
        clobbered the vram back to 0. The cluster view then
        oscillated between "correct" (just after register) and
        "useless" (after the next heartbeat tick)."""
        tracker = NodeTracker()
        caps = {
            "hostname": "SparklesMint",
            "backend": "llama",
            "total_vram_mb": 16303,
            "gpus": [{"name": "RTX 5080", "vram_mb": 16303}],
            "resources": {
                "hostname": "SparklesMint",
                "cpu_count": 32,
                "ram_total_mb": 128709,
                "ram_available_mb": 117307,
            },
        }
        node = tracker.register("w-spark", caps)
        assert node.resources.vram_total_mb == 16303

        # Now heartbeat with live_resources added but the same
        # underlying shape. The tracker must keep the vram intact.
        caps_hb = dict(caps)
        caps_hb["live_resources"] = {"load_avg_1min": 2.5, "cpu_pressure": 0.1}
        ok = tracker.update_heartbeat("w-spark", caps_hb)
        assert ok is True
        node_after = tracker.get("w-spark")
        assert node_after is not None
        assert node_after.resources.vram_total_mb == 16303
        # ram_used still derived from total - available.
        assert node_after.resources.ram_used_mb == 128709 - 117307

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

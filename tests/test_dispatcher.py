"""Tests for the fleet dispatcher.

These tests use a minimal in-process WorkerRegistry plus stub builders for the
node-dicts list and idle-task predicate. The dispatcher is intentionally
designed to work without a NodeTracker (it falls back to "no context loaded"),
which is what these tests exercise.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from towel.gateway.dispatcher import (
    REASON_AFFINITY,
    REASON_CAPABILITY_FALLBACK,
    REASON_MOUNT_OWNER,
    REASON_NO_WORKERS,
    REASON_PINNED,
    REASON_PREEMPT_IDLE,
    REASON_RETRY_EMPTY,
    REASON_ROLE_MATCH,
    REASON_TASK_MATCH,
    Dispatcher,
)
from towel.gateway.workers import WorkerInfo, WorkerRegistry
from towel.nodes.roles import NodeRole, TaskType


def _make_worker(
    registry: WorkerRegistry,
    worker_id: str,
    *,
    role: NodeRole = NodeRole.INFERENCE,
    tasks: list[TaskType] | None = None,
    busy: bool = False,
    enabled: bool = True,
    draining: bool = False,
    backend: str = "ollama",
    mode: str = "ollama_chat",
) -> WorkerInfo:
    worker = registry.register(
        worker_id,
        ws=MagicMock(),
        capabilities={
            "roles": [role.value if hasattr(role, "value") else str(role)],
            "tasks": [t.value if hasattr(t, "value") else str(t) for t in (tasks or [])],
            "backend": backend,
            "modes": [mode],
        },
    )
    worker.busy = busy
    worker.enabled = enabled
    worker.draining = draining
    return worker


def _node_dicts(
    workers: WorkerRegistry,
    roles: dict[str, list[NodeRole]] | None = None,
    tasks: dict[str, list[TaskType]] | None = None,
) -> list[dict[str, Any]]:
    """Return the ``best_node_for_*`` input shape for the current registry.

    The role/task selectors compare against the enum values directly, so we
    keep them as ``NodeRole`` / ``TaskType`` instances rather than their string
    forms (which is what the running server also does in ``_build_node_dicts``).
    """
    out: list[dict[str, Any]] = []
    for worker in workers.list():
        rs = (roles or {}).get(worker.id, [])
        ts = (tasks or {}).get(worker.id, [])
        out.append(
            {
                "id": worker.id,
                "busy": worker.busy,
                "enabled": worker.enabled,
                "draining": worker.draining,
                "roles": list(rs),
                "assigned_tasks": list(ts),
                "capabilities": worker.capabilities,
                "active_sessions": 0,
                "context_pressure": 0.0,
                "last_seen": worker.last_seen.isoformat(),
            }
        )
    return out


def _make_dispatcher(
    workers: WorkerRegistry,
    *,
    roles: dict[str, list[NodeRole]] | None = None,
    tasks: dict[str, list[TaskType]] | None = None,
    session_workers: dict[str, str] | None = None,
    session_pins: dict[str, str] | None = None,
    idle_task_workers: set[str] | None = None,
    preempt_hook=None,
    node_tracker: Any = None,
) -> Dispatcher:
    return Dispatcher(
        workers=workers,
        node_dicts_builder=lambda: _node_dicts(workers, roles=roles, tasks=tasks),
        session_workers=session_workers if session_workers is not None else {},
        session_pins=session_pins if session_pins is not None else {},
        node_tracker=node_tracker,
        idle_task_predicate=lambda wid: wid in (idle_task_workers or set()),
        preempt_hook=preempt_hook,
    )


# --------------------------------------------------------------------------- #
# Selection priority                                                           #
# --------------------------------------------------------------------------- #


class TestSelectionLayers:
    def test_pinned_worker_wins_over_everything(self):
        workers = WorkerRegistry()
        _make_worker(workers, "pinned_one")
        _make_worker(workers, "other_one")
        d = _make_dispatcher(workers, session_pins={"s1": "pinned_one"})
        decision = d.select_for_session("s1")
        assert decision.worker is not None
        assert decision.worker.id == "pinned_one"
        assert decision.reason == REASON_PINNED

    def test_busy_pinned_falls_through_to_other_layers(self):
        workers = WorkerRegistry()
        _make_worker(workers, "pinned_busy", busy=True)
        _make_worker(workers, "free_one")
        d = _make_dispatcher(
            workers,
            roles={"free_one": [NodeRole.INFERENCE]},
            session_pins={"s1": "pinned_busy"},
        )
        decision = d.select_for_session("s1", intent="chat")
        # Pin is unusable, so we fall through to a role-matching worker.
        assert decision.worker is not None
        assert decision.worker.id == "free_one"
        # Fall-through must surface that the pin was bypassed so
        # operators can see in /dispatch/recent that their explicit
        # preference was ignored. Without this, the decision looked
        # identical to a normal role-match route and the operator had
        # no fast way to spot "my pin isn't taking effect".
        assert decision.pin_missed is True
        assert decision.pinned_worker_id == "pinned_busy"

    def test_pin_hit_does_not_set_pin_missed(self):
        """When the pin actually fires, pin_missed must stay False —
        the flag specifically means "pin was set and bypassed". A
        successful pinned dispatch is the happy path."""
        workers = WorkerRegistry()
        _make_worker(workers, "pinned_one")
        d = _make_dispatcher(workers, session_pins={"s1": "pinned_one"})
        decision = d.select_for_session("s1")
        assert decision.reason == REASON_PINNED
        assert decision.pin_missed is False

    def test_no_pin_set_does_not_set_pin_missed(self):
        """Sessions with no pin at all must not surface a `pin_missed`
        flag — that would confuse operators who never pinned anything."""
        workers = WorkerRegistry()
        _make_worker(workers, "free_one")
        d = _make_dispatcher(
            workers,
            roles={"free_one": [NodeRole.INFERENCE]},
        )
        decision = d.select_for_session("s1", intent="chat")
        assert decision.pin_missed is False
        assert decision.pinned_worker_id is None

    def test_task_type_match_preferred_when_assigned(self):
        workers = WorkerRegistry()
        _make_worker(workers, "shell_worker")
        _make_worker(workers, "inference_worker")
        d = _make_dispatcher(
            workers,
            roles={
                "shell_worker": [NodeRole.TOOL_WORKER],
                "inference_worker": [NodeRole.INFERENCE],
            },
            tasks={"shell_worker": [TaskType.SHELL]},
        )
        decision = d.select_for_session("s1", intent="tool", task_type=TaskType.SHELL)
        assert decision.worker is not None
        assert decision.worker.id == "shell_worker"
        assert decision.reason == REASON_TASK_MATCH

    def test_capability_fallback_when_preferred_type_unavailable(self):
        """A SHELL task should NOT wait if there's no TOOL_WORKER — it should
        fall back to any idle worker rather than failing or stalling."""
        workers = WorkerRegistry()
        # Only an INFERENCE worker is available.
        _make_worker(workers, "inference_worker")
        d = _make_dispatcher(
            workers,
            roles={"inference_worker": [NodeRole.INFERENCE]},
            tasks={},  # no worker has SHELL assigned
        )
        decision = d.select_for_session(
            "s1", intent="tool", task_type=TaskType.SHELL
        )
        # We fell through task-match and role-match (tool-worker), so capability
        # fallback should have picked the inference worker.
        assert decision.worker is not None
        assert decision.worker.id == "inference_worker"
        assert decision.reason in (REASON_ROLE_MATCH, REASON_CAPABILITY_FALLBACK)

    def test_no_workers_returns_decision_with_no_worker(self):
        workers = WorkerRegistry()
        d = _make_dispatcher(workers)
        decision = d.select_for_session("s1")
        assert decision.worker is None
        assert decision.reason == REASON_NO_WORKERS

    def test_disabled_and_draining_workers_excluded(self):
        workers = WorkerRegistry()
        _make_worker(workers, "draining_one", draining=True)
        _make_worker(workers, "disabled_one", enabled=False)
        _make_worker(workers, "healthy_one")
        d = _make_dispatcher(workers, roles={"healthy_one": [NodeRole.INFERENCE]})
        decision = d.select_for_session("s1", intent="chat")
        assert decision.worker is not None
        assert decision.worker.id == "healthy_one"


# --------------------------------------------------------------------------- #
# Handoff path — the bug we are fixing                                         #
# --------------------------------------------------------------------------- #


class TestHandoffSelection:
    def test_select_for_handoff_skips_affinity_and_picks_replacement(self):
        workers = WorkerRegistry()
        _make_worker(workers, "old_one", draining=True)  # being drained
        _make_worker(workers, "new_one")
        d = _make_dispatcher(
            workers,
            session_workers={"s1": "old_one"},  # session was on old_one
        )
        decision = d.select_for_handoff("s1", exclude={"old_one"})
        assert decision.worker is not None
        assert decision.worker.id == "new_one"

    def test_select_for_handoff_returns_none_when_no_replacement(self):
        workers = WorkerRegistry()
        _make_worker(workers, "only_worker", draining=True)
        d = _make_dispatcher(workers)
        decision = d.select_for_handoff("s1", exclude={"only_worker"})
        assert decision.worker is None
        assert decision.reason == REASON_NO_WORKERS


# --------------------------------------------------------------------------- #
# Preemption                                                                   #
# --------------------------------------------------------------------------- #


class TestPreemption:
    def test_async_select_preempts_idle_task_when_all_workers_busy(self):
        workers = WorkerRegistry()
        _make_worker(workers, "busy_real", busy=True)
        # idle_task_worker is "busy" because it's running an idle task, but the
        # predicate marks it as preemptable.
        _make_worker(workers, "idle_runner", busy=True)
        preempt_called: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempt_called.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            idle_task_workers={"idle_runner"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(d.async_select_for_session("s1"))
        assert decision.preempted_idle is True
        assert decision.reason == REASON_PREEMPT_IDLE
        assert preempt_called == ["idle_runner"]

    def test_preempt_preserves_pin_missed_flag(self):
        """When the original (no-worker) decision had pin_missed=True
        because the pinned worker was busy/draining, the preempt
        path's replacement decision must still carry that flag.
        Without this, the operator's bypassed pin signal evaporates
        the moment a preempt fires — the new "preempted_idle"
        decision looks clean and the dispatch log loses the
        "pin was ignored" trace."""
        workers = WorkerRegistry()
        # All three workers are busy. One is the pinned one (not
        # idle-runner so it can't be preempted), one is busy_real,
        # and one is idle_runner that can be preempted.
        _make_worker(workers, "pinned_busy", busy=True)
        _make_worker(workers, "busy_real", busy=True)
        _make_worker(workers, "idle_runner", busy=True)
        preempt_called: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempt_called.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            idle_task_workers={"idle_runner"},
            session_pins={"s1": "pinned_busy"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(d.async_select_for_session("s1"))
        # Preempt fired and we got idle_runner.
        assert decision.preempted_idle is True
        assert decision.worker is not None
        assert decision.worker.id == "idle_runner"
        # The pin_missed signal survives the preempt replacement.
        assert decision.pin_missed is True
        assert decision.pinned_worker_id == "pinned_busy"

    def test_sync_select_never_preempts(self):
        workers = WorkerRegistry()
        _make_worker(workers, "busy_real", busy=True)
        _make_worker(workers, "idle_runner", busy=True)
        d = _make_dispatcher(workers, idle_task_workers={"idle_runner"})
        decision = d.select_for_session("s1")
        # Sync path can't await preempt_hook, so it just gives up.
        assert decision.worker is None
        assert decision.reason == REASON_NO_WORKERS

    def test_smaller_busy_idle_worker_preempted_for_chat(self):
        """When a chat-class request lands and the smallest qualified
        worker is busy with an idle task, the dispatcher should
        preempt it rather than route to a larger non-busy worker."""
        workers = WorkerRegistry()
        # The small/fast worker is busy with idle work.
        small = _make_worker(workers, "small_busy_idle", busy=True)
        small.capabilities["total_vram_mb"] = 4096
        small.capabilities["assigned_tasks"] = [TaskType.CHAT]
        # The big/slow worker is free.
        big = _make_worker(workers, "big_free", busy=False)
        big.capabilities["total_vram_mb"] = 24000
        big.capabilities["assigned_tasks"] = [TaskType.CHAT]

        preempt_called: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempt_called.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            tasks={"small_busy_idle": [TaskType.CHAT], "big_free": [TaskType.CHAT]},
            roles={
                "small_busy_idle": [NodeRole.CLASSIFIER, NodeRole.INFERENCE],
                "big_free": [NodeRole.CLASSIFIER, NodeRole.INFERENCE],
            },
            idle_task_workers={"small_busy_idle"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(
            d.async_select_for_session("s1", intent="chat", task_type=TaskType.CHAT)
        )
        assert decision.worker is not None
        assert decision.worker.id == "small_busy_idle"
        assert decision.reason == REASON_PREEMPT_IDLE
        assert decision.preempted_idle is True
        assert preempt_called == ["small_busy_idle"]

    def test_bigger_busy_idle_worker_preempted_for_quality(self):
        """Symmetric to test_smaller_busy_idle_worker_preempted_for_chat
        but in the other direction. Live observation: a GENERATE
        task landed on the 4GB small worker because the 24GB big
        worker was running an idle PROACTIVE_HELP. The small worker
        spent 5+ minutes tool-looping while the big worker generated
        diagnostic prose nobody asked for. Bigger-is-better preempt
        on prefer_quality tasks lets the user's real GENERATE land
        where it has enough VRAM to be productive."""
        workers = WorkerRegistry()
        # The big quality-capable worker is busy with idle work.
        big = _make_worker(workers, "big_busy_idle", busy=True)
        big.capabilities["total_vram_mb"] = 24000
        big.capabilities["assigned_tasks"] = [TaskType.GENERATE]
        # The small worker is free.
        small = _make_worker(workers, "small_free", busy=False)
        small.capabilities["total_vram_mb"] = 4096
        small.capabilities["assigned_tasks"] = [TaskType.GENERATE]

        preempt_called: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempt_called.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            tasks={"big_busy_idle": [TaskType.GENERATE], "small_free": [TaskType.GENERATE]},
            roles={
                "big_busy_idle": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER],
                "small_free": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER],
            },
            idle_task_workers={"big_busy_idle"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(
            d.async_select_for_session("s1", intent="task", task_type=TaskType.GENERATE)
        )
        assert decision.worker is not None
        assert decision.worker.id == "big_busy_idle"
        assert decision.reason == REASON_PREEMPT_IDLE
        assert decision.preempted_idle is True
        assert preempt_called == ["big_busy_idle"]

    def test_bigger_preempt_works_when_assigned_tasks_unset(self):
        """Live observation: workers in the running cluster reported
        `assigned_tasks=None` (a general-purpose default). The
        bigger-preempt filter rejected every such worker, so the
        prefer_quality path silently fell back to the smaller worker
        on every PLAN/GENERATE while the big worker idled on lint
        or proactive_help. `assigned_tasks` is an explicit operator
        override — when unset it must mean "any task," not "no task."
        """
        workers = WorkerRegistry()
        # Big worker busy on idle work and declares NO assigned_tasks —
        # this is the live-cluster shape.
        big = _make_worker(workers, "big_busy_idle", busy=True)
        big.capabilities["total_vram_mb"] = 24000
        # No assigned_tasks key at all.
        small = _make_worker(workers, "small_free", busy=False)
        small.capabilities["total_vram_mb"] = 4096

        preempt_called: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempt_called.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            tasks={
                "big_busy_idle": [TaskType.GENERATE],
                "small_free": [TaskType.GENERATE],
            },
            roles={
                "big_busy_idle": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER],
                "small_free": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER],
            },
            idle_task_workers={"big_busy_idle"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(
            d.async_select_for_session("s1", intent="task", task_type=TaskType.GENERATE)
        )
        assert decision.worker is not None
        assert decision.worker.id == "big_busy_idle"
        assert decision.reason == REASON_PREEMPT_IDLE
        assert preempt_called == ["big_busy_idle"]

    def test_smaller_preempt_works_when_assigned_tasks_unset(self):
        """Mirror of test_bigger_preempt_works_when_assigned_tasks_unset
        for the prefer_fast preempt path. A small worker with no
        explicit assigned_tasks should still be preempt-eligible for
        chat-like tasks — otherwise the live cluster pulls every
        TRIAGE/CHAT to the heaviest worker until it saturates.
        """
        workers = WorkerRegistry()
        big = _make_worker(workers, "big_free", busy=False)
        big.capabilities["total_vram_mb"] = 24000
        small = _make_worker(workers, "small_busy_idle", busy=True)
        small.capabilities["total_vram_mb"] = 4096
        # No assigned_tasks declared on either.

        preempt_called: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempt_called.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            tasks={
                "big_free": [TaskType.CHAT],
                "small_busy_idle": [TaskType.CHAT],
            },
            roles={
                "big_free": [NodeRole.INFERENCE],
                "small_busy_idle": [NodeRole.INFERENCE],
            },
            idle_task_workers={"small_busy_idle"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(
            d.async_select_for_session("s1", intent="chat", task_type=TaskType.CHAT)
        )
        assert decision.worker is not None
        assert decision.worker.id == "small_busy_idle"
        assert decision.reason == REASON_PREEMPT_IDLE
        assert preempt_called == ["small_busy_idle"]

    def test_no_bigger_preempt_when_picked_is_already_biggest(self):
        """If the quality path picked the bigger worker, no preempt
        should happen even when a smaller worker is busy with idle."""
        workers = WorkerRegistry()
        big_free = _make_worker(workers, "big_free", busy=False)
        big_free.capabilities["total_vram_mb"] = 24000
        big_free.capabilities["assigned_tasks"] = [TaskType.GENERATE]
        small_idle = _make_worker(workers, "small_idle", busy=True)
        small_idle.capabilities["total_vram_mb"] = 4096
        small_idle.capabilities["assigned_tasks"] = [TaskType.GENERATE]

        preempted: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempted.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            tasks={"big_free": [TaskType.GENERATE], "small_idle": [TaskType.GENERATE]},
            idle_task_workers={"small_idle"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(
            d.async_select_for_session("s1", intent="task", task_type=TaskType.GENERATE)
        )
        assert decision.worker is not None
        assert decision.worker.id == "big_free"
        # No preempt — we already had the biggest worker available.
        assert preempted == []

    def test_no_smaller_preempt_when_picked_is_already_smallest(self):
        """If the chat path picked the smaller worker, no preempt
        should happen even when a larger worker is busy with idle."""
        workers = WorkerRegistry()
        small_free = _make_worker(workers, "small_free", busy=False)
        small_free.capabilities["total_vram_mb"] = 4096
        small_free.capabilities["assigned_tasks"] = [TaskType.CHAT]
        big_idle = _make_worker(workers, "big_idle", busy=True)
        big_idle.capabilities["total_vram_mb"] = 24000
        big_idle.capabilities["assigned_tasks"] = [TaskType.CHAT]

        preempted: list[str] = []

        async def preempt(w: WorkerInfo) -> None:
            preempted.append(w.id)
            workers.release(w.id)

        d = _make_dispatcher(
            workers,
            tasks={"small_free": [TaskType.CHAT], "big_idle": [TaskType.CHAT]},
            idle_task_workers={"big_idle"},
            preempt_hook=preempt,
        )
        decision = asyncio.run(
            d.async_select_for_session("s1", intent="chat", task_type=TaskType.CHAT)
        )
        assert decision.worker is not None
        assert decision.worker.id == "small_free"
        # No preempt — we already had the fastest worker available.
        assert preempted == []


# --------------------------------------------------------------------------- #
# History / observability                                                      #
# --------------------------------------------------------------------------- #


class TestExplain:
    def test_explain_does_not_record_into_history(self):
        workers = WorkerRegistry()
        _make_worker(workers, "w1")
        d = _make_dispatcher(workers, roles={"w1": [NodeRole.INFERENCE]})

        # Real selects are recorded…
        d.select_for_session("real", intent="chat")
        baseline = len(d.history())

        # …but explain peeks shouldn't pollute the ring buffer.
        decision = d.explain_for_session("peek", intent="chat")
        assert decision.worker is not None
        assert decision.worker.id == "w1"
        assert len(d.history()) == baseline

    def test_explain_returns_same_choice_as_select_when_no_state_change(self):
        workers = WorkerRegistry()
        _make_worker(workers, "w1")
        d = _make_dispatcher(workers, roles={"w1": [NodeRole.INFERENCE]})
        live = d.select_for_session("s1", intent="chat")
        peek = d.explain_for_session("s1", intent="chat")
        # Same worker, same reason — explain is just a read-only mirror.
        assert peek.worker is not None and live.worker is not None
        assert peek.worker.id == live.worker.id
        assert peek.reason == live.reason


class TestObservability:
    def test_history_records_each_decision(self):
        workers = WorkerRegistry()
        _make_worker(workers, "w1")
        d = _make_dispatcher(workers, roles={"w1": [NodeRole.INFERENCE]})
        for sid in ("s1", "s2", "s3"):
            d.select_for_session(sid, intent="chat")
        hist = d.history()
        assert len(hist) == 3
        assert [h.session_id for h in hist] == ["s1", "s2", "s3"]
        assert all(h.worker is not None for h in hist)

    def test_decision_to_dict_serializable(self):
        workers = WorkerRegistry()
        _make_worker(workers, "w1")
        d = _make_dispatcher(workers, roles={"w1": [NodeRole.INFERENCE]})
        decision = d.select_for_session("s1", intent="chat")
        as_dict = decision.to_dict()
        assert as_dict["worker_id"] == "w1"
        assert as_dict["reason"] in (REASON_AFFINITY, REASON_ROLE_MATCH, REASON_CAPABILITY_FALLBACK)
        assert "timestamp" in as_dict
        # Decisions must be JSON-friendly.
        import json
        json.dumps(as_dict)

    def test_history_capped_at_configured_size(self):
        workers = WorkerRegistry()
        _make_worker(workers, "w1")
        d = Dispatcher(
            workers=workers,
            node_dicts_builder=lambda: _node_dicts(workers, roles={"w1": [NodeRole.INFERENCE]}),
            session_workers={},
            session_pins={},
            history_size=3,
        )
        for sid in ("s1", "s2", "s3", "s4", "s5"):
            d.select_for_session(sid, intent="chat")
        hist = d.history()
        assert len(hist) == 3
        assert [h.session_id for h in hist] == ["s3", "s4", "s5"]

    def test_record_retry_appears_in_history(self):
        """The empty-response retry path lives outside select_for_session;
        without record_retry it wouldn't show up in /dispatch/recent and
        operators couldn't see that a fallback happened."""
        workers = WorkerRegistry()
        _make_worker(workers, "primary")
        alt = _make_worker(workers, "alt")
        d = _make_dispatcher(
            workers, roles={"primary": [NodeRole.INFERENCE], "alt": [NodeRole.INFERENCE]},
        )

        # First a primary dispatch lands via the normal path (we don't
        # care which worker the dispatcher chose — only that the retry
        # decision goes on top of it).
        d.select_for_session("s-retry", intent="chat")
        baseline = len(d.history())

        # Then the retry is recorded externally.
        retry_decision = d.record_retry(
            session_id="s-retry",
            retry_worker=alt,
            original_worker_id="primary",
            intent="chat",
        )

        hist = d.history()
        assert len(hist) == baseline + 1
        assert hist[-1] is retry_decision
        assert hist[-1].reason == REASON_RETRY_EMPTY
        assert hist[-1].previous_worker_id == "primary"
        assert hist[-1].worker is alt
        # And the retry decision is JSON-friendly so /dispatch/recent
        # can serialize it.
        import json
        json.dumps(hist[-1].to_dict())

    def test_record_mount_redirect_appears_in_history(self):
        """The data-locality override reroutes outside select_for_session;
        recording it keeps /dispatch/recent honest about where the request ran."""
        workers = WorkerRegistry()
        _make_worker(workers, "small")
        owner = _make_worker(workers, "spark")
        d = _make_dispatcher(
            workers, roles={"small": [NodeRole.INFERENCE], "spark": [NodeRole.INFERENCE]},
        )
        d.select_for_session("s-mnt", intent="tool")
        baseline = len(d.history())

        decision = d.record_mount_redirect(
            session_id="s-mnt",
            owner=owner,
            original_worker_id="small",
            intent="tool",
        )
        hist = d.history()
        assert len(hist) == baseline + 1
        assert hist[-1] is decision
        assert hist[-1].reason == REASON_MOUNT_OWNER
        assert hist[-1].previous_worker_id == "small"
        assert hist[-1].worker is owner
        import json
        json.dumps(hist[-1].to_dict())

    def test_record_retry_notes_reflect_cause(self):
        """The retry path used to hardcode 'retry after empty
        response from X' in notes, but the same code is also called
        from the primary_failed (timeout/exception) branch. Operators
        triaging a timed-out request saw "empty response" — wrong and
        actively misleading. Threading the cause through gives the
        notes line accurate failure semantics."""
        workers = WorkerRegistry()
        _make_worker(workers, "primary")
        alt = _make_worker(workers, "alt")
        d = _make_dispatcher(workers)

        # Default cause: empty text (preserve existing behavior).
        empty_retry = d.record_retry(
            session_id="s-empty",
            retry_worker=alt,
            original_worker_id="primary",
            intent="chat",
        )
        assert "empty response" in empty_retry.notes

        # primary_failed cause: notes say "failed", not "empty".
        failed_retry = d.record_retry(
            session_id="s-failed",
            retry_worker=alt,
            original_worker_id="primary",
            intent="chat",
            cause="primary_failed",
        )
        assert "failed" in failed_retry.notes
        assert "empty response" not in failed_retry.notes

        # Free-text cause (e.g. with timeout details) surfaces verbatim.
        timeout_retry = d.record_retry(
            session_id="s-timeout",
            retry_worker=alt,
            original_worker_id="primary",
            intent="chat",
            cause="primary_failed: worker primary did not respond within 60s",
        )
        assert "did not respond within 60s" in timeout_retry.notes

    def test_empty_text_retry_counts_groups_by_previous_worker(self):
        """A heterogeneous fleet routinely sees one model produce empty
        text (tool calls instead of chat) far more often than another.
        Each such turn costs the user the primary's full latency before
        the retry runs. Operators need a per-worker tally surfaced
        from the buffer — without it, "worker X is flaky" requires
        eyeballing every retry entry."""
        workers = WorkerRegistry()
        _make_worker(workers, "primary")
        _make_worker(workers, "other_primary")
        alt = _make_worker(workers, "alt")
        d = _make_dispatcher(workers)

        # Three empty-text retries from "primary", one from "other_primary".
        for sid in ("s1", "s2", "s3"):
            d.record_retry(
                session_id=sid,
                retry_worker=alt,
                original_worker_id="primary",
                intent="chat",
            )
        d.record_retry(
            session_id="s4",
            retry_worker=alt,
            original_worker_id="other_primary",
            intent="chat",
        )
        # primary_failed retries share the reason code but have
        # different notes ("failed", not "empty response") and should
        # NOT inflate the empty-text tally — operators would mistake a
        # slow worker for a flaky one.
        d.record_retry(
            session_id="s5",
            retry_worker=alt,
            original_worker_id="primary",
            intent="chat",
            cause="primary_failed",
        )

        counts = d.empty_text_retry_counts()
        assert counts == {"primary": 3, "other_primary": 1}

    def test_empty_text_retry_counts_empty_buffer(self):
        """Dispatcher with no retries reports an empty dict so callers
        can render "no flaky workers" without special-casing None."""
        workers = WorkerRegistry()
        _make_worker(workers, "primary")
        d = _make_dispatcher(workers)
        assert d.empty_text_retry_counts() == {}

    def test_empty_text_counts_by_worker_aggregates_primary_decisions(self):
        """Counts every empty-text response, including single-worker
        cases where no retry is possible. Distinct from the retry
        tally — that one only fires when an alternate worker was
        tried, which can't happen on a one-worker fleet."""
        workers = WorkerRegistry()
        _make_worker(workers, "solo")
        d = _make_dispatcher(workers)
        # Three dispatches, two of which returned empty.
        for empty in (True, False, True):
            decision = asyncio.run(
                d.async_select_for_session("s", intent="chat"),
            )
            assert decision.worker is not None
            decision.record_completion(
                ttft_ms=None, total_ms=100.0, empty_text=empty,
            )
            workers.release(decision.worker.id)
        counts = d.empty_text_counts_by_worker()
        assert counts == {"solo": 2}
        # The retry tally is still 0 — no alternate was tried.
        assert d.empty_text_retry_counts() == {}

    def test_empty_text_counts_by_worker_empty_when_all_clean(self):
        workers = WorkerRegistry()
        _make_worker(workers, "solo")
        d = _make_dispatcher(workers)
        decision = asyncio.run(
            d.async_select_for_session("s", intent="chat"),
        )
        decision.record_completion(ttft_ms=10.0, total_ms=50.0, empty_text=False)
        assert d.empty_text_counts_by_worker() == {}


# --------------------------------------------------------------------------- #
# Excludes                                                                     #
# --------------------------------------------------------------------------- #


def _make_worker_with_caps(
    registry: WorkerRegistry,
    worker_id: str,
    *,
    vram_mb: int = 0,
    context_window: int = 0,
    role: NodeRole = NodeRole.INFERENCE,
    tasks: list[TaskType] | None = None,
) -> WorkerInfo:
    """Variant of ``_make_worker`` that lets a test set VRAM + context."""
    from unittest.mock import MagicMock as _MagicMock
    worker = registry.register(
        worker_id,
        ws=_MagicMock(),
        capabilities={
            "roles": [role.value if hasattr(role, "value") else str(role)],
            "tasks": [t.value if hasattr(t, "value") else str(t) for t in (tasks or [])],
            "backend": "ollama",
            "modes": ["ollama_chat"],
            "total_vram_mb": vram_mb,
            "context_window": context_window,
        },
    )
    return worker


def _node_dicts_with_caps(
    workers: WorkerRegistry,
    roles: dict[str, list[NodeRole]] | None = None,
    tasks: dict[str, list[TaskType]] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "id": w.id,
            "busy": w.busy,
            "enabled": w.enabled,
            "draining": w.draining,
            "roles": list((roles or {}).get(w.id, [])),
            "assigned_tasks": list((tasks or {}).get(w.id, [])),
            "capabilities": w.capabilities,
            "active_sessions": 0,
            "context_pressure": 0.0,
            "last_seen": w.last_seen.isoformat(),
        }
        for w in workers.list()
    ]


class TestQualityDegradation:
    def test_flags_when_routed_worker_is_under_spec(self):
        """A SHELL task on a worker with no VRAM and a 4k context is fine
        (SHELL doesn't demand much). A CODE_REVIEW on that same worker
        should be flagged as quality_degraded because CODE_REVIEW declares
        min_vram_mb=4000 and min_context=32768.
        """
        workers = WorkerRegistry()
        _make_worker_with_caps(
            workers, "tiny_worker",
            vram_mb=0, context_window=4096,
            tasks=[TaskType.CODE_REVIEW, TaskType.SHELL],
        )

        d = Dispatcher(
            workers=workers,
            node_dicts_builder=lambda: _node_dicts_with_caps(
                workers,
                roles={"tiny_worker": [NodeRole.INFERENCE]},
                tasks={"tiny_worker": [TaskType.CODE_REVIEW, TaskType.SHELL]},
            ),
            session_workers={},
            session_pins={},
        )

        # SHELL has min_vram_mb=0 and min_context=8192. The tiny worker has
        # context=4096 so SHELL is also degraded.
        shell_decision = d.select_for_session(
            "s1", intent="tool", task_type=TaskType.SHELL
        )
        assert shell_decision.worker is not None
        assert shell_decision.quality_degraded is True

        # CODE_REVIEW is even more demanding — also flagged.
        review_decision = d.select_for_session(
            "s2", intent="task", task_type=TaskType.CODE_REVIEW
        )
        assert review_decision.worker is not None
        assert review_decision.quality_degraded is True

    def test_clean_when_worker_meets_requirements(self):
        workers = WorkerRegistry()
        _make_worker_with_caps(
            workers, "beefy_worker",
            vram_mb=24000, context_window=131072,
            tasks=[TaskType.CODE_REVIEW],
        )
        d = Dispatcher(
            workers=workers,
            node_dicts_builder=lambda: _node_dicts_with_caps(
                workers,
                roles={"beefy_worker": [NodeRole.INFERENCE]},
                tasks={"beefy_worker": [TaskType.CODE_REVIEW]},
            ),
            session_workers={},
            session_pins={},
        )
        decision = d.select_for_session(
            "s1", intent="task", task_type=TaskType.CODE_REVIEW
        )
        assert decision.worker is not None
        assert decision.quality_degraded is False

    def test_prefers_qualified_worker_over_under_spec(self):
        """When the fleet has both a tiny and a beefy worker, the dispatcher
        should pick the beefy one even if the tiny one was registered first."""
        workers = WorkerRegistry()
        _make_worker_with_caps(
            workers, "tiny",
            vram_mb=0, context_window=4096,
            tasks=[TaskType.CODE_REVIEW],
        )
        _make_worker_with_caps(
            workers, "beefy",
            vram_mb=24000, context_window=131072,
            tasks=[TaskType.CODE_REVIEW],
        )
        d = Dispatcher(
            workers=workers,
            node_dicts_builder=lambda: _node_dicts_with_caps(
                workers,
                roles={"tiny": [NodeRole.INFERENCE], "beefy": [NodeRole.INFERENCE]},
                tasks={
                    "tiny": [TaskType.CODE_REVIEW],
                    "beefy": [TaskType.CODE_REVIEW],
                },
            ),
            session_workers={},
            session_pins={},
        )
        decision = d.select_for_session(
            "s1", intent="task", task_type=TaskType.CODE_REVIEW
        )
        assert decision.worker is not None
        assert decision.worker.id == "beefy"
        assert decision.quality_degraded is False

    def test_falls_back_to_under_spec_when_no_qualified_worker(self):
        """The coordinator must be adaptable: if only a tiny worker is online,
        route to it but flag the degradation so operators see what's
        happening."""
        workers = WorkerRegistry()
        _make_worker_with_caps(
            workers, "only_tiny",
            vram_mb=0, context_window=4096,
            tasks=[TaskType.CODE_REVIEW],
        )
        d = Dispatcher(
            workers=workers,
            node_dicts_builder=lambda: _node_dicts_with_caps(
                workers,
                roles={"only_tiny": [NodeRole.INFERENCE]},
                tasks={"only_tiny": [TaskType.CODE_REVIEW]},
            ),
            session_workers={},
            session_pins={},
        )
        decision = d.select_for_session(
            "s1", intent="task", task_type=TaskType.CODE_REVIEW
        )
        assert decision.worker is not None
        assert decision.worker.id == "only_tiny"
        assert decision.quality_degraded is True
        # The notes string must mention the degradation so operators reading
        # /dispatch/recent can see what happened without inspecting the flag.
        assert "under-spec" in decision.notes


class TestAffinityMiss:
    def test_affinity_miss_flagged_when_previous_worker_busy(self):
        workers = WorkerRegistry()
        _make_worker(workers, "stale_affinity", busy=True)
        _make_worker(workers, "replacement")
        d = _make_dispatcher(
            workers,
            roles={"replacement": [NodeRole.INFERENCE]},
            session_workers={"s1": "stale_affinity"},  # session was on stale_affinity
        )
        decision = d.select_for_session("s1", intent="chat")
        assert decision.worker is not None
        assert decision.worker.id == "replacement"
        assert decision.affinity_missed is True
        assert decision.previous_worker_id == "stale_affinity"

    def test_affinity_miss_flagged_when_previous_worker_draining(self):
        workers = WorkerRegistry()
        _make_worker(workers, "draining_affinity", draining=True)
        _make_worker(workers, "replacement")
        d = _make_dispatcher(
            workers,
            roles={"replacement": [NodeRole.INFERENCE]},
            session_workers={"s1": "draining_affinity"},
        )
        decision = d.select_for_session("s1", intent="chat")
        assert decision.affinity_missed is True
        assert decision.previous_worker_id == "draining_affinity"

    def test_affinity_miss_false_when_no_prior_affinity(self):
        workers = WorkerRegistry()
        _make_worker(workers, "w1")
        d = _make_dispatcher(workers, roles={"w1": [NodeRole.INFERENCE]})
        decision = d.select_for_session("brand_new_session", intent="chat")
        assert decision.affinity_missed is False
        assert decision.previous_worker_id is None

    def test_successful_affinity_does_not_set_previous_worker_id(self):
        """Successful session_affinity (worker reachable AND holds the
        context) means nothing was displaced — the session lands on
        the same worker it was on before. Setting previous_worker_id
        to the chosen worker_id (the prior bug) confused operators
        reading /dispatch/recent into thinking a migration happened
        when nothing moved. None correctly signals "no displacement";
        affinity_missed=False already conveys the success state."""
        from unittest.mock import MagicMock
        workers = WorkerRegistry()
        affinity = _make_worker(workers, "affinity-worker")
        # Stub the NodeTracker so _has_context_loaded returns True
        # (otherwise the dispatcher falls through to affinity_missed,
        # which isn't the path we care about here).
        node_tracker = MagicMock()
        node = MagicMock()
        # _has_context_loaded calls node.get_context_slot(session_id);
        # returning a truthy value lands in the REASON_AFFINITY branch.
        node.get_context_slot.return_value = MagicMock()
        node_tracker.get.return_value = node
        d = _make_dispatcher(
            workers,
            roles={"affinity-worker": [NodeRole.INFERENCE]},
            session_workers={"s1": "affinity-worker"},
            node_tracker=node_tracker,
        )
        decision = d.select_for_session("s1", intent="task")
        assert decision.worker is affinity
        assert decision.reason == "session_affinity"
        assert decision.affinity_missed is False
        # The key assertion: no displacement → no previous_worker_id.
        assert decision.previous_worker_id is None, (
            f"successful affinity set previous_worker_id="
            f"{decision.previous_worker_id!r}; expected None"
        )

    def test_affinity_miss_serializes_to_dict(self):
        workers = WorkerRegistry()
        _make_worker(workers, "stale", busy=True)
        _make_worker(workers, "active")
        d = _make_dispatcher(
            workers,
            roles={"active": [NodeRole.INFERENCE]},
            session_workers={"s1": "stale"},
        )
        decision = d.select_for_session("s1", intent="chat")
        as_dict = decision.to_dict()
        assert as_dict["affinity_missed"] is True
        assert as_dict["previous_worker_id"] == "stale"


class TestExclusions:
    def test_excluded_workers_never_selected(self):
        workers = WorkerRegistry()
        _make_worker(workers, "excluded")
        _make_worker(workers, "available")
        d = _make_dispatcher(
            workers,
            roles={"excluded": [NodeRole.INFERENCE], "available": [NodeRole.INFERENCE]},
        )
        decision = d.select_for_session("s1", intent="chat", exclude={"excluded"})
        assert decision.worker is not None
        assert decision.worker.id == "available"


class TestDispatcherDefensiveCoercion:
    """The preempt path reads ``total_vram_mb`` off worker
    capabilities to find a SMALLER idle worker to evict. A worker
    reporting non-numeric vram used to crash ``int(...)`` inside
    that comparison and 500 the user request that triggered the
    dispatch. Defensive coercion at the boundary keeps the preempt
    path resilient to garbage capability fields."""

    def test_smaller_idle_worker_survives_non_numeric_vram(self):
        # Manually construct workers with garbage vram on one of
        # them. The preempt scan iterates all idle-task workers
        # looking for a smaller match; the non-numeric one used to
        # crash the iteration.
        workers = WorkerRegistry()
        picked = _make_worker(workers, "picked")
        # Inject non-numeric vram on the candidate worker. The
        # `_make_worker` factory doesn't set vram by default; mutate
        # the capabilities post-register so the test exercises the
        # malformed-input path directly.
        bad = _make_worker(workers, "bad-vram")
        bad.capabilities["total_vram_mb"] = "huge"
        bad.busy = True  # idle_task_workers requires busy
        picked.capabilities["total_vram_mb"] = 24000

        d = _make_dispatcher(
            workers,
            idle_task_workers={"bad-vram"},
        )
        # The preempt selector runs inside async_select_for_session
        # when all workers are busy. We can also exercise the
        # private helper directly — both paths share the same
        # defensive code now.
        from towel.nodes.roles import TaskType
        result = d._smaller_idle_worker_for_task(
            picked=picked, task=TaskType.CHAT, excluded=set(),
        )
        # No crash. bad-vram has no assigned_tasks so it doesn't
        # match — but the iteration ran through without exploding
        # on the garbage capability field.
        assert result is None

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
    REASON_NO_WORKERS,
    REASON_PINNED,
    REASON_PREEMPT_IDLE,
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
) -> Dispatcher:
    return Dispatcher(
        workers=workers,
        node_dicts_builder=lambda: _node_dicts(workers, roles=roles, tasks=tasks),
        session_workers=session_workers if session_workers is not None else {},
        session_pins=session_pins if session_pins is not None else {},
        node_tracker=None,
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

    def test_sync_select_never_preempts(self):
        workers = WorkerRegistry()
        _make_worker(workers, "busy_real", busy=True)
        _make_worker(workers, "idle_runner", busy=True)
        d = _make_dispatcher(workers, idle_task_workers={"idle_runner"})
        decision = d.select_for_session("s1")
        # Sync path can't await preempt_hook, so it just gives up.
        assert decision.worker is None
        assert decision.reason == REASON_NO_WORKERS


# --------------------------------------------------------------------------- #
# History / observability                                                      #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Excludes                                                                     #
# --------------------------------------------------------------------------- #


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

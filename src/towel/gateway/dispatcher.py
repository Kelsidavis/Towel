"""Fleet dispatcher — chooses which worker handles each incoming request.

This module centralises worker selection so the coordinator has one place to:

- evaluate explicit pins, session affinity, task-type preferences, and role
  fallbacks in a defined priority order;
- fall back to capability-compatible workers when no preferred-type worker
  is available (a SHELL task lands on a TOOL_WORKER if one exists, otherwise
  on any worker whose capabilities admit it — rather than waiting on a busy
  preferred worker);
- preempt idle tasks when the fleet is otherwise saturated;
- emit a structured ``DispatchDecision`` for every selection, so operators
  can ask "why did this request go to worker X?" after the fact.

The previous coordinator scattered this logic across ``_worker_for_task``,
``_worker_for_role``, the body of ``route_to_worker``, and a non-existent
``_select_worker`` referenced from the handoff path (AttributeError on
every drain). Routing the handoff through this dispatcher fixes that bug
in addition to making routing decisions inspectable.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from towel.gateway.workers import WorkerInfo, WorkerRegistry
from towel.nodes.roles import (
    NodeRole,
    TaskType,
    best_node_for_role,
    best_node_for_task,
    node_meets_task_requirements,
)

log = logging.getLogger("towel.gateway.dispatcher")


# A function that converts a WorkerInfo into the dict-shape ``best_node_for_task``
# expects. Injected so the dispatcher doesn't depend on the rest of the server.
NodeDictBuilder = Callable[[], list[dict[str, Any]]]

# Optional async hook for preempting an idle task on a worker. The dispatcher
# only calls this when the entire fleet is otherwise busy.
PreemptHook = Callable[[WorkerInfo], Awaitable[None]]


# Reason codes — stable strings suitable for logging, metrics, and APIs.
REASON_PINNED = "pinned"
REASON_AFFINITY = "session_affinity"
REASON_TASK_MATCH = "task_type_match"
REASON_ROLE_MATCH = "role_match"
REASON_CAPABILITY_FALLBACK = "capability_fallback"
REASON_PREEMPT_IDLE = "preempt_idle"
REASON_NO_WORKERS = "no_workers_available"


@dataclass
class DispatchDecision:
    """Structured explanation of one routing decision."""

    worker: WorkerInfo | None
    intent: str  # "chat" | "tool" | "task"
    task_type: TaskType | None = None
    reason: str = REASON_NO_WORKERS
    notes: str = ""
    candidates_considered: int = 0
    session_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    preempted_idle: bool = False
    # True when the session had a recorded affinity worker but it couldn't be
    # used (busy, draining, disabled, or missing the session's context slot),
    # forcing a migration to a different worker. Operators can grep
    # ``/dispatch/recent`` for these to spot a thrashing session.
    affinity_missed: bool = False
    previous_worker_id: str | None = None
    # True when the selected worker doesn't meet the task's declared
    # ``min_vram_mb`` / ``min_context`` minimums — the coordinator had to
    # adapt to the fleet it has rather than refuse, but operators should
    # see the degradation and either upgrade a worker or accept the
    # quality drop.
    quality_degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker.id if self.worker else None,
            "intent": self.intent,
            "task_type": str(self.task_type) if self.task_type else None,
            "reason": self.reason,
            "notes": self.notes,
            "candidates_considered": self.candidates_considered,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "preempted_idle": self.preempted_idle,
            "affinity_missed": self.affinity_missed,
            "previous_worker_id": self.previous_worker_id,
            "quality_degraded": self.quality_degraded,
        }


class Dispatcher:
    """Selects workers for incoming requests with explicit fallback layers.

    Priority order in :meth:`select_for_session`:

    1. **Explicit pin** — ``session_pins[session_id]`` overrides everything,
       provided the pinned worker is enabled and idle.
    2. **Session affinity** — the worker that already holds this session's
       context, again provided it is enabled and idle. A busy affinity worker
       is skipped (the previous behaviour) but ``allow_busy_affinity`` can
       re-enable falling through to other layers more aggressively.
    3. **Task-type preference** — if a ``TaskType`` was supplied, pick the
       best worker that has that task assigned.
    4. **Role match** — match the request's intent to a node role
       (``chat`` → INFERENCE, ``tool`` → TOOL_WORKER, ``task`` → INFERENCE
       then GENERAL).
    5. **Capability fallback** — when nothing above matched, take *any*
       idle, enabled, non-draining worker that satisfies basic capability
       requirements. This is the fix for the head-of-line stall where a
       SHELL task would wait for the lone TOOL_WORKER even when an
       INFERENCE node was sitting at 10% load.
    6. **Idle preemption** — when no worker is idle, ask the optional
       ``preempt_hook`` to stop an idle task on a worker. (The hook is
       awaited by ``async_select_for_session``; the synchronous variant
       skips this layer.)
    7. **Give up** — return a decision whose ``worker`` is ``None`` so the
       coordinator can handle the request itself.

    The dispatcher does *not* mutate ``session_workers`` directly. Callers
    are expected to update affinity after acting on a decision, so a failed
    send doesn't pollute affinity state.
    """

    def __init__(
        self,
        workers: WorkerRegistry,
        node_dicts_builder: NodeDictBuilder,
        session_workers: dict[str, str],
        session_pins: dict[str, str],
        node_tracker: Any | None = None,
        idle_task_predicate: Callable[[str], bool] | None = None,
        preempt_hook: PreemptHook | None = None,
        history_size: int = 50,
    ) -> None:
        self._workers = workers
        self._build_node_dicts = node_dicts_builder
        self._session_workers = session_workers
        self._session_pins = session_pins
        self._node_tracker = node_tracker
        self._is_idle_task = idle_task_predicate or (lambda _wid: False)
        self._preempt_hook = preempt_hook
        self._history: deque[DispatchDecision] = deque(maxlen=history_size)

    # ─── Public API ──────────────────────────────────────────────────────

    def history(self) -> list[DispatchDecision]:
        """Return the most recent dispatch decisions (newest last)."""
        return list(self._history)

    def select_for_session(
        self,
        session_id: str,
        intent: str = "task",
        task_type: TaskType | None = None,
        estimated_tokens: int = 0,
        exclude: Iterable[str] | None = None,
    ) -> DispatchDecision:
        """Pick a worker for a request without preempting anything.

        Use :meth:`async_select_for_session` instead when you want layer 6
        (idle preemption) to be considered.
        """
        excluded = set(exclude or ())
        decision = self._layered_select(
            session_id=session_id,
            intent=intent,
            task_type=task_type,
            estimated_tokens=estimated_tokens,
            excluded=excluded,
            allow_preempt=False,
        )
        self._record(decision)
        return decision

    def explain_for_session(
        self,
        session_id: str,
        intent: str = "task",
        task_type: TaskType | None = None,
        estimated_tokens: int = 0,
        exclude: Iterable[str] | None = None,
    ) -> DispatchDecision:
        """Preview where a request *would* be routed without side effects.

        Identical to :meth:`select_for_session` except: the decision is **not**
        recorded in the history ring buffer, the dispatch log line is skipped,
        and no preemption can happen (this mirrors the sync select path). Use
        this from operator-facing tooling — e.g. a ``/dispatch/explain``
        endpoint — so peeking at routing doesn't pollute the recent-decisions
        history with synthetic entries.
        """
        excluded = set(exclude or ())
        return self._layered_select(
            session_id=session_id,
            intent=intent,
            task_type=task_type,
            estimated_tokens=estimated_tokens,
            excluded=excluded,
            allow_preempt=False,
        )

    async def async_select_for_session(
        self,
        session_id: str,
        intent: str = "task",
        task_type: TaskType | None = None,
        estimated_tokens: int = 0,
        exclude: Iterable[str] | None = None,
    ) -> DispatchDecision:
        """Async variant that can preempt an idle task (layer 6)."""
        excluded = set(exclude or ())
        decision = self._layered_select(
            session_id=session_id,
            intent=intent,
            task_type=task_type,
            estimated_tokens=estimated_tokens,
            excluded=excluded,
            allow_preempt=True,
        )
        # Layer 6a: no-worker preempt. Same path as before — every
        # worker is busy, so any idle-task interrupt opens a slot.
        if (
            decision.worker is None
            and self._preempt_hook is not None
        ):
            preempt = self._try_preempt(excluded)
            if preempt is not None:
                worker, notes = preempt
                await self._preempt_hook(worker)
                decision = DispatchDecision(
                    worker=worker,
                    intent=intent,
                    task_type=task_type,
                    reason=REASON_PREEMPT_IDLE,
                    notes=notes,
                    candidates_considered=decision.candidates_considered,
                    session_id=session_id,
                    preempted_idle=True,
                )
        # Layer 6b: smaller-is-better preempt. The layered path
        # already picked a non-busy worker, but for prefer_fast
        # tasks (CHAT, TRIAGE, LINT) we'd rather interrupt an idle
        # task on a SMALLER worker than use a larger non-busy one.
        # Without this, every chat request mid-startup-stampede
        # lands on the heavy worker just because the fast one is
        # cycling through email_triage and friends.
        elif (
            decision.worker is not None
            and self._preempt_hook is not None
            and task_type is not None
        ):
            from towel.nodes.roles import TASK_REQUIREMENTS as _TR
            reqs = _TR.get(task_type, {})
            if reqs.get("prefer_fast"):
                smaller = self._smaller_idle_worker_for_task(
                    picked=decision.worker, task=task_type, excluded=excluded,
                )
                if smaller is not None:
                    await self._preempt_hook(smaller)
                    decision = DispatchDecision(
                        worker=smaller,
                        intent=intent,
                        task_type=task_type,
                        reason=REASON_PREEMPT_IDLE,
                        notes=(
                            f"preempted idle task on {smaller.id} "
                            f"(smaller model than {decision.worker.id})"
                        ),
                        candidates_considered=decision.candidates_considered,
                        session_id=session_id,
                        preempted_idle=True,
                    )
        self._record(decision)
        return decision

    def _smaller_idle_worker_for_task(
        self, *, picked: WorkerInfo, task: TaskType, excluded: set[str]
    ) -> WorkerInfo | None:
        """Find a smaller-VRAM worker currently running an idle task.

        Eligible iff: enabled, not draining, currently running an
        idle task (per the idle-task predicate), assigned to the
        same task type as ``task``, AND has smaller total_vram_mb
        than ``picked``. Returns None when no such worker exists —
        the original picked worker is fine to use.
        """
        picked_vram = int(
            (picked.capabilities or {}).get("total_vram_mb") or 0
        )
        for w in self._workers.list():
            if w.id in excluded or w.id == picked.id:
                continue
            if not w.enabled or w.draining:
                continue
            if not self._is_idle_task(w.id):
                continue
            caps = w.capabilities or {}
            if task not in (caps.get("assigned_tasks") or []):
                # Worker isn't allowed this task type even if it
                # weren't busy — don't preempt for nothing.
                continue
            w_vram = int(caps.get("total_vram_mb") or 0)
            # Strictly smaller, with a hard floor so a 0-vram
            # worker (no GPU advertised) doesn't beat anything.
            if 0 < w_vram < picked_vram:
                return w
        return None

    def select_for_handoff(
        self,
        session_id: str,
        estimated_tokens: int = 0,
        exclude: Iterable[str] | None = None,
    ) -> DispatchDecision:
        """Pick a replacement worker after a drain/disconnect handoff.

        This intentionally skips layers 1-2 (a handoff *is* moving the session
        off its current worker, so pins/affinity for the old worker would be
        wrong) and goes straight to capability fallback so we always find a
        target as long as one suitable worker exists. This is what the old
        ``_select_worker`` reference was supposed to call.
        """
        excluded = set(exclude or ())
        worker = self._capability_fallback(estimated_tokens, excluded, session_id)
        if worker is not None:
            decision = DispatchDecision(
                worker=worker,
                intent="task",
                reason=REASON_CAPABILITY_FALLBACK,
                notes="handoff target",
                candidates_considered=len(self._idle_workers(excluded)),
                session_id=session_id,
            )
        else:
            decision = DispatchDecision(
                worker=None,
                intent="task",
                reason=REASON_NO_WORKERS,
                notes="no idle workers available for handoff",
                session_id=session_id,
            )
        self._record(decision)
        return decision

    # ─── Internal layered selection ──────────────────────────────────────

    def _layered_select(
        self,
        *,
        session_id: str,
        intent: str,
        task_type: TaskType | None,
        estimated_tokens: int,
        excluded: set[str],
        allow_preempt: bool,  # noqa: ARG002 — used by caller after this returns
    ) -> DispatchDecision:
        # Layer 1: explicit pin
        pinned_id = self._session_pins.get(session_id)
        if pinned_id and pinned_id not in excluded:
            worker = self._workers.get(pinned_id)
            if worker and worker.enabled and not worker.busy and not worker.draining:
                return DispatchDecision(
                    worker=worker,
                    intent="task",
                    reason=REASON_PINNED,
                    notes=f"session pinned to worker {pinned_id}",
                    candidates_considered=1,
                    session_id=session_id,
                )

        # Layer 2: session affinity
        affinity_id = self._session_workers.get(session_id)
        affinity_missed = False
        if affinity_id and affinity_id not in excluded:
            affinity_worker = self._workers.get(affinity_id)
            if affinity_worker and self._is_routable(affinity_worker):
                if self._has_context_loaded(affinity_worker, session_id):
                    return DispatchDecision(
                        worker=affinity_worker,
                        intent="task",
                        task_type=task_type,
                        reason=REASON_AFFINITY,
                        notes=f"context loaded on worker {affinity_id}",
                        candidates_considered=1,
                        session_id=session_id,
                        previous_worker_id=affinity_id,
                    )
                # Worker is reachable but doesn't currently hold the context;
                # falling through means a cold transfer — flag it as a miss.
                affinity_missed = True
            else:
                # Affinity worker exists but is busy/draining/disabled — also
                # a context migration event worth surfacing to operators.
                affinity_missed = True

        # Layer 3: task-type preference
        if task_type is not None:
            worker = self._best_for_task(task_type, session_id, excluded)
            if worker is not None:
                degraded = self._worker_is_under_spec(worker, task_type)
                return DispatchDecision(
                    worker=worker,
                    intent=intent,
                    task_type=task_type,
                    reason=REASON_TASK_MATCH,
                    notes=f"worker assigned task {task_type}"
                    + (" (under-spec — quality degraded)" if degraded else ""),
                    candidates_considered=self._candidate_count(),
                    session_id=session_id,
                    affinity_missed=affinity_missed,
                    previous_worker_id=affinity_id if affinity_missed else None,
                    quality_degraded=degraded,
                )

        # Layer 4: role match for the request intent
        role = _role_for_intent(intent)
        if role is not None:
            worker = self._best_for_role(role, session_id, excluded)
            if worker is not None:
                degraded = self._worker_is_under_spec(worker, task_type)
                return DispatchDecision(
                    worker=worker,
                    intent=intent,
                    task_type=task_type,
                    reason=REASON_ROLE_MATCH,
                    notes=f"role={role}"
                    + (" (under-spec — quality degraded)" if degraded else ""),
                    candidates_considered=self._candidate_count(),
                    session_id=session_id,
                    affinity_missed=affinity_missed,
                    previous_worker_id=affinity_id if affinity_missed else None,
                    quality_degraded=degraded,
                )

        # Try the GENERAL role before falling all the way through to "any
        # idle worker" — preserves the previous behaviour for fleets that
        # actually have a GENERAL node.
        general_worker = self._best_for_role(NodeRole.GENERAL, session_id, excluded)
        if general_worker is not None:
            return DispatchDecision(
                worker=general_worker,
                intent=intent,
                task_type=task_type,
                reason=REASON_ROLE_MATCH,
                notes="role=general (fallback)",
                candidates_considered=self._candidate_count(),
                session_id=session_id,
                affinity_missed=affinity_missed,
                previous_worker_id=affinity_id if affinity_missed else None,
            )

        # Layer 5: capability fallback — any idle, non-draining worker.
        worker = self._capability_fallback(estimated_tokens, excluded, session_id)
        if worker is not None:
            degraded = self._worker_is_under_spec(worker, task_type)
            return DispatchDecision(
                worker=worker,
                intent=intent,
                task_type=task_type,
                reason=REASON_CAPABILITY_FALLBACK,
                notes="no preferred-type worker available; using any idle worker"
                + (" (under-spec — quality degraded)" if degraded else ""),
                candidates_considered=len(self._idle_workers(excluded)),
                session_id=session_id,
                affinity_missed=affinity_missed,
                previous_worker_id=affinity_id if affinity_missed else None,
                quality_degraded=degraded,
            )

        # Layer 7 (preempt is layer 6, handled by caller in async variant)
        return DispatchDecision(
            worker=None,
            intent=intent,
            task_type=task_type,
            reason=REASON_NO_WORKERS,
            notes="all workers busy or absent",
            candidates_considered=0,
            session_id=session_id,
            affinity_missed=affinity_missed,
            previous_worker_id=affinity_id if affinity_missed else None,
        )

    # ─── Selection helpers ───────────────────────────────────────────────

    def _best_for_task(
        self, task: TaskType, session_id: str, excluded: set[str]
    ) -> WorkerInfo | None:
        nodes = [n for n in self._build_node_dicts() if n["id"] not in excluded]
        best = best_node_for_task(task, nodes, exclude_busy=True, session_id=session_id)
        return self._workers.get(best["id"]) if best else None

    def _best_for_role(
        self, role: NodeRole, session_id: str, excluded: set[str]
    ) -> WorkerInfo | None:
        nodes = [n for n in self._build_node_dicts() if n["id"] not in excluded]
        best = best_node_for_role(role, nodes, exclude_busy=True, session_id=session_id)
        return self._workers.get(best["id"]) if best else None

    def _capability_fallback(
        self, estimated_tokens: int, excluded: set[str], session_id: str | None
    ) -> WorkerInfo | None:
        """Pick the least-loaded idle worker without role/task constraints."""
        idle = self._idle_workers(excluded)
        if not idle:
            return None
        requirements: dict[str, Any] = {}
        if estimated_tokens:
            requirements["estimated_tokens"] = estimated_tokens
        if session_id:
            requirements["session_id"] = session_id
        return self._workers.acquire(
            requirements=requirements or None,
            node_tracker=self._node_tracker,
        )

    def _try_preempt(self, excluded: set[str]) -> tuple[WorkerInfo, str] | None:
        for worker in self._workers.list():
            if worker.id in excluded:
                continue
            if not worker.enabled or worker.draining:
                continue
            if self._is_idle_task(worker.id):
                return worker, f"preempted idle task on {worker.id}"
        return None

    def _idle_workers(self, excluded: set[str]) -> list[WorkerInfo]:
        return [
            w
            for w in self._workers.list()
            if w.id not in excluded and self._is_routable(w)
        ]

    def _is_routable(self, worker: WorkerInfo) -> bool:
        return worker.enabled and not worker.busy and not worker.draining

    def _has_context_loaded(self, worker: WorkerInfo, session_id: str) -> bool:
        if self._node_tracker is None:
            # Without a NodeTracker we can't tell whether context is loaded;
            # treating "affinity exists" as proof would cause false positives,
            # so be conservative and accept affinity only when we can confirm.
            return False
        node = self._node_tracker.get(worker.id)
        return bool(node and node.get_context_slot(session_id) is not None)

    def _candidate_count(self) -> int:
        return sum(1 for w in self._workers.list() if self._is_routable(w))

    def _worker_is_under_spec(self, worker: WorkerInfo, task: TaskType | None) -> bool:
        """Return True iff this worker doesn't meet ``task``'s declared minimums.

        Used to set ``DispatchDecision.quality_degraded`` so operators can see
        when the coordinator had to adapt down (e.g. routed a CODE_REVIEW to
        a small-model worker because no bigger one was available).
        """
        if task is None:
            return False
        # Build the dict shape ``node_meets_task_requirements`` expects.
        node = {"capabilities": worker.capabilities or {}}
        return not node_meets_task_requirements(node, task)

    def _record(self, decision: DispatchDecision) -> None:
        self._history.append(decision)
        log.info(
            "dispatch: session=%s intent=%s reason=%s worker=%s (notes: %s)",
            decision.session_id,
            decision.intent,
            decision.reason,
            decision.worker.id if decision.worker else "<none>",
            decision.notes,
        )
        if decision.affinity_missed:
            log.warning(
                "affinity-miss: session=%s previous=%s now=%s — context will be migrated",
                decision.session_id,
                decision.previous_worker_id,
                decision.worker.id if decision.worker else "<none>",
            )
        if decision.quality_degraded:
            log.warning(
                "quality-degraded: task=%s routed to under-spec worker=%s — "
                "no fleet member meets the declared minimums",
                decision.task_type,
                decision.worker.id if decision.worker else "<none>",
            )


def _role_for_intent(intent: str) -> NodeRole | None:
    """Pick the canonical NodeRole for a coordinator intent.

    chat falls through to CLASSIFIER, not INFERENCE — chat-class
    queries want the fastest worker, not the biggest. INFERENCE
    sorts by descending VRAM (biggest first), which on a fleet with
    a small + large worker silently routed every "what's 2+2?"
    request to the heavy model. CLASSIFIER's score prefers small
    VRAM, matching the intent.

    task stays on INFERENCE because that's where the heavy work
    actually belongs (refactor, explain, generate).
    """
    if intent == "chat":
        return NodeRole.CLASSIFIER
    if intent == "tool":
        return NodeRole.TOOL_WORKER
    if intent == "task":
        return NodeRole.INFERENCE
    return None

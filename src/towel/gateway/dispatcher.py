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
# When the primary worker returned empty text and the coordinator
# retried on a different worker. Used by /api/ask and
# /v1/chat/completions so the dispatch log surfaces retries —
# without this entry, operators looking at /dispatch/recent saw
# only the failed primary and couldn't tell that a retry rescued
# the request (or didn't).
REASON_RETRY_EMPTY = "retry_empty_text"

# Recorded when /api/ask (or another endpoint) opted into the
# ensemble collaboration mode. The individual per-worker fan-out
# decisions are tracked under ephemeral _ens_<...> session_ids;
# this aggregate captures the operator-visible session that
# triggered the run so /dispatch/recent has one entry per user
# request rather than only the ephemeral fan-outs (which are
# hidden by default — see the dispatch_recent endpoint).
REASON_ENSEMBLE = "ensemble"

# Recorded when verify=true ran a second-worker review after the
# primary returned. Symmetric to REASON_ENSEMBLE for the sequential
# collaboration mode.
REASON_VERIFY = "verify"


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
    # True when the session had an EXPLICIT pin (`POST
    # /sessions/<id>/pin-worker`) but the pinned worker was busy /
    # draining / disabled at dispatch time, forcing the dispatcher to
    # silently route elsewhere. The pin is an operator-set preference;
    # without this flag a bypassed pin looked identical to a normal
    # task-match decision and the operator had no way to see that
    # their pin was being ignored.
    pin_missed: bool = False
    pinned_worker_id: str | None = None
    # True when the selected worker doesn't meet the task's declared
    # ``min_vram_mb`` / ``min_context`` minimums — the coordinator had to
    # adapt to the fleet it has rather than refuse, but operators should
    # see the degradation and either upgrade a worker or accept the
    # quality drop.
    quality_degraded: bool = False
    # Timing data filled in by the inference path after the request
    # completes. None until the response lands; populated in-place via
    # `record_completion` below so the dispatch log shows operators
    # how long each routing decision actually took to satisfy.
    ttft_ms: float | None = None
    total_ms: float | None = None

    def record_completion(self, *, ttft_ms: float | None, total_ms: float | None) -> None:
        """Stamp post-dispatch timing onto this decision.

        Inference paths (_quick_remote_infer / _step_remote_inference)
        call this when the worker's response lands so the dispatch
        log surfaces cold-vs-warm without operators having to grep
        worker logs. Idempotent — second call overwrites; harmless
        when called with None.
        """
        if ttft_ms is not None:
            self.ttft_ms = float(ttft_ms)
        if total_ms is not None:
            self.total_ms = float(total_ms)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
            "pin_missed": self.pin_missed,
            "pinned_worker_id": self.pinned_worker_id,
        }
        if self.ttft_ms is not None:
            out["ttft_ms"] = round(self.ttft_ms, 1)
        if self.total_ms is not None:
            out["total_ms"] = round(self.total_ms, 1)
        return out


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

    def empty_text_retry_counts(self) -> dict[str, int]:
        """Tally retry_empty_text decisions by their failing primary worker.

        On a heterogeneous fleet, one worker tends to produce empty
        text far more often than the others — small models routinely
        emit tool calls instead of chat text for trivial prompts, and
        every such turn costs the user the primary's full latency
        before the retry runs on the alt. The retry decisions are
        already in the ring buffer; counting them by
        ``previous_worker_id`` surfaces "worker X has had N empty-text
        retries in the recent window" without forcing operators to
        eyeball every entry.
        """
        counts: dict[str, int] = {}
        for d in self._history:
            if d.reason != REASON_RETRY_EMPTY:
                continue
            prev = d.previous_worker_id
            if not prev:
                continue
            # Only the actual empty-text retries — primary_failed
            # retries (timeout / exception) live in the same reason
            # code but carry "failed" in notes. Counting them under
            # "empty text" would conflate a slow worker with a flaky
            # one, and the operator-facing signal is different.
            if "empty response" not in (d.notes or ""):
                continue
            counts[prev] = counts.get(prev, 0) + 1
        return counts

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
                # Propagate the original decision's miss flags onto
                # the preempt-replacement decision. Without this, a
                # session whose explicit pin or context affinity was
                # bypassed lost that signal the moment a preempt
                # fired — operators saw a clean "preempted_idle" entry
                # with no hint that the operator's pin had been
                # ignored or context was being thrown away.
                decision = DispatchDecision(
                    worker=worker,
                    intent=intent,
                    task_type=task_type,
                    reason=REASON_PREEMPT_IDLE,
                    notes=notes,
                    candidates_considered=decision.candidates_considered,
                    session_id=session_id,
                    preempted_idle=True,
                    affinity_missed=decision.affinity_missed,
                    previous_worker_id=decision.previous_worker_id,
                    pin_missed=decision.pin_missed,
                    pinned_worker_id=decision.pinned_worker_id,
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
                    # Propagate miss flags onto the smaller-preempt
                    # decision too (same rationale as the no-worker
                    # preempt path above).
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
                        affinity_missed=decision.affinity_missed,
                        previous_worker_id=decision.previous_worker_id,
                        pin_missed=decision.pin_missed,
                        pinned_worker_id=decision.pinned_worker_id,
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

        Uses ``_safe_int`` for the same crash-class reason the
        capability fields in roles.py get coerced — a worker
        reporting ``total_vram_mb`` as a non-numeric value would
        otherwise crash ``int(...)`` deep in this preempt path
        and 500 the user request that triggered the dispatch.
        """
        from towel.nodes.capability import _safe_int
        picked_vram = _safe_int(
            (picked.capabilities or {}).get("total_vram_mb")
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
            w_vram = _safe_int(caps.get("total_vram_mb"))
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
        pin_missed = False
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
                    pinned_worker_id=pinned_id,
                )
            # Pin set but unusable — surface as `pin_missed` on the
            # downstream decision so operators can see in
            # /dispatch/recent that their pin was silently bypassed
            # (busy/draining/disabled at dispatch time). Without this
            # the bypass looked identical to a normal route.
            pin_missed = True

        # Layer 2: session affinity
        affinity_id = self._session_workers.get(session_id)
        affinity_missed = False
        if affinity_id and affinity_id not in excluded:
            affinity_worker = self._workers.get(affinity_id)
            if affinity_worker and self._is_routable(affinity_worker):
                if self._has_context_loaded(affinity_worker, session_id):
                    # Successful affinity: the session lands on the SAME
                    # worker it was on before. `previous_worker_id` is
                    # meant to indicate displacement (a worker that was
                    # bypassed or replaced) — leaving it equal to the
                    # chosen worker_id confused operators reading
                    # /dispatch/recent into thinking a migration had
                    # happened when nothing actually moved. None is the
                    # honest "no displacement" signal; affinity_missed=
                    # False already conveys the success state.
                    return DispatchDecision(
                        worker=affinity_worker,
                        intent="task",
                        task_type=task_type,
                        reason=REASON_AFFINITY,
                        notes=f"context loaded on worker {affinity_id}",
                        candidates_considered=1,
                        session_id=session_id,
                        previous_worker_id=None,
                        pin_missed=pin_missed,
                        pinned_worker_id=pinned_id if pin_missed else None,
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
                    pin_missed=pin_missed,
                    pinned_worker_id=pinned_id if pin_missed else None,
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
                    pin_missed=pin_missed,
                    pinned_worker_id=pinned_id if pin_missed else None,
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
                pin_missed=pin_missed,
                pinned_worker_id=pinned_id if pin_missed else None,
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
                pin_missed=pin_missed,
                pinned_worker_id=pinned_id if pin_missed else None,
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
            pin_missed=pin_missed,
            pinned_worker_id=pinned_id if pin_missed else None,
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

    def last_decision_for_session(self, session_id: str) -> DispatchDecision | None:
        """Most-recent decision recorded for ``session_id``, or None.

        Used by the inference paths to stamp ttft/total timing onto
        the dispatch decision after the response lands — gives the
        operator a single place (dispatch log) to see both who got
        chosen AND how long it took.
        """
        for d in reversed(self._history):
            if d.session_id == session_id:
                return d
        return None

    def record_ensemble(
        self,
        session_id: str,
        contributions: list[dict[str, Any]],
        arbitration_mode: str,
    ) -> DispatchDecision:
        """Log an ensemble run as an aggregate dispatch decision.

        The fan-out workers each get their own decisions under
        ephemeral session_ids (hidden from /dispatch/recent by
        default). This top-level entry surfaces under the user's
        actual session_id so the operator can see "session X ran
        ensemble" alongside other dispatch events.

        Stamps total_ms from the longest contributing worker
        (parallel fan-out runs bound by the slowest worker that
        actually answered) plus any synthesis_ms reported. Gives
        operators latency-at-a-glance for the aggregate entry
        without needing to drill into the per-worker decisions.
        """
        contributing = [c.get("worker_id", "?") for c in contributions if c.get("answer")]
        # Slowest worker that returned text bounds the fan-out
        # wall-clock; failed/timeout workers don't extend it. Record
        # whenever at least one contribution succeeded — even a
        # near-zero ms (mocked-worker test scenarios) is still a
        # legitimate timing signal worth surfacing.
        any_answered = any(c.get("answer") for c in contributions)
        worker_ms = max(
            (c.get("ms", 0.0) for c in contributions if c.get("answer")),
            default=0.0,
        )
        synth_ms = next(
            (c.get("synthesis_ms") for c in contributions if c.get("synthesis_ms")),
            None,
        )
        total_ms = (
            round(worker_ms + (synth_ms or 0.0), 1) if any_answered else None
        )
        # Empty-contributions case = "ensemble was requested but had no
        # candidate workers" — every worker was busy/draining/disabled at
        # the moment of fan-out. The user opted in to ensemble; without
        # an aggregate log entry they had no way to see why their
        # request got handled as a single-worker dispatch instead. Show
        # the skip distinctly from "ensemble ran but every worker
        # failed" (which is `0/N answered`).
        if not contributions:
            notes = "ensemble: skipped (no idle workers available)"
        elif contributing:
            notes = (
                f"ensemble: {arbitration_mode} "
                f"({len(contributing)}/{len(contributions)} answered: "
                f"{', '.join(contributing)})"
            )
        else:
            # Every worker tried but none returned a real answer
            # (errors, empty text, or timeouts). Drop the trailing
            # ":" + empty list — operators read this in /dispatch/recent
            # and the dangling colon kept making people ask whether
            # the entry was truncated.
            notes = (
                f"ensemble: {arbitration_mode} "
                f"(0/{len(contributions)} answered)"
            )
        decision = DispatchDecision(
            worker=None,
            intent="task",
            reason=REASON_ENSEMBLE,
            notes=notes,
            session_id=session_id,
            candidates_considered=len(contributions),
        )
        if total_ms is not None:
            decision.total_ms = total_ms
        self._record(decision)
        return decision

    def record_verify(
        self,
        session_id: str,
        verifier_id: str | None,
        primary_id: str,
        was_corrected: bool,
        skip_reason: str | None = None,
    ) -> DispatchDecision:
        """Log a verify pass as an aggregate dispatch decision.

        Symmetric to record_ensemble for the sequential mode. The
        primary's own dispatch decision was recorded normally
        (under the user's session_id), and the verifier's call has
        its own decision under the ephemeral _verify_<...> session
        (accessible via /dispatch/recent?include_ephemeral=1). This
        entry is the operator-visible marker that links them.

        Pass ``verifier_id=None`` with a ``skip_reason`` (e.g.
        ``"no alternate worker"``) to log a "verify skipped" entry —
        symmetric to the ensemble-skipped path. The user opted in
        to two-worker verification; if no alternate is available we
        still want the operator to see the request was made.
        """
        if verifier_id is None:
            reason_text = skip_reason or "no alternate worker available"
            notes = f"verify: skipped ({reason_text})"
            candidates = 1
        else:
            notes = (
                f"verify: primary={primary_id} verifier={verifier_id} "
                f"{'corrected' if was_corrected else 'confirmed'}"
            )
            candidates = 2
        decision = DispatchDecision(
            worker=None,
            intent="task",
            reason=REASON_VERIFY,
            notes=notes,
            session_id=session_id,
            previous_worker_id=primary_id,
            candidates_considered=candidates,
        )
        self._record(decision)
        return decision

    def record_retry(
        self,
        session_id: str,
        retry_worker: "WorkerInfo",
        original_worker_id: str,
        intent: str,
        cause: str = "empty_text",
    ) -> DispatchDecision:
        """Log a coordinator-side retry as its own dispatch decision.

        The retry path (commit 534e40f) didn't go through
        ``async_select_for_session`` so it never showed up in
        ``/dispatch/recent``. Operators looking at the log only saw
        the primary worker's failed dispatch and couldn't tell that a
        retry on a different worker followed. This method records the
        retry as a separate decision with ``reason=retry_empty_text``
        and ``previous_worker_id=<primary>`` so the log entry is
        self-describing.

        ``cause`` distinguishes WHY the retry happened so the notes
        line accurately reflects the failure mode. Operators reading
        /dispatch/recent for a timed-out request previously saw
        "retry after empty response" — misleading; the primary
        actually hit the worker_inference_timeout (default 300s),
        not an empty-text fallback. Pass ``cause="primary_failed"``
        (or any free-text description, e.g. "primary_failed: timeout
        after 60s") so the dispatch entry surfaces the real cause.
        """
        if cause == "empty_text":
            note_phrase = f"retry after empty response from {original_worker_id}"
        elif cause == "primary_failed":
            note_phrase = f"retry after {original_worker_id} failed"
        else:
            # Free-text cause (e.g. "primary_failed: timeout after 60s").
            note_phrase = f"retry after {original_worker_id}: {cause}"
        decision = DispatchDecision(
            worker=retry_worker,
            intent=intent,
            reason=REASON_RETRY_EMPTY,
            notes=note_phrase,
            session_id=session_id,
            previous_worker_id=original_worker_id,
            candidates_considered=1,
        )
        self._record(decision)
        return decision

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

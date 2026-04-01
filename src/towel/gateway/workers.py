"""Worker registry for controller-managed remote Towel runtimes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from websockets.asyncio.server import ServerConnection

log = logging.getLogger("towel.gateway.workers")


@dataclass
class WorkerInfo:
    """A connected remote worker."""

    id: str
    ws: ServerConnection
    capabilities: dict[str, Any] = field(default_factory=dict)
    busy: bool = False
    enabled: bool = True
    draining: bool = False
    current_job_id: str | None = None
    current_session_id: str | None = None
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))

    def touch(self, capabilities: dict[str, Any] | None = None) -> None:
        """Refresh heartbeat metadata."""
        self.last_seen = datetime.now(UTC)
        if capabilities:
            self.capabilities.update(capabilities)

    def to_dict(self) -> dict[str, Any]:
        """Serialize worker state for APIs."""
        return {
            "id": self.id,
            "capabilities": dict(self.capabilities),
            "busy": self.busy,
            "enabled": self.enabled,
            "draining": self.draining,
            "current_job_id": self.current_job_id,
            "current_session_id": self.current_session_id,
            "last_seen": self.last_seen.isoformat(),
        }


class WorkerRegistry:
    """Tracks connected workers and simple job assignment."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}

    def register(
        self,
        worker_id: str,
        ws: ServerConnection,
        capabilities: dict[str, Any] | None = None,
    ) -> WorkerInfo:
        worker = WorkerInfo(id=worker_id, ws=ws, capabilities=capabilities or {})
        self._workers[worker_id] = worker
        return worker

    def unregister(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    def heartbeat(self, worker_id: str, capabilities: dict[str, Any] | None = None) -> None:
        worker = self._workers.get(worker_id)
        if worker:
            worker.touch(capabilities)

    def get(self, worker_id: str) -> WorkerInfo | None:
        return self._workers.get(worker_id)

    def set_enabled(self, worker_id: str, enabled: bool) -> bool:
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        worker.enabled = enabled
        worker.touch()
        return True

    def set_draining(self, worker_id: str, draining: bool) -> bool:
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        worker.draining = draining
        worker.touch()
        return True

    def list(self) -> list[WorkerInfo]:
        return list(self._workers.values())

    def apply_state(
        self,
        worker_id: str,
        *,
        enabled: bool | None = None,
        draining: bool | None = None,
    ) -> bool:
        """Apply persisted or API-driven operational state to a worker."""
        worker = self._workers.get(worker_id)
        if not worker:
            return False
        if enabled is not None:
            worker.enabled = enabled
        if draining is not None:
            worker.draining = draining
        worker.touch()
        return True

    def state_snapshot(self) -> dict[str, dict[str, bool]]:
        """Return persistent operational state for all known workers."""
        return {
            worker_id: {
                "enabled": worker.enabled,
                "draining": worker.draining,
            }
            for worker_id, worker in self._workers.items()
        }

    def _score_worker(
        self,
        worker: WorkerInfo,
        requirements: dict[str, Any] | None = None,
        node_tracker: Any | None = None,
    ) -> tuple[int, datetime, str]:
        """Rank workers by capability fit, context pressure, then recency.

        When a NodeTracker is provided, context-aware signals are folded in:
        - Workers with lower context pressure get a bonus
        - Workers that can't fit the estimated conversation size get penalized
        - Workers already holding the session's context get a locality bonus
        """
        score = 0
        req = requirements or {}
        caps = worker.capabilities

        required_backend = req.get("backend")
        if required_backend:
            if caps.get("backend") == required_backend:
                score += 40
            else:
                score -= 100

        required_mode = req.get("mode")
        if required_mode:
            supported_modes = caps.get("modes") or []
            if required_mode in supported_modes:
                score += 30
            else:
                score -= 100

        required_model = req.get("model")
        if required_model:
            worker_model = str(caps.get("model", ""))
            if worker_model == required_model:
                score += 20
            elif worker_model.split("/")[-1] == required_model.split("/")[-1]:
                score += 10

        required_tools = req.get("tools")
        if required_tools is not None:
            if bool(caps.get("tools")) == bool(required_tools):
                score += 5
            elif required_tools:
                score -= 50

        # ── Context-aware scoring ───────────────────────────────────
        if node_tracker is not None:
            node = node_tracker.get(worker.id)
            if node is not None:
                # Bonus for low context pressure (0-15 points)
                pressure = node.context_pressure
                score += int((1.0 - pressure) * 15)

                # Penalty if the conversation won't fit
                estimated_tokens = req.get("estimated_tokens", 0)
                if estimated_tokens > 0 and not node.can_fit_conversation(estimated_tokens):
                    score -= 60

                # Context locality bonus: if this worker already has the
                # session's context loaded, prefer it to avoid cold transfer
                target_session = req.get("session_id")
                if target_session and node.get_context_slot(target_session) is not None:
                    score += 25

        return (score, worker.last_seen, worker.id)

    def acquire(
        self,
        preferred_id: str | None = None,
        requirements: dict[str, Any] | None = None,
        node_tracker: Any | None = None,
    ) -> WorkerInfo | None:
        """Return the best idle worker, preferring the sticky one when it fits.

        When node_tracker is provided, scoring includes context pressure,
        capacity checks, and context locality bonuses.
        """
        if preferred_id:
            preferred = self._workers.get(preferred_id)
            if preferred and preferred.enabled and not preferred.draining and not preferred.busy:
                if requirements is None or self._score_worker(preferred, requirements, node_tracker)[0] >= 0:
                    return preferred

        idle = [
            worker
            for worker in self._workers.values()
            if worker.enabled and not worker.draining and not worker.busy
        ]
        if not idle:
            return None

        if requirements:
            ranked = sorted(
                idle,
                key=lambda worker: self._score_worker(worker, requirements, node_tracker),
                reverse=True,
            )
            best = ranked[0]
            if self._score_worker(best, requirements, node_tracker)[0] < 0:
                return None
            return best

        if node_tracker is not None:
            # Even without specific requirements, prefer least-loaded nodes
            idle.sort(
                key=lambda w: self._score_worker(w, node_tracker=node_tracker),
                reverse=True,
            )
            return idle[0]

        idle.sort(key=lambda worker: (worker.last_seen, worker.id), reverse=True)
        return idle[0]

    def matching(self, requirements: dict[str, Any] | None = None) -> list[WorkerInfo]:
        """Return workers ordered by capability fit."""
        workers = list(self._workers.values())
        if requirements is None:
            return sorted(workers, key=lambda worker: (worker.last_seen, worker.id), reverse=True)
        return sorted(
            workers,
            key=lambda worker: self._score_worker(worker, requirements),
            reverse=True,
        )

    def assign(self, worker_id: str, job_id: str, session_id: str) -> None:
        worker = self._workers[worker_id]
        worker.busy = True
        worker.current_job_id = job_id
        worker.current_session_id = session_id
        worker.touch()

    def release(self, worker_id: str) -> None:
        worker = self._workers.get(worker_id)
        if worker:
            worker.busy = False
            worker.current_job_id = None
            worker.current_session_id = None
            worker.touch()

    def stats(self) -> dict[str, int]:
        total = len(self._workers)
        busy = sum(1 for worker in self._workers.values() if worker.busy)
        enabled = sum(1 for worker in self._workers.values() if worker.enabled)
        draining = sum(1 for worker in self._workers.values() if worker.draining)
        disabled = total - enabled
        idle = sum(
            1
            for worker in self._workers.values()
            if worker.enabled and not worker.draining and not worker.busy
        )
        return {
            "total": total,
            "busy": busy,
            "idle": idle,
            "enabled": enabled,
            "draining": draining,
            "disabled": disabled,
        }

    def __len__(self) -> int:
        return len(self._workers)

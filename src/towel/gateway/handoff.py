"""Graceful context handoff for LAN cluster worker transitions.

When a worker drains, disconnects, or becomes overloaded, sessions pinned
to it need to migrate to another node. The handoff manager coordinates this:

  1. Captures the session's current conversation state from the departing worker
  2. Selects a replacement worker using context-aware scheduling
  3. Pre-warms the replacement by syncing the conversation
  4. Updates session affinity to point to the new worker
  5. Notifies the session owner that the handoff occurred

This avoids the cold-start problem where a session suddenly lands on a worker
that has no context and must receive the full conversation from scratch under
time pressure (e.g., the user is waiting for a response).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

log = logging.getLogger("towel.gateway.handoff")


class HandoffReason(Enum):
    """Why a session is being handed off."""
    WORKER_DRAINING = "worker_draining"
    WORKER_DISCONNECTED = "worker_disconnected"
    WORKER_OVERLOADED = "worker_overloaded"
    MANUAL_REBALANCE = "manual_rebalance"
    CAPACITY_EXCEEDED = "capacity_exceeded"


@dataclass
class HandoffRecord:
    """Records a completed or in-progress handoff for auditing."""
    session_id: str
    from_worker_id: str
    to_worker_id: str
    reason: HandoffReason
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    success: bool = False
    message_count: int = 0
    tokens_transferred: int = 0
    error: str | None = None

    def complete(self, success: bool, error: str | None = None) -> None:
        self.completed_at = datetime.now(UTC)
        self.success = success
        self.error = error

    @property
    def duration_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "from_worker_id": self.from_worker_id,
            "to_worker_id": self.to_worker_id,
            "reason": self.reason.value,
            "started_at": self.started_at.isoformat(),
            "success": self.success,
            "message_count": self.message_count,
            "tokens_transferred": self.tokens_transferred,
        }
        if self.completed_at:
            d["completed_at"] = self.completed_at.isoformat()
            d["duration_ms"] = self.duration_ms
        if self.error:
            d["error"] = self.error
        return d


class HandoffManager:
    """Manages graceful session handoffs between workers in the cluster.

    Works with the WorkerRegistry, NodeTracker, and ContextSyncManager
    to orchestrate smooth transitions. The GatewayServer calls into this
    when workers drain or disconnect.
    """

    def __init__(self, max_history: int = 100) -> None:
        self._history: list[HandoffRecord] = []
        self._max_history = max_history
        self._pending: dict[str, HandoffRecord] = {}  # session_id -> active handoff

    def plan_handoff(
        self,
        session_id: str,
        from_worker_id: str,
        reason: HandoffReason,
        conversation_messages: int = 0,
        estimated_tokens: int = 0,
    ) -> HandoffRecord:
        """Create a handoff plan (before selecting the target worker)."""
        record = HandoffRecord(
            session_id=session_id,
            from_worker_id=from_worker_id,
            to_worker_id="",  # Will be set when target is chosen
            reason=reason,
            message_count=conversation_messages,
            tokens_transferred=estimated_tokens,
        )
        self._pending[session_id] = record
        return record

    def assign_target(self, session_id: str, to_worker_id: str) -> HandoffRecord | None:
        """Set the target worker for a pending handoff."""
        record = self._pending.get(session_id)
        if record:
            record.to_worker_id = to_worker_id
        return record

    def complete_handoff(
        self,
        session_id: str,
        success: bool,
        error: str | None = None,
    ) -> HandoffRecord | None:
        """Mark a handoff as complete."""
        record = self._pending.pop(session_id, None)
        if not record:
            return None
        record.complete(success, error)
        self._history.append(record)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        if success:
            log.info(
                "Handoff complete: session %s moved from %s to %s (%s, %.0fms)",
                session_id,
                record.from_worker_id,
                record.to_worker_id,
                record.reason.value,
                record.duration_ms or 0,
            )
        else:
            log.warning(
                "Handoff failed: session %s from %s — %s",
                session_id,
                record.from_worker_id,
                error,
            )
        return record

    def sessions_needing_handoff(
        self,
        worker_id: str,
        session_workers: dict[str, str],
    ) -> list[str]:
        """Find sessions assigned to a worker that need to be moved.

        Called when a worker starts draining or disconnects. Returns session
        IDs that were sticky-assigned to this worker.
        """
        return [
            session_id
            for session_id, assigned_worker in session_workers.items()
            if assigned_worker == worker_id and session_id not in self._pending
        ]

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def history(self) -> list[HandoffRecord]:
        return list(self._history)

    def recent_handoffs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent handoff records for monitoring."""
        return [r.to_dict() for r in self._history[-limit:]]

    def stats(self) -> dict[str, Any]:
        total = len(self._history)
        successful = sum(1 for r in self._history if r.success)
        failed = total - successful
        avg_duration = 0.0
        durations = [r.duration_ms for r in self._history if r.duration_ms is not None]
        if durations:
            avg_duration = sum(durations) / len(durations)

        by_reason: dict[str, int] = {}
        for r in self._history:
            by_reason[r.reason.value] = by_reason.get(r.reason.value, 0) + 1

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "pending": len(self._pending),
            "avg_duration_ms": round(avg_duration, 1),
            "by_reason": by_reason,
        }

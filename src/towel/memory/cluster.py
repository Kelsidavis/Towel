"""Shared cluster memory — distributed learning across worker nodes.

While the base MemoryStore is local to each node, ClusterMemorySync
ensures memories propagate across the LAN cluster so all workers share
the same learned facts about the user and their work.

Architecture:
  - The controller runs the authoritative MemoryStore
  - Workers send memory mutations (remember/forget) to the controller
  - The controller broadcasts memory updates to all workers
  - Workers merge incoming updates into their local stores
  - On reconnection, workers receive a full memory snapshot

This is eventually-consistent: workers may briefly have stale memories
after a partition, but the controller's version is always authoritative.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from towel.memory.store import MemoryEntry, MemoryStore

log = logging.getLogger("towel.memory.cluster")


@dataclass
class MemoryMutation:
    """A single memory change to replicate across the cluster."""

    action: str  # "remember" or "forget"
    key: str
    content: str = ""
    memory_type: str = "fact"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    origin_worker_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "key": self.key,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "origin_worker_id": self.origin_worker_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryMutation:
        return cls(
            action=data["action"],
            key=data["key"],
            content=data.get("content", ""),
            memory_type=data.get("memory_type", "fact"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(UTC),
            origin_worker_id=data.get("origin_worker_id", ""),
        )


class ClusterMemorySync:
    """Synchronizes the MemoryStore across controller and workers.

    The controller instantiates this with its authoritative store.
    Workers instantiate it with their local store and a send callback
    to dispatch mutations to the controller.
    """

    def __init__(
        self,
        store: MemoryStore,
        is_controller: bool = True,
    ) -> None:
        self.store = store
        self.is_controller = is_controller
        self._pending_mutations: list[MemoryMutation] = []
        self._version: int = 0

    @property
    def version(self) -> int:
        return self._version

    def remember(
        self,
        key: str,
        content: str,
        memory_type: str = "fact",
        origin_worker_id: str = "",
    ) -> MemoryEntry:
        """Store a memory and record the mutation for cluster sync."""
        entry = self.store.remember(key, content, memory_type)
        mutation = MemoryMutation(
            action="remember",
            key=key,
            content=content,
            memory_type=memory_type,
            origin_worker_id=origin_worker_id,
        )
        self._pending_mutations.append(mutation)
        self._version += 1
        return entry

    def forget(self, key: str, origin_worker_id: str = "") -> bool:
        """Remove a memory and record the mutation."""
        removed = self.store.forget(key)
        if removed:
            mutation = MemoryMutation(
                action="forget",
                key=key,
                origin_worker_id=origin_worker_id,
            )
            self._pending_mutations.append(mutation)
            self._version += 1
        return removed

    def apply_mutation(self, mutation: MemoryMutation) -> bool:
        """Apply an incoming mutation from another node.

        Called on the controller when a worker sends a mutation, or
        on a worker when the controller broadcasts an update.
        """
        if mutation.action == "remember":
            self.store.remember(mutation.key, mutation.content, mutation.memory_type)
            self._version += 1
            log.debug("Applied remote remember: %s from %s", mutation.key, mutation.origin_worker_id)
            return True
        elif mutation.action == "forget":
            removed = self.store.forget(mutation.key)
            if removed:
                self._version += 1
            log.debug("Applied remote forget: %s from %s", mutation.key, mutation.origin_worker_id)
            return removed
        return False

    def apply_mutations(self, mutations: list[dict[str, Any]]) -> int:
        """Apply a batch of mutations. Returns count of successful applications."""
        applied = 0
        for data in mutations:
            mutation = MemoryMutation.from_dict(data)
            if self.apply_mutation(mutation):
                applied += 1
        return applied

    def drain_pending(self) -> list[MemoryMutation]:
        """Get and clear pending mutations for broadcast."""
        mutations = self._pending_mutations
        self._pending_mutations = []
        return mutations

    def snapshot(self) -> dict[str, Any]:
        """Create a full memory snapshot for syncing to a new/reconnecting worker."""
        entries = self.store.recall_all()
        return {
            "version": self._version,
            "memories": {e.key: e.to_dict() for e in entries},
        }

    def apply_snapshot(self, snapshot: dict[str, Any]) -> int:
        """Replace the local store with a full snapshot from the controller.

        Used when a worker first connects or reconnects after a long absence.
        Returns the number of memories loaded.
        """
        memories = snapshot.get("memories", {})
        count = 0
        for key, entry_data in memories.items():
            entry = MemoryEntry.from_dict(entry_data)
            self.store.remember(entry.key, entry.content, entry.memory_type)
            count += 1
        self._version = snapshot.get("version", 0)
        log.info("Applied memory snapshot: %d memories, version %d", count, self._version)
        return count

    def build_sync_message(self, target_worker_id: str = "") -> dict[str, Any]:
        """Build a WebSocket message for memory sync.

        For the controller: includes pending mutations to broadcast.
        For workers: includes pending mutations to send to controller.
        """
        mutations = self.drain_pending()
        if not mutations:
            return {}
        return {
            "type": "memory_sync",
            "mutations": [m.to_dict() for m in mutations if m.origin_worker_id != target_worker_id],
            "version": self._version,
        }

    def build_snapshot_message(self) -> dict[str, Any]:
        """Build a full snapshot message for a newly connected worker."""
        return {
            "type": "memory_snapshot",
            **self.snapshot(),
        }

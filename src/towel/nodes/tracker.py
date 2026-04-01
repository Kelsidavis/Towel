"""Node tracker — maintains the cluster's view of all node capabilities.

The controller runs a NodeTracker alongside the WorkerRegistry. While the
registry handles connection state and job assignment, the tracker maintains
the richer capability picture: hardware resources, context window usage,
and model availability. Workers report their resources on registration and
via heartbeats; the tracker aggregates this into a queryable view.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from towel.nodes.capability import ContextSlot, NodeCapability, NodeResources

log = logging.getLogger("towel.nodes.tracker")


class NodeTracker:
    """Cluster-wide node capability tracker.

    Sits alongside WorkerRegistry on the controller. The registry owns
    connection/job state; this tracker owns the capability/resource view
    that feeds into context-aware scheduling.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NodeCapability] = {}

    def register(self, worker_id: str, capabilities: dict[str, Any]) -> NodeCapability:
        """Register or update a node from its capabilities dict."""
        node = NodeCapability.from_worker_capabilities(worker_id, capabilities)
        existing = self._nodes.get(worker_id)
        if existing:
            # Preserve context slots from the existing record
            node.context_slots = existing.context_slots
        self._nodes[worker_id] = node
        log.info(
            "Node %s registered: backend=%s model=%s ctx=%d vram=%dMB",
            worker_id,
            node.backend,
            node.model,
            node.context_window,
            node.resources.vram_total_mb,
        )
        return node

    def unregister(self, worker_id: str) -> NodeCapability | None:
        """Remove a node when its worker disconnects."""
        return self._nodes.pop(worker_id, None)

    def get(self, worker_id: str) -> NodeCapability | None:
        return self._nodes.get(worker_id)

    def update_resources(self, worker_id: str, resources: dict[str, Any]) -> bool:
        """Update a node's hardware resource snapshot (from heartbeat)."""
        node = self._nodes.get(worker_id)
        if not node:
            return False
        node.resources = NodeResources.from_dict(resources)
        node.updated_at = datetime.now(UTC)
        return True

    def update_heartbeat(self, worker_id: str, capabilities: dict[str, Any]) -> bool:
        """Process a heartbeat update that may include refreshed capabilities."""
        node = self._nodes.get(worker_id)
        if not node:
            return False

        # Update resources if provided
        resources = capabilities.get("resources")
        if resources:
            node.resources = NodeResources.from_dict(resources)

        # Update context window if it changed (model swap, etc)
        new_ctx = capabilities.get("context_window")
        if new_ctx and new_ctx != node.context_window:
            node.context_window = new_ctx

        node.updated_at = datetime.now(UTC)
        return True

    # ── Context slot management ─────────────────────────────────────

    def open_context_slot(
        self,
        worker_id: str,
        session_id: str,
        tokens_used: int = 0,
    ) -> ContextSlot | None:
        """Record that a session has an active context on this node."""
        node = self._nodes.get(worker_id)
        if not node:
            return None
        # Avoid duplicates
        existing = node.get_context_slot(session_id)
        if existing:
            existing.tokens_used = tokens_used
            return existing
        return node.add_context_slot(session_id, tokens_used)

    def close_context_slot(self, worker_id: str, session_id: str) -> bool:
        """Remove a session's context slot from a node."""
        node = self._nodes.get(worker_id)
        if not node:
            return False
        return node.remove_context_slot(session_id)

    def update_context_usage(self, worker_id: str, session_id: str, tokens_used: int) -> bool:
        """Update how many tokens a session's conversation is using."""
        node = self._nodes.get(worker_id)
        if not node:
            return False
        return node.update_context_slot(session_id, tokens_used)

    # ── Querying ────────────────────────────────────────────────────

    def all_nodes(self) -> list[NodeCapability]:
        return list(self._nodes.values())

    def nodes_for_backend(self, backend: str) -> list[NodeCapability]:
        return [n for n in self._nodes.values() if n.backend == backend]

    def nodes_with_capacity(self, min_tokens: int) -> list[NodeCapability]:
        """Nodes that can fit a conversation of at least min_tokens."""
        return [n for n in self._nodes.values() if n.can_fit_conversation(min_tokens)]

    def least_loaded_node(self, backend: str | None = None) -> NodeCapability | None:
        """Find the node with the lowest context pressure.

        Useful for load-balancing new sessions across the cluster.
        """
        candidates = self._nodes.values()
        if backend:
            candidates = [n for n in candidates if n.backend == backend]
        else:
            candidates = list(candidates)
        if not candidates:
            return None
        return min(candidates, key=lambda n: (n.context_pressure, n.active_sessions))

    def cluster_stats(self) -> dict[str, Any]:
        """Aggregate cluster statistics for monitoring."""
        nodes = list(self._nodes.values())
        if not nodes:
            return {
                "total_nodes": 0,
                "total_vram_mb": 0,
                "used_vram_mb": 0,
                "total_context_tokens": 0,
                "active_sessions": 0,
                "avg_context_pressure": 0.0,
            }

        total_vram = sum(n.resources.vram_total_mb for n in nodes)
        used_vram = sum(n.resources.vram_used_mb for n in nodes)
        total_context = sum(n.total_context_tokens_used for n in nodes)
        active_sessions = sum(n.active_sessions for n in nodes)
        pressures = [n.context_pressure for n in nodes]
        avg_pressure = sum(pressures) / len(pressures) if pressures else 0.0

        return {
            "total_nodes": len(nodes),
            "total_vram_mb": total_vram,
            "used_vram_mb": used_vram,
            "total_context_tokens": total_context,
            "active_sessions": active_sessions,
            "avg_context_pressure": round(avg_pressure, 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {wid: node.to_dict() for wid, node in self._nodes.items()},
            "stats": self.cluster_stats(),
        }

    def __len__(self) -> int:
        return len(self._nodes)

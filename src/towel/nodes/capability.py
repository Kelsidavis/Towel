"""Node capability and resource descriptors for LAN cluster scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class NodeResources:
    """Hardware resources available on a node."""

    hostname: str = ""
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    ram_total_mb: int = 0
    ram_used_mb: int = 0
    cpu_count: int = 0

    @property
    def vram_free_mb(self) -> int:
        return max(0, self.vram_total_mb - self.vram_used_mb)

    @property
    def ram_free_mb(self) -> int:
        return max(0, self.ram_total_mb - self.ram_used_mb)

    @property
    def vram_utilization(self) -> float:
        """VRAM usage as a fraction 0.0-1.0."""
        if self.vram_total_mb <= 0:
            return 0.0
        return min(1.0, self.vram_used_mb / self.vram_total_mb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "ram_total_mb": self.ram_total_mb,
            "ram_used_mb": self.ram_used_mb,
            "ram_free_mb": self.ram_free_mb,
            "cpu_count": self.cpu_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeResources:
        return cls(
            hostname=data.get("hostname", ""),
            vram_total_mb=data.get("vram_total_mb", 0),
            vram_used_mb=data.get("vram_used_mb", 0),
            ram_total_mb=data.get("ram_total_mb", 0),
            ram_used_mb=data.get("ram_used_mb", 0),
            cpu_count=data.get("cpu_count", 0),
        )


@dataclass
class ContextSlot:
    """Tracks one active context window on a node."""

    session_id: str
    tokens_used: int = 0
    context_window: int = 0
    model: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def tokens_free(self) -> int:
        return max(0, self.context_window - self.tokens_used)

    @property
    def utilization(self) -> float:
        if self.context_window <= 0:
            return 0.0
        return min(1.0, self.tokens_used / self.context_window)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "tokens_used": self.tokens_used,
            "context_window": self.context_window,
            "model": self.model,
            "utilization": round(self.utilization, 3),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class NodeCapability:
    """Full capability snapshot for a node in the cluster.

    Combines the worker's runtime capabilities (backend, model, modes)
    with hardware resources and active context window tracking. This is
    the complete picture the controller needs for intelligent scheduling.
    """

    worker_id: str
    resources: NodeResources = field(default_factory=NodeResources)
    backend: str = ""
    model: str = ""
    modes: list[str] = field(default_factory=list)
    context_window: int = 0
    max_tokens: int = 0
    tools: bool = False
    context_slots: list[ContextSlot] = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_context_tokens_used(self) -> int:
        """Sum of tokens used across all active context slots."""
        return sum(slot.tokens_used for slot in self.context_slots)

    @property
    def active_sessions(self) -> int:
        return len(self.context_slots)

    @property
    def context_pressure(self) -> float:
        """How much context capacity is consumed (0.0 = empty, 1.0 = full).

        This measures the aggregate context load. A node running many
        near-full conversations has high pressure and should be deprioritized
        for new sessions that might need lots of context space.
        """
        if self.context_window <= 0:
            return 0.0
        # Theoretical max: one full context window per slot. But practically
        # the node can only handle one inference at a time, so we measure
        # relative to a single context window (the bottleneck).
        return min(1.0, self.total_context_tokens_used / self.context_window)

    def add_context_slot(
        self, session_id: str, tokens_used: int = 0, model: str = ""
    ) -> ContextSlot:
        """Register an active context window on this node."""
        slot = ContextSlot(
            session_id=session_id,
            tokens_used=tokens_used,
            context_window=self.context_window,
            model=model or self.model,
        )
        self.context_slots.append(slot)
        return slot

    def remove_context_slot(self, session_id: str) -> bool:
        """Remove a context slot when a session leaves this node."""
        before = len(self.context_slots)
        self.context_slots = [s for s in self.context_slots if s.session_id != session_id]
        return len(self.context_slots) < before

    def update_context_slot(self, session_id: str, tokens_used: int) -> bool:
        """Update token usage for an active session's context slot."""
        for slot in self.context_slots:
            if slot.session_id == session_id:
                slot.tokens_used = tokens_used
                return True
        return False

    def get_context_slot(self, session_id: str) -> ContextSlot | None:
        """Get the context slot for a specific session."""
        for slot in self.context_slots:
            if slot.session_id == session_id:
                return slot
        return None

    def can_fit_conversation(self, estimated_tokens: int) -> bool:
        """Check if this node can accommodate a conversation of the given size."""
        if self.context_window <= 0:
            return True  # Unknown capacity — assume yes
        return estimated_tokens <= self.context_window

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "resources": self.resources.to_dict(),
            "backend": self.backend,
            "model": self.model,
            "modes": self.modes,
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "tools": self.tools,
            "context_pressure": round(self.context_pressure, 3),
            "active_sessions": self.active_sessions,
            "total_context_tokens_used": self.total_context_tokens_used,
            "context_slots": [s.to_dict() for s in self.context_slots],
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_worker_capabilities(cls, worker_id: str, caps: dict[str, Any]) -> NodeCapability:
        """Build a NodeCapability from the flat capabilities dict workers send."""
        resources_data = caps.get("resources", {})
        if not resources_data and caps.get("hostname"):
            resources_data = {"hostname": caps["hostname"]}
        return cls(
            worker_id=worker_id,
            resources=NodeResources.from_dict(resources_data),
            backend=caps.get("backend", ""),
            model=str(caps.get("model", "")),
            modes=caps.get("modes", []),
            context_window=caps.get("context_window", 0),
            max_tokens=caps.get("max_tokens", 0),
            tools=bool(caps.get("tools", False)),
        )

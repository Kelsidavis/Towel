"""Node capability and resource descriptors for LAN cluster scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    """``int(x)`` that doesn't crash on garbage worker-reported input.

    Capability and resource fields flow in from worker self-reports
    over the WS register / heartbeat path. A worker sending
    ``total_vram_mb: "huge"`` (typo, JSON-serialised wrong) would
    otherwise crash ``int("huge")`` with ValueError deep inside the
    dispatcher's gate check or this module's resource-normalisation
    code, and 500 the user's request. Coerce defensively at every
    capability-derived numeric read site.
    """
    if isinstance(value, bool):
        # bool is a subtype of int; preserve `or 0` historical behaviour.
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return default


def resources_from_worker_caps(caps: dict[str, Any]) -> NodeResources:
    """Build a NodeResources from a worker's full capabilities dict.

    Workers report VRAM at the top level of ``caps`` (``total_vram_mb``)
    and inside a ``gpus`` array, NOT inside the ``resources`` sub-dict.
    They also report ``ram_available_mb`` (psutil's convention) rather
    than ``ram_used_mb``. Without this helper:

      - /cluster/nodes showed ``vram_total_mb: 0`` for every worker
        regardless of GPU presence.
      - RAM-used always read 0.
      - The heartbeat path (every 15s) re-applied the buggy
        `NodeResources.from_dict` over the resources sub-dict and
        clobbered any correct value the register path had managed
        to set.

    Used by both ``NodeCapability.from_worker_capabilities`` and the
    NodeTracker heartbeat update so the cluster view stays correct
    across both code paths.
    """
    resources_data = dict(caps.get("resources") or {})
    if not resources_data and caps.get("hostname"):
        resources_data = {"hostname": caps["hostname"]}
    if "vram_total_mb" not in resources_data:
        top_vram = caps.get("total_vram_mb")
        if not isinstance(top_vram, int | float):
            # Includes None (the .get default) and any garbage value
            # a worker happens to report. Fall through to the per-GPU
            # sum below.
            top_vram = None
        if top_vram is None:
            # Older workers report only per-GPU vram_mb in `gpus`.
            gpus = caps.get("gpus") or []
            if isinstance(gpus, list):
                top_vram = sum(
                    _safe_int(g.get("vram_mb"))
                    for g in gpus
                    if isinstance(g, dict)
                )
        if top_vram:
            resources_data["vram_total_mb"] = _safe_int(top_vram)
    # Prefer the fresh `live_resources.ram_available_mb` (refreshed on
    # every 15s heartbeat) over the stale value in `resources` (set
    # once at register). The `resources` sub-dict in caps never
    # updates after startup, so deriving ram_used from it makes the
    # cluster view show register-time usage for the entire session
    # lifetime — useless for the operator question "how loaded is
    # this node right now?".
    live = caps.get("live_resources") or {}
    fresh_avail = live.get("ram_available_mb")
    if fresh_avail is None:
        fresh_avail = resources_data.get("ram_available_mb")
    if (
        "ram_used_mb" not in resources_data
        and "ram_total_mb" in resources_data
        and fresh_avail is not None
    ):
        # Same defensive coercion as the vram path above — a worker
        # reporting ram_total_mb or ram_available_mb as garbage
        # would otherwise crash inside `int("huge")` and 500 the
        # /cluster/nodes render.
        resources_data["ram_used_mb"] = max(
            0,
            _safe_int(resources_data["ram_total_mb"])
            - _safe_int(fresh_avail),
        )
    return NodeResources.from_dict(resources_data)


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
    # External/removable mount points this node can read (e.g. /media/k/drive).
    # Advertised by the worker so the coordinator can route a request that
    # references a path under one of them to the node that actually has it.
    mounts: list[str] = field(default_factory=list)
    context_slots: list[ContextSlot] = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_context_tokens_used(self) -> int:
        """Sum of tokens used across all active context slots.

        Each slot's contribution is capped at this node's
        ``context_window`` on read, even though ``add_context_slot``
        and ``update_context_slot`` also cap on write. Defense in
        depth: a slot can outlive the cap-on-write fix when the
        coordinator was running pre-fix code at slot-creation time
        (the slot persists across worker reconnects via
        NodeTracker.register's `node.context_slots = existing.context_slots`)
        and the bogus token count then permanently pins
        context_pressure to 1.0 — observed live as a stale 250k-token
        slot keeping a 24GB worker looking permanently maxed out
        on an 8k context window. Read-side cap fixes the in-memory
        state without forcing a restart.
        """
        if self.context_window <= 0:
            return sum(slot.tokens_used for slot in self.context_slots)
        return sum(
            min(slot.tokens_used, self.context_window)
            for slot in self.context_slots
        )

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
        """Register an active context window on this node.

        ``tokens_used`` is capped at this node's ``context_window``
        because the worker can't physically load more than that — the
        coordinator-side token estimate is an upper bound on what the
        request needs, not what the worker actually holds. Without the
        cap, a single oversized request (e.g. a 1MB user message
        flowing into the count) drove context_pressure to 1.0 and
        steered the dispatcher away from a worker that was actually
        idle.
        """
        capped = min(tokens_used, self.context_window) if self.context_window > 0 else tokens_used
        slot = ContextSlot(
            session_id=session_id,
            tokens_used=capped,
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
        """Update token usage for an active session's context slot.

        Same context-window cap as ``add_context_slot`` so a wildly
        inflated estimate can't poison context_pressure.
        """
        capped = min(tokens_used, self.context_window) if self.context_window > 0 else tokens_used
        for slot in self.context_slots:
            if slot.session_id == session_id:
                slot.tokens_used = capped
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
            "mounts": self.mounts,
            "context_pressure": round(self.context_pressure, 3),
            "active_sessions": self.active_sessions,
            "total_context_tokens_used": self.total_context_tokens_used,
            "context_slots": [s.to_dict() for s in self.context_slots],
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_worker_capabilities(cls, worker_id: str, caps: dict[str, Any]) -> NodeCapability:
        """Build a NodeCapability from the flat capabilities dict workers send."""
        return cls(
            worker_id=worker_id,
            resources=resources_from_worker_caps(caps),
            backend=caps.get("backend", ""),
            model=str(caps.get("model", "")),
            modes=caps.get("modes", []),
            context_window=caps.get("context_window", 0),
            max_tokens=caps.get("max_tokens", 0),
            tools=bool(caps.get("tools", False)),
            mounts=list(caps.get("mounts", []) or []),
        )

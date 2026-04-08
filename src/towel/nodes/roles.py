"""Automatic role assignment for heterogeneous cluster nodes.

When a worker connects, the coordinator inspects its capabilities —
GPU VRAM, model size, context window, tool support, estimated speed —
and assigns one or more roles that determine how requests get routed.

Roles are not exclusive: a powerful GPU node might be both INFERENCE
and CLASSIFIER. A small node with tools enabled might be TOOL_WORKER
and CLASSIFIER. The scheduler picks the best node for each role when
routing a request.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

log = logging.getLogger("towel.nodes.roles")


class NodeRole(Enum):
    """Roles a cluster node can fulfill."""

    CLASSIFIER = "classifier"      # Fast single-token classification
    TOOL_WORKER = "tool_worker"    # Executes tool calls (URL fetch, search, shell)
    INFERENCE = "inference"        # Primary text generation (big model, quality)
    GENERAL = "general"            # Fallback — can do anything

    def __str__(self) -> str:
        return self.value


# ── Heuristic thresholds ───────────────────────────────────────────

# VRAM thresholds (MB) for inference tier classification
_LARGE_VRAM_MB = 8_000     # 8 GB+ = strong inference node
_MEDIUM_VRAM_MB = 4_000    # 4 GB+ = decent inference

# Model parameter estimates from quantized file sizes (GB → rough params)
_LARGE_MODEL_GB = 4.0      # 4 GB+ on disk = likely 7B+ params
_SMALL_MODEL_GB = 2.0      # <2 GB on disk = likely ≤3B params


def assign_roles(capabilities: dict[str, Any]) -> list[NodeRole]:
    """Assign roles to a node based on its reported capabilities.

    The coordinator calls this when a worker registers or updates its
    heartbeat. Roles are additive — a node gets every role it qualifies for.
    """
    roles: list[NodeRole] = []

    backend = capabilities.get("backend", "")
    has_tools = bool(capabilities.get("tools", False))
    context_window = capabilities.get("context_window", 0)

    # GPU info
    gpus = capabilities.get("gpus", [])
    total_vram = capabilities.get("total_vram_mb", 0)
    if not total_vram and gpus:
        total_vram = sum(g.get("vram_mb", 0) for g in gpus)

    # RAM info (for CPU-only nodes like Pi)
    ram_total = capabilities.get("resources", {}).get("ram_total_mb", 0)

    # ── Inference: can this node do quality text generation? ────────
    # GPU nodes with good VRAM, or large-model CPU nodes
    if total_vram >= _LARGE_VRAM_MB:
        roles.append(NodeRole.INFERENCE)
    elif backend == "claude":
        # Claude API is always a strong inference backend
        roles.append(NodeRole.INFERENCE)
    elif backend in ("mlx", "llama", "ollama"):
        # CPU/Apple Silicon nodes — check if model is substantial enough
        # Use context_window as a proxy for model capability when VRAM is unknown
        if total_vram >= _MEDIUM_VRAM_MB or context_window >= 32768:
            roles.append(NodeRole.INFERENCE)

    # ── Classifier: can this node do fast single-token classification? ──
    # Any node with a running model can classify. Prefer lightweight nodes.
    if backend in ("mlx", "llama", "ollama"):
        roles.append(NodeRole.CLASSIFIER)
    # Claude API can classify too, but it's expensive — only if no local option
    # (the scheduler will prefer local classifiers via cost scoring)
    if backend == "claude":
        roles.append(NodeRole.CLASSIFIER)

    # ── Tool worker: can this node execute tools? ──────────────────
    if has_tools:
        roles.append(NodeRole.TOOL_WORKER)

    # ── Everyone gets GENERAL as a fallback ────────────────────────
    roles.append(NodeRole.GENERAL)

    return roles


def best_node_for_role(
    role: NodeRole,
    nodes: list[dict[str, Any]],
    *,
    exclude_busy: bool = True,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Pick the best node to fulfill a given role.

    Scoring priorities by role:
    - CLASSIFIER: cheapest/fastest node (prefer local over API, small over big)
    - TOOL_WORKER: must have tools=True, prefer low context pressure
    - INFERENCE: biggest model / most VRAM, prefer low context pressure
    - GENERAL: least loaded

    Each node dict must include 'capabilities', 'busy', 'enabled', 'roles'.
    """
    candidates = [
        n for n in nodes
        if role in (n.get("roles") or [])
        and n.get("enabled", True)
        and not n.get("draining", False)
        and (not exclude_busy or not n.get("busy", False))
    ]

    if not candidates:
        return None

    if role == NodeRole.CLASSIFIER:
        # Prefer cheapest: local over API, smallest model, lowest latency
        def classifier_score(n: dict[str, Any]) -> tuple[int, int, int]:
            caps = n.get("capabilities", {})
            # Local backends are free, Claude API costs money
            is_local = 0 if caps.get("backend") != "claude" else 1
            # Smaller VRAM = cheaper/faster for classification
            vram = caps.get("total_vram_mb", 0)
            # Context locality bonus
            locality = 0
            if session_id:
                slots = n.get("context_slots", [])
                if any(s.get("session_id") == session_id for s in slots):
                    locality = -1000  # strong preference
            return (is_local, vram, locality)

        candidates.sort(key=classifier_score)
        return candidates[0]

    if role == NodeRole.TOOL_WORKER:
        # Must have tools, prefer least busy
        def tool_score(n: dict[str, Any]) -> tuple[float, int]:
            pressure = n.get("context_pressure", 0.0)
            sessions = n.get("active_sessions", 0)
            return (pressure, sessions)

        candidates.sort(key=tool_score)
        return candidates[0]

    if role == NodeRole.INFERENCE:
        # Prefer most capable: highest VRAM, biggest context, lowest pressure
        def inference_score(n: dict[str, Any]) -> tuple[int, int, float]:
            caps = n.get("capabilities", {})
            vram = caps.get("total_vram_mb", 0)
            ctx = caps.get("context_window", 0)
            pressure = n.get("context_pressure", 0.0)
            # Negate vram and ctx so sort ascending gives us the biggest first
            return (-vram, -ctx, pressure)

        candidates.sort(key=inference_score)
        return candidates[0]

    # GENERAL — least loaded
    candidates.sort(key=lambda n: (n.get("context_pressure", 0.0), n.get("active_sessions", 0)))
    return candidates[0]


def classify_message_intent(text: str) -> str | None:
    """Quick local heuristic for obvious message types.

    Returns 'chat', 'tool', or None (needs LLM classification).
    Catches the easy cases without burning an inference call.
    """
    stripped = text.strip().lower()

    # Obvious greetings / acknowledgements
    if len(stripped) < 25:
        chat_starters = (
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
            "lol", "haha", "cool", "nice", "great", "sure", "yep", "yes",
            "no", "nah", "bye", "goodbye", "gn", "gm", "sup", "yo",
        )
        for starter in chat_starters:
            if stripped == starter or stripped.startswith(starter + " ") or stripped.startswith(starter + "!"):
                return "chat"

    # Obvious tool requests
    tool_signals = (
        "fetch ", "download ", "curl ", "open http", "go to http",
        "search for ", "search the ", "look up ", "google ",
        "what's at http", "get the url", "visit http",
    )
    for signal in tool_signals:
        if stripped.startswith(signal) or f" {signal}" in stripped:
            return "tool"

    # Contains a URL — likely needs fetching
    if "http://" in stripped or "https://" in stripped:
        return "tool"

    return None  # Needs LLM classification

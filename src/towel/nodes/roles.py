"""Automatic role and task assignment for heterogeneous cluster nodes.

When a worker connects, the coordinator inspects its capabilities —
GPU VRAM, model size, context window, tool support, estimated speed —
and assigns one or more roles that determine how requests get routed.

Roles are not exclusive: a powerful GPU node might be both INFERENCE
and CLASSIFIER. A small node with tools enabled might be TOOL_WORKER
and CLASSIFIER. The scheduler picks the best node for each role when
routing a request.

Tasks are higher-level workloads that map to roles + hardware fitness.
The coordinator auto-assigns tasks based on capabilities but allows
manual override per worker via the fleet UI.
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


class TaskType(Enum):
    """Assignable task types for cluster workers.

    Each task type represents a category of work that can be routed to
    a suitable worker. Tasks map to roles + hardware requirements.
    """

    # ── Code quality ──────────────────────────────────────────────────
    LINT = "lint"                        # Static analysis, style checks
    CODE_REVIEW = "code_review"          # Review diffs, suggest improvements
    REFACTOR = "refactor"               # Code restructuring, cleanup
    TEST_GEN = "test_gen"               # Generate test cases
    TEST_RUN = "test_run"               # Execute test suites (needs shell)
    TYPE_CHECK = "type_check"           # Type analysis, annotation

    # ── Research & analysis ───────────────────────────────────────────
    RESEARCH = "research"               # Web search, read docs, summarize
    SUMMARIZE = "summarize"             # Condense long text/conversations
    EXPLAIN = "explain"                 # Explain code, concepts, errors
    ANALYZE = "analyze"                 # Deep analysis, architecture review

    # ── Generation ────────────────────────────────────────────────────
    GENERATE = "generate"               # Write new code, features
    DRAFT = "draft"                     # Write docs, specs, plans
    TRANSLATE = "translate"             # Convert between languages/formats

    # ── Tool-heavy ────────────────────────────────────────────────────
    FETCH = "fetch"                     # URL fetching, API calls
    SHELL = "shell"                     # Run shell commands, scripts
    FILE_OPS = "file_ops"              # File read/write/search
    GIT_OPS = "git_ops"                # Git operations, history analysis
    BUILD = "build"                     # Compile, build, package

    # ── Orchestration ─────────────────────────────────────────────────
    TRIAGE = "triage"                   # Classify and route incoming requests
    PLAN = "plan"                       # Break tasks into subtasks
    CHAT = "chat"                       # Conversational, lightweight replies

    def __str__(self) -> str:
        return self.value


# ── Task → requirements mapping ──────────────────────────────────────

# Each task type specifies what it needs from a worker node.
# roles: required NodeRole(s) — worker must have at least one
# needs_tools: whether the task requires tool execution
# min_vram_mb: minimum VRAM for good performance (0 = any)
# min_context: minimum context window tokens (0 = any)
# prefer_fast: prefer lowest latency over quality
# prefer_quality: prefer largest model over speed

TASK_REQUIREMENTS: dict[TaskType, dict[str, Any]] = {
    # Code quality — moderate models, tools helpful
    TaskType.LINT:        {"roles": [NodeRole.TOOL_WORKER, NodeRole.GENERAL], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.CODE_REVIEW: {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.REFACTOR:    {"roles": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.TEST_GEN:    {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 4000, "min_context": 16384, "prefer_quality": True},
    TaskType.TEST_RUN:    {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.TYPE_CHECK:  {"roles": [NodeRole.TOOL_WORKER, NodeRole.GENERAL], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},

    # Research & analysis — large context, quality models
    TaskType.RESEARCH:    {"roles": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.SUMMARIZE:   {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 4000, "min_context": 65536, "prefer_quality": True},
    TaskType.EXPLAIN:     {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 2000, "min_context": 16384},
    TaskType.ANALYZE:     {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 8000, "min_context": 32768, "prefer_quality": True},

    # Generation — quality models, decent context
    TaskType.GENERATE:    {"roles": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.DRAFT:       {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 4000, "min_context": 16384, "prefer_quality": True},
    TaskType.TRANSLATE:   {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 2000, "min_context": 16384},

    # Tool-heavy — needs tools, speed matters
    TaskType.FETCH:       {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.SHELL:       {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.FILE_OPS:    {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.GIT_OPS:     {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 16384, "prefer_fast": True},
    TaskType.BUILD:       {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},

    # Orchestration — lightweight, fast
    TaskType.TRIAGE:      {"roles": [NodeRole.CLASSIFIER], "needs_tools": False, "min_vram_mb": 0, "min_context": 4096, "prefer_fast": True},
    TaskType.PLAN:        {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.CHAT:        {"roles": [NodeRole.CLASSIFIER, NodeRole.GENERAL], "needs_tools": False, "min_vram_mb": 0, "min_context": 4096, "prefer_fast": True},
}


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


def assign_tasks(
    capabilities: dict[str, Any],
    roles: list[NodeRole],
) -> list[TaskType]:
    """Auto-assign suitable tasks based on capabilities and roles.

    Called when a worker registers. Returns tasks the worker is well-suited
    for given its hardware, model, and tool support.
    """
    tasks: list[TaskType] = []

    has_tools = bool(capabilities.get("tools", False))
    context_window = capabilities.get("context_window", 0)
    total_vram = capabilities.get("total_vram_mb", 0)
    gpus = capabilities.get("gpus", [])
    if not total_vram and gpus:
        total_vram = sum(g.get("vram_mb", 0) for g in gpus)
    backend = capabilities.get("backend", "")

    for task_type, reqs in TASK_REQUIREMENTS.items():
        # Check role match — worker must have at least one required role
        required_roles = reqs.get("roles", [])
        if not any(r in roles for r in required_roles):
            continue

        # Check tool requirement
        if reqs.get("needs_tools") and not has_tools:
            continue

        # Check minimum VRAM
        min_vram = reqs.get("min_vram_mb", 0)
        if min_vram > 0 and total_vram < min_vram and backend != "claude":
            continue

        # Check minimum context window
        min_ctx = reqs.get("min_context", 0)
        if min_ctx > 0 and context_window < min_ctx and context_window > 0:
            continue

        tasks.append(task_type)

    return tasks


def best_node_for_task(
    task: TaskType,
    nodes: list[dict[str, Any]],
    *,
    exclude_busy: bool = True,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Pick the best node to handle a specific task type.

    Uses task requirements to filter and score candidates.
    Nodes must have the task in their assigned_tasks list
    (supports manual override — only considers explicitly assigned tasks).
    """
    reqs = TASK_REQUIREMENTS.get(task, {})

    candidates = [
        n for n in nodes
        if task in (n.get("assigned_tasks") or [])
        and n.get("enabled", True)
        and not n.get("draining", False)
        and (not exclude_busy or not n.get("busy", False))
    ]

    if not candidates:
        # Fall back to role-based selection
        required_roles = reqs.get("roles", [NodeRole.GENERAL])
        for role in required_roles:
            result = best_node_for_role(role, nodes, exclude_busy=exclude_busy, session_id=session_id)
            if result:
                return result
        return None

    if reqs.get("prefer_fast"):
        # Prefer cheapest: local over API, lowest pressure
        def fast_score(n: dict[str, Any]) -> tuple[int, float, int]:
            caps = n.get("capabilities", {})
            is_api = 1 if caps.get("backend") == "claude" else 0
            pressure = n.get("context_pressure", 0.0)
            locality = 0
            if session_id:
                slots = n.get("context_slots", [])
                if any(s.get("session_id") == session_id for s in slots):
                    locality = -1000
            return (is_api, pressure, locality)

        candidates.sort(key=fast_score)
    elif reqs.get("prefer_quality"):
        # Prefer most capable: biggest model, most VRAM
        def quality_score(n: dict[str, Any]) -> tuple[int, int, float]:
            caps = n.get("capabilities", {})
            vram = caps.get("total_vram_mb", 0)
            ctx = caps.get("context_window", 0)
            pressure = n.get("context_pressure", 0.0)
            return (-vram, -ctx, pressure)

        candidates.sort(key=quality_score)
    else:
        # Default: least loaded
        candidates.sort(key=lambda n: (n.get("context_pressure", 0.0), n.get("active_sessions", 0)))

    return candidates[0]


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


def classify_task_type(text: str) -> TaskType | None:
    """Heuristic task type classification from message text.

    Returns a specific TaskType for obvious cases, None if LLM
    classification is needed. Cheap — no inference cost.
    """
    stripped = text.strip().lower()

    # Chat — greetings, short acks
    if len(stripped) < 25:
        chat_words = (
            "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
            "lol", "haha", "cool", "nice", "great", "sure", "yep", "yes",
            "no", "nah", "bye", "goodbye", "gn", "gm", "sup", "yo",
        )
        for w in chat_words:
            if stripped == w or stripped.startswith(w + " ") or stripped.startswith(w + "!"):
                return TaskType.CHAT

    # Fetch / URL
    if "http://" in stripped or "https://" in stripped:
        return TaskType.FETCH
    for sig in ("fetch ", "download ", "curl ", "get the url"):
        if sig in stripped:
            return TaskType.FETCH

    # Test (before shell — "run tests" should match test, not shell)
    for sig in ("run test", "run the test", "pytest", "unittest", "test suite"):
        if sig in stripped:
            return TaskType.TEST_RUN
    for sig in ("write test", "generate test", "add test", "create test"):
        if sig in stripped:
            return TaskType.TEST_GEN

    # Shell
    for sig in ("run ", "execute ", "shell ", "bash ", "$ "):
        if stripped.startswith(sig):
            return TaskType.SHELL

    # Git
    for sig in ("git ", "commit", "push", "pull request", "diff", "blame", "log "):
        if sig in stripped:
            return TaskType.GIT_OPS

    # Build
    for sig in ("build", "compile", "make ", "cargo ", "npm run", "pip install"):
        if stripped.startswith(sig) or f" {sig}" in stripped:
            return TaskType.BUILD

    # Lint / type check
    for sig in ("lint", "flake8", "ruff ", "eslint", "pylint", "clippy"):
        if sig in stripped:
            return TaskType.LINT
    for sig in ("type check", "typecheck", "mypy", "pyright", "tsc "):
        if sig in stripped:
            return TaskType.TYPE_CHECK

    # Code review
    for sig in ("review ", "code review", "review this", "review the"):
        if sig in stripped:
            return TaskType.CODE_REVIEW

    # Refactor
    for sig in ("refactor", "clean up", "restructure", "reorganize"):
        if sig in stripped:
            return TaskType.REFACTOR

    # Explain
    for sig in ("explain", "what does", "what is", "how does", "why does", "what's this"):
        if sig in stripped:
            return TaskType.EXPLAIN

    # Summarize
    for sig in ("summarize", "summary", "tldr", "tl;dr", "condense"):
        if sig in stripped:
            return TaskType.SUMMARIZE

    # Research
    for sig in ("research", "search for", "look up", "find out", "google", "investigate"):
        if sig in stripped:
            return TaskType.RESEARCH

    # Plan
    for sig in ("plan ", "break down", "outline", "roadmap", "strategy for"):
        if sig in stripped:
            return TaskType.PLAN

    # Draft / docs
    for sig in ("write a doc", "draft ", "write docs", "document ", "readme", "spec "):
        if sig in stripped:
            return TaskType.DRAFT

    # Translate
    for sig in ("translate", "convert to ", "port to ", "rewrite in "):
        if sig in stripped:
            return TaskType.TRANSLATE

    # Analyze
    for sig in ("analyze", "analyse", "audit", "architecture"):
        if sig in stripped:
            return TaskType.ANALYZE

    # Generate (broad — code generation)
    for sig in ("write ", "create ", "implement ", "add ", "generate ", "make a ", "build a "):
        if stripped.startswith(sig):
            return TaskType.GENERATE

    # File ops
    for sig in ("read file", "write file", "find file", "list files", "search file", "grep "):
        if sig in stripped:
            return TaskType.FILE_OPS

    return None

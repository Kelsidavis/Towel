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
import re
from enum import Enum
from typing import Any

from towel.nodes.capability import _safe_int

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
    TaskType.CODE_REVIEW: {"roles": [NodeRole.INFERENCE], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.REFACTOR:    {"roles": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.TEST_GEN:    {"roles": [NodeRole.INFERENCE], "needs_tools": True, "min_vram_mb": 4000, "min_context": 16384, "prefer_quality": True},
    TaskType.TEST_RUN:    {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.TYPE_CHECK:  {"roles": [NodeRole.TOOL_WORKER, NodeRole.GENERAL], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},

    # Research & analysis — large context, quality models
    TaskType.RESEARCH:    {"roles": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.SUMMARIZE:   {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 4000, "min_context": 65536, "prefer_quality": True},
    TaskType.EXPLAIN:     {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 2000, "min_context": 16384, "prefer_quality": True},
    TaskType.ANALYZE:     {"roles": [NodeRole.INFERENCE], "needs_tools": True, "min_vram_mb": 8000, "min_context": 32768, "prefer_quality": True},

    # Generation — quality models, decent context
    TaskType.GENERATE:    {"roles": [NodeRole.INFERENCE, NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.DRAFT:       {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 4000, "min_context": 16384, "prefer_quality": True},
    TaskType.TRANSLATE:   {"roles": [NodeRole.INFERENCE], "needs_tools": False, "min_vram_mb": 2000, "min_context": 16384, "prefer_quality": True},

    # Tool-heavy — needs tools, speed matters
    TaskType.FETCH:       {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.SHELL:       {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.FILE_OPS:    {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},
    TaskType.GIT_OPS:     {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 16384, "prefer_fast": True},
    TaskType.BUILD:       {"roles": [NodeRole.TOOL_WORKER], "needs_tools": True, "min_vram_mb": 0, "min_context": 8192, "prefer_fast": True},

    # Orchestration — lightweight, fast
    TaskType.TRIAGE:      {"roles": [NodeRole.CLASSIFIER], "needs_tools": False, "min_vram_mb": 0, "min_context": 4096, "prefer_fast": True},
    TaskType.PLAN:        {"roles": [NodeRole.INFERENCE], "needs_tools": True, "min_vram_mb": 4000, "min_context": 32768, "prefer_quality": True},
    TaskType.CHAT:        {"roles": [NodeRole.CLASSIFIER, NodeRole.GENERAL], "needs_tools": False, "min_vram_mb": 0, "min_context": 4096, "prefer_fast": True},
}


# Task types whose answers justify the reasoning model's <think> phase. On a
# reasoning model (Qwen3, DeepSeek-R1, …) thinking can add minutes of latency,
# so it's reserved for work where the deliberation pays off (code, analysis,
# planning). Everything else — chat, explain, summarize, tool execution —
# answers directly and stays snappy.
REASONING_TASK_TYPES: frozenset[TaskType] = frozenset(
    {
        TaskType.CODE_REVIEW,
        TaskType.REFACTOR,
        TaskType.TEST_GEN,
        TaskType.ANALYZE,
        TaskType.GENERATE,
        TaskType.PLAN,
        TaskType.RESEARCH,
    }
)


def task_wants_thinking(task_type: TaskType | None) -> bool:
    """True when this task type benefits from the model's <think> reasoning phase.

    Unknown / None defaults to False — the snappy path. A misclassified hard
    task loses some reasoning depth; a misclassified easy one would otherwise
    eat minutes of <think> for nothing, which is the worse failure.
    """
    return task_type in REASONING_TASK_TYPES


def task_needs_tools(task_type: TaskType | None) -> bool:
    """True when this task type should be offered the (large) tool list.

    The full builtin tool payload is ~25k prompt tokens; attaching it to a pure
    chat/explain/summarize turn is wasted prefill the model never uses. Unknown
    / None defaults to True — better to pay the prompt cost than silently strip
    tools from a task that genuinely needed them.
    """
    if task_type is None:
        return True
    return bool(TASK_REQUIREMENTS.get(task_type, {}).get("needs_tools", True))


# ── Heuristic thresholds ───────────────────────────────────────────

# VRAM thresholds (MB) for inference tier classification
_LARGE_VRAM_MB = 8_000     # 8 GB+ = strong inference node
_MEDIUM_VRAM_MB = 4_000    # 4 GB+ = decent inference

# Model parameter estimates from quantized file sizes (GB → rough params)
_LARGE_MODEL_GB = 4.0      # 4 GB+ on disk = likely 7B+ params
_SMALL_MODEL_GB = 2.0      # <2 GB on disk = likely ≤3B params


_POLITE_REQUEST_PREFIXES = (
    "can you ",
    "could you ",
    "would you ",
    "will you ",
    "please ",
    "pls ",
)


def _strip_polite_request_prefix(text: str) -> str:
    """Remove conversational wrappers that hide the actual task verb.

    Without this, short prompts like "can you add tests?" hit the
    trivial yes/no chat heuristic before task matching, so the request
    routes to the no-tool prose path and the model can only say "sure".
    """
    stripped = text.strip().lower()
    changed = True
    while changed:
        changed = False
        for prefix in _POLITE_REQUEST_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
                changed = True
                break
    if stripped.endswith(" please"):
        stripped = stripped[:-7].strip()
    return stripped


def assign_roles(capabilities: dict[str, Any]) -> list[NodeRole]:
    """Assign roles to a node based on its reported capabilities.

    The coordinator calls this when a worker registers or updates its
    heartbeat. Roles are additive — a node gets every role it qualifies for.

    Defensive coercion at the field level: a worker registering with
    a non-list ``gpus`` (string, dict, None) would otherwise crash
    inside the ``sum(g.get(...) ...)`` reduction with AttributeError,
    propagating up through the WS register handler and tearing down
    the connection — the worker reconnects, hits the same crash, and
    loops forever. Same defensive shape the WS handler applies to
    ``capabilities`` itself (non-dict → empty dict) extended down to
    the specific fields that get iterated / dereferenced here.
    """
    roles: list[NodeRole] = []

    backend = capabilities.get("backend", "")
    if not isinstance(backend, str):
        backend = ""
    has_tools = bool(capabilities.get("tools", False))
    context_window = capabilities.get("context_window", 0)
    if not isinstance(context_window, int | float):
        context_window = 0

    # GPU info
    gpus = capabilities.get("gpus", [])
    if not isinstance(gpus, list):
        gpus = []
    # Each entry should be a dict with vram_mb; non-dicts (string,
    # int) would crash on `.get()`. Filter at the iteration boundary.
    gpus = [g for g in gpus if isinstance(g, dict)]
    total_vram = capabilities.get("total_vram_mb", 0)
    if not isinstance(total_vram, int | float):
        total_vram = 0
    if not total_vram and gpus:
        total_vram = sum(g.get("vram_mb", 0) for g in gpus)

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

    Same defensive coercion as ``assign_roles`` — a malformed
    capabilities field would otherwise crash and tear down the
    worker's WS connection from the coordinator side. See the
    ``assign_roles`` docstring for the full reconnect-loop scenario.
    """
    tasks: list[TaskType] = []

    has_tools = bool(capabilities.get("tools", False))
    context_window = capabilities.get("context_window", 0)
    if not isinstance(context_window, int | float):
        context_window = 0
    total_vram = capabilities.get("total_vram_mb", 0)
    if not isinstance(total_vram, int | float):
        total_vram = 0
    gpus = capabilities.get("gpus", [])
    if not isinstance(gpus, list):
        gpus = []
    gpus = [g for g in gpus if isinstance(g, dict)]
    if not total_vram and gpus:
        total_vram = sum(g.get("vram_mb", 0) for g in gpus)
    backend = capabilities.get("backend", "")
    if not isinstance(backend, str):
        backend = ""

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


def worker_quality_tier(capabilities: dict[str, Any]) -> str:
    """Bucket a worker into ``high`` / ``medium`` / ``low`` for at-a-glance UX.

    The tier is derived from the same signals the dispatcher uses for
    per-task gating, so a worker that's labelled ``low`` won't surprise
    anyone by being filtered out of a CODE_REVIEW. Rules:

    - ``high``:   ≥8 GB VRAM, or ≥64k context, or backend is ``claude``
                  (Anthropic API is always strong inference)
    - ``medium``: ≥4 GB VRAM, or ≥32k context
    - ``low``:    everything else (small CPU-only nodes, tiny models)

    Workers that don't advertise VRAM or context default to ``low``.
    """
    vram = _safe_int(capabilities.get("total_vram_mb"))
    ctx = _safe_int(capabilities.get("context_window"))
    backend = capabilities.get("backend") or ""
    if vram >= _LARGE_VRAM_MB or ctx >= 65536 or backend == "claude":
        return "high"
    if vram >= _MEDIUM_VRAM_MB or ctx >= 32768:
        return "medium"
    return "low"


def node_meets_task_requirements(node: dict[str, Any], task: TaskType) -> bool:
    """Return True iff ``node`` meets the declared minimums for ``task``.

    Checked against ``capabilities.total_vram_mb`` and ``capabilities.context_window``.
    Missing values are treated as zero — i.e. workers that don't advertise their
    VRAM/context are assumed to be on the low end and will fail any task with a
    non-zero requirement. This is the "filter out the 3B model" gate: a worker
    running a small fast model must opt-in to a quality task by advertising
    that it's actually capable.
    """
    reqs = TASK_REQUIREMENTS.get(task, {})
    caps = node.get("capabilities") or {}
    min_vram = _safe_int(reqs.get("min_vram_mb"))
    min_ctx = _safe_int(reqs.get("min_context"))
    have_vram = _safe_int(caps.get("total_vram_mb"))
    have_ctx = _safe_int(caps.get("context_window"))
    if min_vram and have_vram < min_vram:
        return False
    if min_ctx and have_ctx < min_ctx:
        return False
    return True


def best_node_for_task(
    task: TaskType,
    nodes: list[dict[str, Any]],
    *,
    exclude_busy: bool = True,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Pick the best node to handle a specific task type.

    Filters by declared task requirements (``min_vram_mb`` / ``min_context``)
    first, then by enabled/draining/busy state, then sorts by the task's
    quality/speed preference. If no worker meets the requirements we fall
    back to the full candidate set rather than refuse — the coordinator is
    expected to be adaptable to the fleet it has. Callers can detect
    degradation via :func:`node_meets_task_requirements` on the returned
    node and surface it in their decision log.

    Nodes must have the task in their assigned_tasks list
    (supports manual override — only considers explicitly assigned tasks).
    """
    reqs = TASK_REQUIREMENTS.get(task, {})

    base_candidates = [
        n for n in nodes
        if task in (n.get("assigned_tasks") or [])
        and n.get("enabled", True)
        and not n.get("draining", False)
        and (not exclude_busy or not n.get("busy", False))
    ]

    # Prefer workers that actually meet the declared minimums.
    qualified = [n for n in base_candidates if node_meets_task_requirements(n, task)]
    if qualified:
        candidates = qualified
    else:
        candidates = base_candidates
        if base_candidates:
            log.warning(
                "No worker meets the declared requirements for %s; falling back to "
                "%d under-spec candidate(s). Quality may degrade.",
                task,
                len(base_candidates),
            )

    if not candidates:
        # Fall back to role-based selection
        required_roles = reqs.get("roles", [NodeRole.GENERAL])
        for role in required_roles:
            result = best_node_for_role(role, nodes, exclude_busy=exclude_busy, session_id=session_id)
            if result:
                return result
        return None

    if reqs.get("prefer_fast"):
        # Prefer cheapest: local over API, lowest pressure, session
        # affinity, and — critically — SMALLER models. Without the
        # vram tiebreak, a fleet with both a 2B and a 27B worker
        # routes chat queries to whichever happened to be first in
        # the worker dict, which observable tracing shows is the 27B.
        # For prefer_fast tasks (CHAT, TRIAGE, LINT, etc.) a smaller
        # model is essentially always better — the cost of being
        # wrong is "a slightly worse 1-line answer" not "a wrong
        # multi-step refactor".
        def fast_score(n: dict[str, Any]) -> tuple[int, float, int, int]:
            caps = n.get("capabilities", {})
            is_api = 1 if caps.get("backend") == "claude" else 0
            pressure = n.get("context_pressure", 0.0)
            locality = 0
            if session_id:
                slots = n.get("context_slots", [])
                if any(s.get("session_id") == session_id for s in slots):
                    locality = -1000
            # Smaller VRAM ≈ smaller model ≈ faster for chat-sized
            # generations. Workers without a vram estimate sort to
            # the end via a large default so we don't accidentally
            # prefer "unknown size" over "known 2B". `_safe_int`
            # keeps a worker that reports vram as a non-numeric
            # string from crashing the sort and 500-ing the route.
            vram = (
                _safe_int(caps.get("total_vram_mb"))
                or _safe_int(caps.get("vram_mb"))
                or 1_000_000
            )
            return (is_api, pressure, locality, vram)

        candidates.sort(key=fast_score)
    elif reqs.get("prefer_quality"):
        # Prefer most capable: biggest model, most VRAM
        def quality_score(n: dict[str, Any]) -> tuple[int, int, float]:
            caps = n.get("capabilities", {})
            # Same defensive coercion as fast_score — `-vram` would
            # raise TypeError on a string vram and tear down the
            # dispatch sort.
            vram = _safe_int(caps.get("total_vram_mb"))
            ctx = _safe_int(caps.get("context_window"))
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
            vram = _safe_int(caps.get("total_vram_mb"))
            ctx = _safe_int(caps.get("context_window"))
            pressure = n.get("context_pressure", 0.0)
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
    task_text = _strip_polite_request_prefix(stripped)
    if task_text != stripped:
        task_type = classify_task_type(task_text)
        if task_type is not None and task_type != TaskType.CHAT:
            reqs = TASK_REQUIREMENTS.get(task_type, {})
            return "tool" if reqs.get("needs_tools") else "task"

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

    # Short trivial questions also belong on the chat path — same
    # signal classify_task_type uses, lifted here so callers that
    # only care about intent don't have to call into task-type
    # land just to find out "this is a small conversational ask".
    # Same URL/fetch-verb exclusions as classify_task_type so an
    # accidental "can you fetch http://..." gets the tool path.
    has_url = "http://" in stripped or "https://" in stripped
    fetch_verbs = any(
        sig in stripped for sig in ("fetch ", "download ", "curl ", "get the url")
    )
    if len(stripped) < 60 and not has_url and not fetch_verbs:
        trivial_question_starts = (
            "what's", "whats", "what is",
            "who's", "whos", "who is",
        )
        for sig in trivial_question_starts:
            if stripped.startswith(sig + " ") or stripped.startswith(sig + "?"):
                return "chat"
        # Bare single-word yes/no heads — same shape as the task-type
        # heuristic so intent and task-type agree on "is python typed?".
        bare_heads = ("is", "are", "do", "does", "can", "should", "did")
        first_word = stripped.split(" ", 1)[0].rstrip("?")
        if first_word in bare_heads:
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
    task_text = _strip_polite_request_prefix(stripped)
    if task_text != stripped:
        task_type = classify_task_type(task_text)
        if task_type is not None and task_type != TaskType.CHAT:
            return task_type

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

    # Trivial Q&A — short conversational questions that the
    # heaviest worker doesn't need to handle. "what's 2+2?",
    # "is python typed?", "do you sleep?" all belong on the fast
    # path. Length cap keeps long explanatory prompts on the
    # quality path even when they start with these question
    # words. The contraction "what's" is the dominant miss the
    # original heuristic had — it only checked "what is" with
    # the space, so "what's the time?" silently got routed as
    # an EXPLAIN task and burned 100+ seconds on a 27B model.
    #
    # Exclusion: a URL anywhere in the message is a fetch
    # signal — let the FETCH heuristic below claim it. Same for
    # explicit "fetch"/"download" verbs which would otherwise
    # get swallowed by "can you ...".
    has_url = "http://" in stripped or "https://" in stripped
    fetch_verbs = any(
        sig in stripped for sig in ("fetch ", "download ", "curl ", "get the url")
    )
    if len(stripped) < 60 and not has_url and not fetch_verbs:
        # Multi-word starters (most specific first — "what is" before
        # "is" so a single greedy prefix match works).
        trivial_starts = (
            "what's", "whats", "what is",
            "who's", "whos", "who is",
            "are these", "are those",
            "does it", "does this",
            "should i", "should we",
            "how many", "how much", "how old",
            "when is", "when's", "where is", "where's",
            "why is", "why's", "why does",
        )
        for sig in trivial_starts:
            if stripped.startswith(sig + " ") or stripped.startswith(sig + "?"):
                return TaskType.CHAT
        # Bare yes/no question heads — "is python typed?", "are you
        # sure", "do you know". Single-word triggers that miss when
        # spelled out as "is python" but match "is it" earlier.
        # Capture both forms by checking the single word + word boundary.
        bare_question_heads = ("is", "are", "do", "does", "can", "should", "did")
        first_word = stripped.split(" ", 1)[0].rstrip("?")
        if first_word in bare_question_heads:
            return TaskType.CHAT
        # Pure arithmetic: "2+2", "3 * 7 = ?", "what's 5 plus 4"
        # — these get pushed onto a heavy model out of all
        # proportion to the cost of answering.
        arithmetic = ("plus", "minus", "times", "divided by") + ("+", "-", "*", "/")
        # Length already capped < 60. Require either a digit OR
        # one of the arithmetic operator words so we don't match
        # general prose that incidentally contains "+".
        if any(c.isdigit() for c in stripped):
            for sig in arithmetic:
                if sig in stripped:
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

    # Build (compile / package — NOT "build a game", which is generation).
    # The bare "build" keyword used to match "build a tic-tac-toe" and
    # silently routed the entire orchestration onto the smaller
    # prefer_fast worker because BUILD is prefer_fast. Require either
    # a programming-specific build verb (compile, make, cargo, …) or
    # an unambiguous "build the project / target / artifact" phrase.
    for sig in ("compile", "make ", "cargo ", "npm run", "pip install"):
        if stripped.startswith(sig) or f" {sig}" in stripped:
            return TaskType.BUILD
    for sig in (
        "build the project", "build the package", "build the target",
        "build the artifact", "build the docker", "build script",
    ):
        if sig in stripped:
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
    for sig in ("refactor", "fix ", "clean up", "restructure", "reorganize"):
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
    doc_edit_starts = ("update ", "modify ", "change ", "add ", "create ", "write ")
    for sig in ("write a doc", "draft ", "write docs", "document ", "readme", "spec "):
        if sig in stripped:
            if sig == "readme" and stripped.startswith(doc_edit_starts):
                continue
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
    for sig in (
        "write ", "create ", "implement ", "add ", "generate ",
        "update ", "modify ", "change ", "make a ", "build a ",
    ):
        if stripped.startswith(sig):
            return TaskType.GENERATE

    # File ops
    for sig in ("read file", "write file", "find file", "list files", "search file", "grep "):
        if sig in stripped:
            return TaskType.FILE_OPS

    return None


# Absolute paths mentioned in a request — used to route by data locality.
# Quotes/backticks/parens are stripped by the surrounding non-path delimiters,
# so this catches `/media/k/drive/x`, "/mnt/data", and bare /srv/app paths.
_ABS_PATH_RE = re.compile(r"/(?:[\w.\-]+/)*[\w.\-]+")


def extract_paths(text: str) -> list[str]:
    """Return absolute filesystem paths mentioned in ``text``."""
    if not text:
        return []
    return _ABS_PATH_RE.findall(text)


def resolve_mount_owners(text: str, mount_owners: dict[str, set[str]]) -> set[str]:
    """Worker ids that own a mount containing a path mentioned in ``text``.

    Matches each referenced path against the advertised mount points and returns
    the owners of the *longest* matching mount (most specific wins). Empty when
    no mounted path is referenced — callers then fall back to normal scheduling.
    """
    if not mount_owners:
        return set()
    paths = extract_paths(text)
    if not paths:
        return set()
    best_len = -1
    owners: set[str] = set()
    for path in paths:
        for mount, workers in mount_owners.items():
            prefix = mount.rstrip("/")
            if path == prefix or path.startswith(prefix + "/"):
                if len(prefix) > best_len:
                    best_len = len(prefix)
                    owners = set(workers)
                elif len(prefix) == best_len:
                    owners |= workers
    return owners

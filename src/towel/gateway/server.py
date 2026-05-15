"""Gateway server — the central nervous system of Towel.

Handles WebSocket connections from channels, nodes, and the web UI.
Routes messages to the agent runtime and streams responses back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import uvicorn
import websockets
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from websockets.asyncio.server import Server, ServerConnection

from towel.agent.context import count_tokens_fallback
from towel.agent.conversation import Role
from towel.agent.events import AgentEvent
from towel.agent.runtime import (
    MAX_TOOL_ITERATIONS,
    AgentRuntime,
    format_tool_feedback,
    tool_result_is_error,
)
from towel.agent.tool_parser import parse_tool_calls
from towel.config import TowelConfig
from towel.gateway.context_sync import ContextSyncManager
from towel.gateway.dispatcher import REASON_NO_WORKERS, Dispatcher
from towel.gateway.handoff import HandoffManager, HandoffReason
from towel.gateway.idle_tasks import IDLE_TASK_PROMPTS, IdleTaskManager
from towel.gateway.sessions import SessionManager
from towel.gateway.workers import WorkerInfo, WorkerRegistry
from towel.memory.cluster import ClusterMemorySync
from towel.memory.store import MemoryStore
from towel.nodes.roles import (
    NodeRole,
    TaskType,
    assign_roles,
    assign_tasks,
    best_node_for_role,
    best_node_for_task,
    classify_message_intent,
    classify_task_type,
    worker_quality_tier,
)
from towel.nodes.tracker import NodeTracker
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore

log = logging.getLogger("towel.gateway")


def _stripped_str(value: Any, default: str = "") -> str:
    """Return ``value.strip()`` when ``value`` is a string, else ``default``.

    Used by request handlers to defensively coerce a body field that's
    supposed to be a string — the previous
    ``(body.get(X) or "").strip()`` pattern crashed on a truthy
    non-string (e.g. an integer or list), surfacing as
    "Internal Server Error" HTTP 500 instead of a clean 400.
    """
    if isinstance(value, str):
        return value.strip()
    return default


def _err_str(exc: BaseException) -> str:
    """Return a non-empty error string for any exception.

    Several stdlib exceptions stringify to empty: ``asyncio.TimeoutError``,
    ``asyncio.CancelledError``, ``KeyError("")``, etc. Returning
    ``{"error": ""}`` from a 500 handler gives the caller nothing to
    work with. This helper falls back to the type name when ``str(exc)``
    is empty.
    """
    s = str(exc)
    return s if s else type(exc).__name__


# Allowlist of WS message types the dispatch loop knows how to
# handle. Lives at module scope so the function-local-uppercase
# (N806) lint complains nothing, and the set is built once at
# import rather than on every WS message.
_WS_KNOWN_TYPES: frozenset[str] = frozenset({
    "register", "heartbeat", "memory_sync",
    "job_event", "job_done", "job_error",
    "cancel", "message",
})


def _memory_entry_dict(entry: Any) -> dict[str, Any]:
    """Render a MemoryEntry with consistent optional fields filled in.

    ``MemoryEntry.to_dict`` conditionally omits ``tags``, ``source``,
    ``scope``, and ``last_recalled_at`` when empty / None — sensible
    for on-disk JSON (keeps files small) but confusing for API
    consumers who have to special-case ``if "tags" in data``.

    Apply defaults at the response boundary so every memory-entry
    payload has the same keys.
    """
    d = entry.to_dict()
    d.setdefault("tags", [])
    d.setdefault("source", "")
    d.setdefault("scope", "")
    d.setdefault("last_recalled_at", None)
    return d


def _answers_in_near_consensus(
    contributions: list[dict[str, Any]],
    threshold: float = 0.7,
) -> bool:
    """Return True iff every pair of answers shares ≥ threshold of
    tokens (Jaccard similarity on lowercased word sets).

    Used by ensemble arbitration to skip the synthesis call when the
    workers basically agreed — synthesis takes ~30s of local-agent
    compute and adds no value when the answers were already close.

    Conservative threshold (0.7): real worker outputs phrase things
    differently even for the same factual answer. Only fires on
    near-identical responses (short factual answers, definitions,
    yes/no with reasoning).
    """
    import re

    def _tokens(text: str) -> set[str]:
        # Word-ish tokens, lowercased. Strip very short tokens —
        # function words ("a", "in") aren't a useful agreement signal.
        return {
            t for t in re.findall(r"\w+", text.lower())
            if len(t) >= 3
        }

    sets = [_tokens(c["answer"]) for c in contributions]
    # Empty sets mean no overlap is possible — treat as not-consensus.
    if any(not s for s in sets):
        return False
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = a | b
            if not union:
                return False
            similarity = len(a & b) / len(union)
            if similarity < threshold:
                return False
    return True


def _guess_model_param_b(name: str) -> float | None:
    """Pull a rough parameter count out of a model name.

    Recognises common patterns like ``Llama-3.3-70B-Instruct-4bit``,
    ``qwen3.6:27b``, ``Phi-3.5-mini-3.8B``. Returns the float (in billions)
    or ``None`` when nothing matches — the caller treats unknown as "we
    can't tell, don't reject the worker."
    """
    import re as _re

    if not name:
        return None
    # Match an integer or decimal followed by a B (case-insensitive). The
    # b must NOT be followed by another letter — "4bit" is a quant tag,
    # not a parameter count, so we reject "4b" in that context with a
    # negative-lookahead.
    for match in _re.finditer(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", name):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        # 0.x and very small numbers are probably "4bit" / "8bit" quant
        # tags, not param counts. Param-count Bs are >= 0.5 in practice.
        if value >= 0.5:
            return value
    return None


@dataclass
class GatewayServer:
    """WebSocket + HTTP gateway."""

    config: TowelConfig
    agent: AgentRuntime
    sessions: SessionManager = field(
        default_factory=lambda: SessionManager(store=ConversationStore())
    )
    pin_store: SessionPinStore = field(default_factory=SessionPinStore)
    worker_state_store: WorkerStateStore = field(default_factory=WorkerStateStore)
    _ws_server: Server | None = None
    _connections: dict[str, ServerConnection] = field(default_factory=dict)
    _active_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _workers: WorkerRegistry = field(default_factory=WorkerRegistry)
    _node_tracker: NodeTracker = field(default_factory=NodeTracker)
    _context_sync: ContextSyncManager = field(default_factory=ContextSyncManager)
    _handoff_manager: HandoffManager = field(default_factory=HandoffManager)
    _cluster_memory: ClusterMemorySync | None = None
    _job_queues: dict[str, asyncio.Queue[dict[str, Any]]] = field(default_factory=dict)
    _session_workers: dict[str, str] = field(default_factory=dict)
    _session_pins: dict[str, str] = field(default_factory=dict)
    _session_jobs: dict[str, str] = field(default_factory=dict)
    _worker_states: dict[str, dict[str, bool]] = field(default_factory=dict)
    _node_roles: dict[str, list[NodeRole]] = field(default_factory=dict)
    _node_tasks: dict[str, list[TaskType]] = field(default_factory=dict)
    # Operator-set overrides that survive worker reconnect. Without this, a
    # brief network blip wipes whatever the operator configured via the
    # fleet panel and the auto-assigned defaults take over.
    _manual_tasks: dict[str, list[TaskType]] = field(default_factory=dict)
    _idle_manager: IdleTaskManager = field(default_factory=IdleTaskManager)
    _dispatcher: Dispatcher | None = None

    def __post_init__(self) -> None:
        self._session_pins = self.pin_store.load()
        self._worker_states = self.worker_state_store.load()
        # Hydrate manual task overrides from disk so they survive a
        # coordinator restart, not just a worker reconnect. Unknown task
        # values are silently skipped — schema may have evolved.
        for worker_id, state in self._worker_states.items():
            raw_tasks = state.get("tasks") if isinstance(state, dict) else None
            if not isinstance(raw_tasks, list):
                continue
            restored: list[TaskType] = []
            for name in raw_tasks:
                try:
                    restored.append(TaskType(name))
                except ValueError:
                    continue
            if restored:
                self._manual_tasks[worker_id] = restored
        # Build the dispatcher last so it can capture references to the
        # already-initialised registries and dicts.
        self._dispatcher = Dispatcher(
            workers=self._workers,
            node_dicts_builder=self._build_node_dicts,
            session_workers=self._session_workers,
            session_pins=self._session_pins,
            node_tracker=self._node_tracker,
            idle_task_predicate=self._idle_manager.is_idle_task,
            preempt_hook=self._preempt_idle_task,
            history_size=int(getattr(self.config, "dispatch_history_size", 500) or 500),
        )
        # Initialize cluster memory from agent's memory store if available
        memory_store = getattr(self.agent, "memory", None)
        if isinstance(memory_store, MemoryStore):
            self._cluster_memory = ClusterMemorySync(memory_store, is_controller=True)

    async def start(self) -> None:
        """Start the gateway (WebSocket + HTTP), advertise via mDNS."""
        import os as _os
        import signal as _signal
        import sys as _sys

        def _handle_sighup(*_: Any) -> None:
            """Re-exec on SIGHUP — reload config and restart cleanly."""
            log.info("SIGHUP received — restarting...")
            _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

        try:
            _signal.signal(_signal.SIGHUP, _handle_sighup)
        except (OSError, ValueError):
            pass  # SIGHUP not available on Windows

        gw = self.config.gateway

        # Start WebSocket server
        self._ws_server = await websockets.serve(
            self._handle_ws,
            gw.host,
            gw.port,
        )
        log.info(f"WebSocket listening on ws://{gw.host}:{gw.port}")

        # Advertise via mDNS so workers can discover us
        self._mdns_advertiser = None
        try:
            from towel.gateway.mdns import TowelServiceAdvertiser

            self._mdns_advertiser = TowelServiceAdvertiser(
                port=gw.port,
                advertise_ip=getattr(self.config, "mdns_advertise_ip", "") or "",
            )
            await self._mdns_advertiser.start()
        except Exception as exc:
            log.warning("mDNS advertisement failed (workers can still connect manually): %s", exc)

        # Start HTTP API on port+1
        http_app = self._build_http_app()
        http_config = uvicorn.Config(
            http_app,
            host=gw.host,
            port=gw.port + 1,
            log_level="warning",
        )
        http_server = uvicorn.Server(http_config)
        log.info(f"HTTP API listening on http://{gw.host}:{gw.port + 1}")

        reaper = asyncio.create_task(self._reap_stale_workers())
        idle_sweeper = asyncio.create_task(self._sweep_idle_results())
        try:
            await http_server.serve()
        finally:
            reaper.cancel()
            idle_sweeper.cancel()
            if self._mdns_advertiser:
                await self._mdns_advertiser.stop()

    async def _handle_ws(self, ws: ServerConnection) -> None:
        """Handle an incoming WebSocket connection."""
        conn_id: str | None = None
        # Track which session_ids THIS connection started streaming
        # tasks for. The outer finally only cancels these — without
        # this scoping, a channel client A disconnecting would have
        # cancelled channel client B's in-flight streaming task too
        # because `_active_tasks` is a coordinator-wide dict keyed
        # by session_id (not by connection).
        my_session_tasks: set[str] = set()
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    # A malformed frame (truncated, non-JSON binary,
                    # whatever) shouldn't kill the entire WebSocket
                    # connection — that would force the worker to
                    # reconnect and re-sync state. Log and skip the
                    # frame; the next valid one runs normally.
                    log.warning(
                        "Ignoring malformed JSON on WS conn_id=%s", conn_id
                    )
                    continue
                if not isinstance(msg, dict):
                    # A JSON array or scalar can't be a message — same
                    # treatment as a malformed frame. Skipping keeps the
                    # connection alive for the next valid one.
                    log.warning(
                        "Ignoring non-object WS frame on conn_id=%s (%s)",
                        conn_id, type(msg).__name__,
                    )
                    continue
                msg_type = msg.get("type", "message")
                # Unknown msg_types silently fell through to the end
                # of the loop — a WS client sending `{"type": "foo"}`
                # got no response and no log entry, making typos
                # ("messsage", "msg") nearly impossible to diagnose.
                # Log at debug so a probing client doesn't flood the
                # log, but the trail is at least there for operators
                # looking for the message that disappeared. The
                # allowlist itself lives at module scope so it's
                # built once at import (and the dispatch loop above
                # keeps using inline per-type handlers — adding a
                # type means an `if` here AND an entry there).
                if msg_type not in _WS_KNOWN_TYPES:
                    log.debug(
                        "WS msg from %s: unknown type %r (ignored)",
                        conn_id, msg_type,
                    )
                    continue

                if msg_type == "register":
                    raw_id = msg.get("id")
                    # Worker IDs flow into HTTP URL paths
                    # (/workers/{id}/state) and JSON responses keyed by
                    # string. A non-string id (int, list, None) would
                    # silently store the worker under a non-string key,
                    # then disappear from /workers because the HTTP
                    # lookup uses the URL string "42" not the integer
                    # 42 the worker registered as. Coerce to a sane
                    # string, fall back to a random handle, and cap
                    # length so a 100KB id can't bloat /workers JSON.
                    if isinstance(raw_id, str) and raw_id:
                        conn_id = raw_id[:256]
                    else:
                        conn_id = ws.id.hex[:12]
                    self._connections[conn_id] = ws
                    role = msg.get("role", "channel")
                    capabilities = msg.get("capabilities", {})
                    # A worker that registers with a non-object
                    # `capabilities` (probing client, buggy worker
                    # build, hand-rolled curl) would crash several
                    # levels deeper in `assign_roles` /
                    # `_node_tracker.register`. Coerce to empty dict
                    # and continue — the worker gets defaults, fleet
                    # state stays consistent.
                    if not isinstance(capabilities, dict):
                        log.warning(
                            "Worker %s registered with non-object capabilities "
                            "(%s); treating as empty",
                            conn_id, type(capabilities).__name__,
                        )
                        capabilities = {}
                    if role == "worker":
                        self._workers.register(conn_id, ws, capabilities)
                        self._node_tracker.register(conn_id, capabilities)
                        # Auto-assign roles and tasks based on hardware capabilities
                        roles = assign_roles(capabilities)
                        self._node_roles[conn_id] = roles
                        # Honour an operator-set task override if one was
                        # configured before this worker disconnected — the
                        # fleet panel's "save tasks" button shouldn't lose
                        # its effect across a transient drop.
                        manual_override = self._manual_tasks.get(conn_id)
                        if manual_override is not None:
                            tasks = manual_override
                            log.info(
                                "Worker %s reconnected — restored manual task "
                                "override: %s",
                                conn_id,
                                ", ".join(str(t) for t in tasks),
                            )
                        else:
                            tasks = assign_tasks(capabilities, roles)
                        self._node_tasks[conn_id] = tasks
                        log.info(
                            "Worker %s assigned roles: %s | tasks: %s",
                            conn_id,
                            ", ".join(str(r) for r in roles),
                            ", ".join(str(t) for t in tasks),
                        )
                        state = self._worker_states.get(conn_id)
                        if state:
                            self._workers.apply_state(
                                conn_id,
                                enabled=state.get("enabled"),
                                draining=state.get("draining"),
                            )
                    await ws.send(
                        json.dumps(
                            {
                                "type": "registered",
                                "id": conn_id,
                                "role": role,
                                "motto": "Don't Panic.",
                            }
                        )
                    )
                    # Send memory snapshot to newly connected worker
                    if role == "worker" and self._cluster_memory:
                        await ws.send(json.dumps(self._cluster_memory.build_snapshot_message()))
                    # Start idle work on newly registered worker
                    if role == "worker":
                        worker_obj = self._workers.get(conn_id)
                        if worker_obj:
                            await self._dispatch_idle_task(worker_obj)
                    continue

                if msg_type == "heartbeat":
                    if conn_id and self._workers.get(conn_id):
                        caps = msg.get("capabilities")
                        # Coerce non-dict capabilities to None so
                        # update_heartbeat doesn't crash deeper.
                        if caps is not None and not isinstance(caps, dict):
                            log.warning(
                                "Heartbeat from %s carried non-object "
                                "capabilities (%s); skipping update",
                                conn_id, type(caps).__name__,
                            )
                            caps = None
                        self._workers.heartbeat(conn_id, caps)
                        if caps:
                            self._node_tracker.update_heartbeat(conn_id, caps)
                    continue

                if msg_type == "memory_sync":
                    # Worker is sending memory mutations to the controller
                    if conn_id and self._cluster_memory:
                        mutations = msg.get("mutations", [])
                        # Empty / non-list mutations are silently
                        # ignored — same effect as no mutations.
                        if not isinstance(mutations, list):
                            log.warning(
                                "memory_sync from %s carried non-list "
                                "mutations (%s); ignoring",
                                conn_id, type(mutations).__name__,
                            )
                            mutations = []
                        self._cluster_memory.apply_mutations(mutations)
                        # Broadcast to other workers
                        await self._broadcast_memory_sync(conn_id)
                    continue

                if msg_type in {"job_event", "job_done", "job_error"}:
                    job_id = msg.get("job_id")
                    queue = self._job_queues.get(job_id or "")
                    if queue:
                        await queue.put(msg)
                    continue

                if msg_type == "cancel":
                    session_id = msg.get("session", "default")
                    self.agent.cancel()
                    # Also cancel the running task if any
                    task = self._active_tasks.pop(session_id, None)
                    if task and not task.done():
                        task.cancel()
                    await self._cancel_remote_job(session_id)
                    log.info(f"Cancelled generation for session {session_id}")
                    continue

                if msg_type == "message":
                    session_id = msg.get("session", "default")
                    # WS clients sending {"session": 42} or {"session":
                    # null} previously crashed in ConversationStore
                    # ._path_for (which iterates the id char-by-char,
                    # not supported on int/None). That TypeError
                    # propagated out of the per-message handler, exited
                    # the read loop, and killed the WebSocket
                    # connection — a single bad client could disconnect
                    # itself. Coerce at the boundary instead.
                    if not isinstance(session_id, str) or not session_id:
                        session_id = "default"
                    content = msg.get("content", "")
                    # Same defensive coercion for content — a non-string
                    # would crash inside conversation.add later.
                    if not isinstance(content, str):
                        content = str(content)
                    channel = msg.get("channel", "unknown")
                    if not isinstance(channel, str):
                        channel = "unknown"
                    stream = bool(msg.get("stream", True))
                    # Opt-in ensemble collaboration on the WS path.
                    # Same semantics as /api/ask?ensemble=true — fan
                    # out to all idle inference workers, synthesize
                    # via local agent. Requires stream=false because
                    # the synthesis step is non-streaming; if a WS
                    # client sets ensemble=true with stream=true we
                    # silently fall back to streaming single-worker
                    # (mirrors the openai-compat reject, but quieter
                    # since the WS protocol doesn't carry HTTP error
                    # codes).
                    ensemble_flag = bool(msg.get("ensemble", False))
                    # Opt-in verify (sequential collaboration). Same
                    # constraint as ensemble: synthesis/review can't
                    # be streamed, so stream=true silently falls
                    # through. Mutually exclusive with ensemble at
                    # the WS layer too — if both are set, ensemble
                    # wins (the more thorough mode).
                    verify_flag = bool(msg.get("verify", False)) and not ensemble_flag
                    # The openai-compat path rejects stream+collab
                    # with 400, but the WS protocol has no error
                    # code path that fits cleanly — log a warning
                    # so operators can see the degradation in
                    # coordinator logs instead of wondering why a
                    # WS client's `ensemble=true` did nothing. The
                    # client still gets a streaming single-worker
                    # response; this just makes the silent fall-
                    # through visible server-side.
                    if stream and (ensemble_flag or verify_flag):
                        log.warning(
                            "WS session %s requested %s with stream=true; "
                            "ignoring collaboration flag (synthesis/review "
                            "is non-streaming). Send stream=false to opt in.",
                            session_id,
                            "ensemble" if ensemble_flag else "verify",
                        )

                    session = self.sessions.get_or_create(session_id)
                    session.conversation.add(Role.USER, content, channel=channel)

                    # ── Role-based dispatch ─────────────────────────────
                    # try/finally so the save runs even on inference
                    # failure. Without it, a brand-new session that
                    # errored before any reply showed up later in
                    # /conversations as if the user had never asked
                    # — the user turn was added in memory but the
                    # save() at the bottom got skipped when the
                    # inference call raised.
                    try:
                        # Ensemble short-circuit (non-streaming only):
                        # fan out, synthesize, send one response event.
                        # Streaming clients fall through to the normal
                        # single-worker path — synthesis can't be
                        # streamed.
                        if ensemble_flag and not stream:
                            arbitrated, _contribs, arb_mode = await self._ensemble_dispatch(
                                session_id, content, user_session=session,
                            )
                            # Aggregate dispatch entry — parity with
                            # /api/ask's record_ensemble call (e00fb6d).
                            # Record unconditionally when the user opted
                            # in: even a no-candidates skip needs to be
                            # visible so operators don't miss the silent
                            # fall-through to single-worker dispatch.
                            if self._dispatcher is not None:
                                try:
                                    self._dispatcher.record_ensemble(
                                        session_id=session_id,
                                        contributions=_contribs,
                                        arbitration_mode=arb_mode,
                                    )
                                except Exception as exc:
                                    log.debug(
                                        "Failed to record WS ensemble dispatch: %s",
                                        exc,
                                    )
                            if arbitrated:
                                from towel.agent.conversation import Message as _M
                                response = _M(
                                    role=Role.ASSISTANT,
                                    content=arbitrated,
                                    metadata={
                                        "ensemble": True,
                                        "ensemble_arbitration": arb_mode,
                                        # Surface the per-worker
                                        # contributions so WS clients
                                        # can show which workers
                                        # answered — parity with
                                        # /api/ask, which has carried
                                        # this field since the
                                        # ensemble feature landed.
                                        # Without it, WS UIs couldn't
                                        # build the same "Workers:
                                        # A, B, C" badge that the
                                        # HTTP path supported.
                                        "ensemble_contributions": _contribs,
                                        "remote_worker": "ensemble",
                                    },
                                )
                                session.conversation.messages.append(response)
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": "response",
                                            "session": session_id,
                                            "content": response.content,
                                            "metadata": response.metadata,
                                        }
                                    )
                                )
                                self._maybe_set_auto_title(session)
                                continue
                            # Ensemble produced nothing useful — fall
                            # through to the normal single-worker
                            # path so the user isn't left empty-handed.
                        worker, intent = await self._route_by_role(content, session_id)
                        if stream:
                            if worker:
                                task = asyncio.create_task(
                                    self._stream_remote_inference(
                                        ws, session_id, session, worker
                                    )
                                )
                            else:
                                task = asyncio.create_task(
                                    self._stream_response(ws, session_id, session)
                                )
                            self._active_tasks[session_id] = task
                            my_session_tasks.add(session_id)
                            try:
                                await task
                            except asyncio.CancelledError:
                                await ws.send(
                                    json.dumps(
                                        {
                                            "type": "cancelled",
                                            "session": session_id,
                                            "content": "",
                                            "metadata": {"reason": "user_cancelled"},
                                        }
                                    )
                                )
                            finally:
                                self._active_tasks.pop(session_id, None)
                                my_session_tasks.discard(session_id)
                        else:
                            if worker and intent == "chat":
                                response = await self._quick_remote_infer(
                                    session_id, session, worker, max_tokens=256
                                )
                            elif worker:
                                response = await self._step_remote_inference(
                                    session_id, session, worker
                                )
                            else:
                                response = await self.agent.step(session.conversation)
                                session.conversation.messages.append(response)
                            # Verify pass: opt-in second-worker review.
                            # Same shape as /api/ask?verify=true (see
                            # commits a315ea1, dc48d36). Skipped when
                            # we have no remote worker (local-agent
                            # path), when the answer is empty, or when
                            # no alternate worker exists.
                            if (
                                verify_flag
                                and worker is not None
                                and response.content
                                and not (response.metadata or {}).get("empty_text_fallback")
                            ):
                                final, was_corrected, verifier_id = (
                                    await self._verify_pass(
                                        session_id, content,
                                        response.content, worker.id,
                                    )
                                )
                                # Aggregate dispatch entry — parity
                                # with /api/ask's record_verify call
                                # (e00fb6d). Always record when verify
                                # was opted in, including the no-alt
                                # skipped case (verifier_id=None).
                                if self._dispatcher is not None:
                                    try:
                                        self._dispatcher.record_verify(
                                            session_id=session_id,
                                            verifier_id=verifier_id,
                                            primary_id=worker.id,
                                            was_corrected=was_corrected,
                                        )
                                    except Exception as exc:
                                        log.debug(
                                            "Failed to record WS verify dispatch: %s",
                                            exc,
                                        )
                                if was_corrected and final != response.content:
                                    log.info(
                                        "WS verifier %s corrected %s answer for session %s",
                                        verifier_id, worker.id, session_id,
                                    )
                                    if (
                                        session.conversation.messages
                                        and session.conversation.messages[-1].role
                                        == Role.ASSISTANT
                                    ):
                                        session.conversation.messages[-1].content = final
                                    response.content = final
                                    response.metadata = (response.metadata or {}) | {
                                        "verified_by": verifier_id,
                                        "verifier_corrected": True,
                                        "primary_worker": worker.id,
                                    }
                                elif verifier_id is not None:
                                    response.metadata = (response.metadata or {}) | {
                                        "verified_by": verifier_id,
                                        "verifier_corrected": False,
                                        "primary_worker": worker.id,
                                    }
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "response",
                                        "session": session_id,
                                        "content": response.content,
                                        "metadata": response.metadata,
                                    }
                                )
                            )

                        # Auto-title after first exchange.
                        self._maybe_set_auto_title(session)
                    finally:
                        # Persist after each exchange — including
                        # partial state on inference failure.
                        try:
                            self.sessions.save(session_id)
                        except Exception as save_exc:
                            log.debug(
                                "Failed to persist session %s: %s",
                                session_id, save_exc,
                            )

        except websockets.ConnectionClosed:
            pass
        finally:
            if conn_id and conn_id in self._connections:
                del self._connections[conn_id]
            if conn_id:
                # Wake any coordinator coroutine waiting on this worker's
                # in-flight job. Without this, classify/quick-infer/run paths
                # block on their queue.get() until the per-call timeout
                # (5s-300s) before they realise the worker is gone.
                self._notify_in_flight_disconnect(conn_id)

                # Trigger handoffs for sessions on this worker before unregistering
                sessions_to_handoff = self._handoff_manager.sessions_needing_handoff(
                    conn_id, self._session_workers
                )
                for sid in sessions_to_handoff:
                    self._handoff_manager.plan_handoff(
                        sid, conn_id, HandoffReason.WORKER_DISCONNECTED
                    )
                    # Clear affinity so the dispatcher picks a new one.
                    self._session_workers.pop(sid, None)
                    # Complete handoff — next request will land on a new worker
                    self._handoff_manager.complete_handoff(sid, success=True)

                self._workers.unregister(conn_id)
                self._node_tracker.unregister(conn_id)
                self._node_roles.pop(conn_id, None)
                self._node_tasks.pop(conn_id, None)
                self._context_sync.clear_worker(conn_id)
            # Cancel only the streaming tasks THIS connection started.
            # Iterating self._active_tasks blindly would cancel tasks
            # belonging to other clients' WS connections too because
            # the dict is coordinator-wide (keyed by session_id, not
            # connection).
            for sid in my_session_tasks:
                task = self._active_tasks.get(sid)
                if task is not None and not task.done():
                    task.cancel()

    def _notify_in_flight_disconnect(self, worker_id: str) -> None:
        """Push a synthetic ``job_error`` into the queue for any in-flight job.

        Coordinator coroutines that sent a job to ``worker_id`` and are blocked
        on ``queue.get()`` would otherwise wait for their per-call timeout
        before noticing the worker died. We resolve them immediately with an
        explicit error so the user sees a fast fallback (classify defaults to
        ``"task"``, infer paths re-route) rather than a stall.
        """
        worker = self._workers.get(worker_id)
        if worker is None:
            return
        job_id = worker.current_job_id
        if not job_id:
            return
        queue = self._job_queues.get(job_id)
        if queue is None:
            return
        try:
            queue.put_nowait(
                {
                    "type": "job_error",
                    "job_id": job_id,
                    "worker_id": worker_id,
                    "error": "Worker disconnected mid-job",
                }
            )
        except asyncio.QueueFull:
            # Queues here are unbounded (no maxsize), but be defensive — a
            # bounded queue in a future refactor shouldn't break this path.
            log.warning(
                "Could not notify waiter for job %s about worker %s disconnect",
                job_id,
                worker_id,
            )

    async def _reap_stale_workers(self, interval: float = 30.0, timeout: float = 60.0) -> None:
        """Periodically close connections to workers that missed heartbeats."""
        while True:
            await asyncio.sleep(interval)
            for worker in self._workers.stale(timeout):
                log.warning("Reaping stale worker %s (no heartbeat for %.0fs)", worker.id, timeout)
                try:
                    await worker.ws.close(1001, "heartbeat timeout")
                except Exception:
                    pass

    async def _sweep_idle_results(self, interval: float = 300.0) -> None:
        """Evict expired idle-task results so the cache doesn't accumulate.

        ``IdleTaskManager`` already drops stale entries on read, but readers
        only fire when a UI page loads or a worker finishes a task. Without
        this sweeper a long-idle coordinator would still hold onto results
        far past their TTL. Every 5 minutes is plenty — the TTLs themselves
        sit between 5 min and 2 h.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                removed = self._idle_manager.purge_expired()
                if removed:
                    log.info("Idle-result sweeper evicted %d stale entries", removed)
            except Exception:
                log.exception("Idle-result sweep failed")

    async def _stream_response(self, ws: ServerConnection, session_id: str, session: Any) -> None:
        """Stream agent response events to the WebSocket."""
        async for event in self.agent.step_streaming(session.conversation):
            await ws.send(json.dumps(event.to_ws_message(session_id)))

    # ── Role-based routing ──────────────────────────────────────────

    def _build_node_dicts(self) -> list[dict[str, Any]]:
        """Build the node descriptor list that the role selector needs."""
        nodes = []
        for worker in self._workers.list():
            node = self._node_tracker.get(worker.id)
            nd: dict[str, Any] = {
                "id": worker.id,
                "capabilities": dict(worker.capabilities),
                "busy": worker.busy,
                "enabled": worker.enabled,
                "draining": worker.draining,
                "roles": self._node_roles.get(worker.id, []),
                "context_pressure": node.context_pressure if node else 0.0,
                "active_sessions": node.active_sessions if node else 0,
                "context_slots": [s.to_dict() for s in node.context_slots] if node else [],
                "assigned_tasks": self._node_tasks.get(worker.id, []),
            }
            nodes.append(nd)
        return nodes

    def _worker_for_task(
        self,
        task: TaskType,
        session_id: str | None = None,
        exclude_busy: bool = True,
    ) -> WorkerInfo | None:
        """Find the best worker for a specific task type."""
        nodes = self._build_node_dicts()
        best = best_node_for_task(task, nodes, exclude_busy=exclude_busy, session_id=session_id)
        if best:
            return self._workers.get(best["id"])
        return None

    def _worker_for_role(
        self,
        role: NodeRole,
        session_id: str | None = None,
        exclude_busy: bool = True,
    ) -> WorkerInfo | None:
        """Find the best worker for a given role."""
        nodes = self._build_node_dicts()
        best = best_node_for_role(
            role, nodes, exclude_busy=exclude_busy, session_id=session_id,
        )
        if best is None:
            return None
        return self._workers.get(best["id"])

    async def _classify_on_worker(self, message: str, worker: WorkerInfo) -> str:
        """Run classification on a remote worker instead of the coordinator.

        Sends a minimal classification prompt and expects 'task', 'chat', or 'tool'.
        Falls back to 'task' on any error.
        """
        # The actual wire payload below builds plain dict messages — no
        # Conversation/Message construction is needed here.
        classify_prompt = (
            "Classify this user message. Reply with exactly one word:\n"
            "- 'chat' if it's casual conversation, greetings, acknowledgements, small talk\n"
            "- 'tool' if it requires fetching URLs, web search, or running commands\n"
            "- 'task' if it's a question, instruction, coding, analysis, or creative request\n\n"
            f"Message: {message[:400]}\n\nClassification:"
        )

        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._workers.assign(worker.id, job_id, "classify")

        # Determine the worker's inference mode from its capabilities.
        # Defensive against a buggy worker registering with `modes:
        # "llama_chat"` (a bare string instead of a list) — modes[0]
        # on a string would return "l" and silently send a garbage
        # mode to the worker. isinstance check skips that.
        modes = worker.capabilities.get("modes")
        mode = modes[0] if isinstance(modes, list) and modes else "llama_chat"

        await worker.ws.send(
            json.dumps(
                {
                    "type": "infer",
                    "job_id": job_id,
                    "session": "classify",
                    "stream": False,
                    "request": {
                        "mode": mode,
                        "model": worker.capabilities.get("model", ""),
                        "system": "",
                        "messages": [{"role": "user", "content": classify_prompt}],
                        "max_tokens": 4,
                        "temperature": 0.0,
                        "reasoning_effort": "none",
                    },
                }
            )
        )

        completed_normally = False
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=5.0)
            if msg.get("type") == "job_done":
                completed_normally = True
                text = msg.get("result", {}).get("text", "").strip().lower()
                for label in ("chat", "tool", "task"):
                    if label in text:
                        return label
                return "task"
            return "task"
        except Exception as exc:
            log.debug("Worker classification failed (%s), defaulting to task", exc)
            return "task"
        finally:
            if not completed_normally:
                # Same cancel-on-early-exit pattern as the inference
                # paths — a 5s timeout that doesn't cancel the worker
                # leaves it generating a classification nobody reads.
                # The classify prompt asks for one word so the waste
                # is small, but the principle is consistent.
                try:
                    await worker.ws.send(
                        json.dumps(
                            {"type": "cancel_job", "job_id": job_id, "session": "classify"}
                        )
                    )
                except Exception as cancel_exc:
                    log.debug(
                        "Failed to send cancel_job (classify) to %s: %s",
                        worker.id, cancel_exc,
                    )
            self._job_queues.pop(job_id, None)
            self._workers.release(worker.id)

    async def _classify_task_on_worker(
        self, message: str, worker: WorkerInfo
    ) -> TaskType | None:
        """Classify a message into a specific TaskType using a remote worker.

        Returns a TaskType or None if classification fails.
        """

        task_labels = ", ".join(f"'{t.value}'" for t in TaskType)
        classify_prompt = (
            "Classify this user message into exactly one task type. "
            f"Reply with exactly one word from: {task_labels}\n\n"
            f"Message: {message[:400]}\n\nTask type:"
        )

        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._workers.assign(worker.id, job_id, "classify_task")

        modes = worker.capabilities.get("modes", [])
        mode = modes[0] if modes else "llama_chat"

        await worker.ws.send(
            json.dumps(
                {
                    "type": "infer",
                    "job_id": job_id,
                    "session": "classify_task",
                    "stream": False,
                    "request": {
                        "mode": mode,
                        "model": worker.capabilities.get("model", ""),
                        "system": "",
                        "messages": [{"role": "user", "content": classify_prompt}],
                        "max_tokens": 8,
                        "temperature": 0.0,
                        "reasoning_effort": "none",
                    },
                }
            )
        )

        completed_normally = False
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=5.0)
            if msg.get("type") == "job_done":
                completed_normally = True
                text = msg.get("result", {}).get("text", "").strip().lower()
                # Match against known task types
                for task_type in TaskType:
                    if task_type.value in text:
                        return task_type
            return None
        except Exception as exc:
            log.debug("Task classification failed (%s)", exc)
            return None
        finally:
            if not completed_normally:
                # Same cancel-on-early-exit pattern (see _classify_on_worker).
                try:
                    await worker.ws.send(
                        json.dumps(
                            {"type": "cancel_job", "job_id": job_id, "session": "classify_task"}
                        )
                    )
                except Exception as cancel_exc:
                    log.debug(
                        "Failed to send cancel_job (classify_task) to %s: %s",
                        worker.id, cancel_exc,
                    )
            self._job_queues.pop(job_id, None)
            self._workers.release(worker.id)

    # Map TaskType to the routing intent that determines the execution path
    _TASK_TO_INTENT: ClassVar[dict[TaskType, str]] = {
        TaskType.CHAT: "chat",
        TaskType.TRIAGE: "chat",
        # Tool-heavy tasks use the full agent loop with tools
        TaskType.LINT: "tool",
        TaskType.TEST_RUN: "tool",
        TaskType.TYPE_CHECK: "tool",
        TaskType.FETCH: "tool",
        TaskType.SHELL: "tool",
        TaskType.FILE_OPS: "tool",
        TaskType.GIT_OPS: "tool",
        TaskType.BUILD: "tool",
        TaskType.RESEARCH: "tool",
        TaskType.REFACTOR: "tool",
        TaskType.GENERATE: "tool",
    }
    # Everything else defaults to "task" (quality inference)

    def _conv_dict_with_memory(self, conversation: Any) -> dict[str, Any]:
        """Build a conversation dict for a remote-worker payload with
        the coordinator's memory injected as a leading system message.

        Memory lives on the COORDINATOR's store; worker runtimes have
        empty local memory. Without this injection any /api/ask or
        /v1/chat/completions question that needs stored context (the
        user's name, a saved preference, prior project notes) loses
        the data on its way to the worker. The recall query is the
        most recent user turn so the fused-search RRF ranks relevant
        entries first.

        Returns the original dict unchanged when the corpus is empty
        — we don't ship a stray empty system message because some
        worker runtimes treat that as an identity override signal.
        """
        conv_dict = conversation.to_dict()
        memory = getattr(self.agent, "memory", None)
        if memory is None:
            return conv_dict
        last_user_msg = next(
            (m.content for m in reversed(conversation.messages)
             if m.role == Role.USER),
            "",
        )
        try:
            memory_block = memory.to_prompt_block(query=last_user_msg)
        except Exception as exc:
            log.warning("memory.to_prompt_block failed: %s", exc)
            return conv_dict
        if not memory_block:
            return conv_dict
        return {
            **conv_dict,
            "messages": [
                {
                    "role": "system",
                    "content": memory_block,
                    "metadata": {"source": "coord_memory_injection"},
                },
                *conv_dict["messages"],
            ],
        }

    async def _verify_pass(
        self,
        session_id: str,
        question: str,
        primary_answer: str,
        primary_worker_id: str,
    ) -> tuple[str, bool, str | None]:
        """Run a second-worker verifier on a (question, answer) pair.

        Picks an alternate worker (not the primary), poses the
        question + answer in a verification prompt, and returns
        ``(final_answer, was_corrected, verifier_worker_id)``.

        - If no alternate is available, returns the primary answer
          unchanged with was_corrected=False, verifier_worker_id=None.
        - If the verifier says "VERIFIED" (case-insensitive, allowing
          trailing punctuation), returns the primary answer with
          was_corrected=False.
        - Otherwise returns the verifier's text as the corrected
          answer with was_corrected=True.

        This is the smallest piece of real multi-worker collaboration:
        two workers acting on a single user request rather than the
        dispatcher routing one request to one worker. Opt-in via
        ``verify=true`` on /api/ask so operators trade latency for
        accuracy only when they want to.
        """
        alt = self._pick_alternate_chat_worker(exclude={primary_worker_id})
        if alt is None:
            return primary_answer, False, None

        # uuid suffix so concurrent verify passes on the same
        # user-facing session_id don't collide on the ephemeral
        # `_verify_{session_id}` key. Same collision class fixed for
        # ensemble. Without this, two concurrent /api/ask?verify=true
        # requests on one session_id would share the verifier's
        # session and corrupt each other's prompts.
        import uuid as _uuid
        verify_sess_id = f"_verify_{session_id}_{_uuid.uuid4().hex[:8]}"

        # Build a short conversation that's just the verification
        # prompt. We don't reuse the user's session — the verifier
        # gets a focused, single-turn ask.
        from towel.agent.conversation import Conversation
        from towel.agent.conversation import Role as _Role

        # Cap the embedded question + primary answer so a user who
        # pasted a 100KB document doesn't push the verifier's prompt
        # past the smallest worker's context window. The verifier
        # judges accuracy from a representative sample — the first
        # ~8KB of each field is plenty to spot factual errors;
        # past that we'd be embedding the workers' irrelevant
        # context overlap anyway.
        verify_field_cap = 8000
        capped_q = (
            question if len(question) <= verify_field_cap
            else question[:verify_field_cap] + "…"
        )
        capped_a = (
            primary_answer if len(primary_answer) <= verify_field_cap
            else primary_answer[:verify_field_cap] + "…"
        )

        verify_conv = Conversation(id=verify_sess_id)
        verify_conv.add(
            _Role.USER,
            (
                "Review this question/answer pair. If the answer is "
                "correct and complete, respond with EXACTLY the word "
                "VERIFIED (uppercase, no other text). If incorrect or "
                "incomplete, respond with the corrected answer only "
                "(no explanation, no preamble).\n\n"
                f"Question: {capped_q}\n\nProposed answer: {capped_a}"
            ),
        )
        # Reuse the gateway's Session class so _quick_remote_infer
        # has the right shape to work with. The Session is ephemeral
        # — never registered in self.sessions.
        from towel.gateway.sessions import Session as _Session

        verify_session = _Session(
            id=verify_sess_id, conversation=verify_conv,
        )
        try:
            verify_response = await self._quick_remote_infer(
                verify_sess_id, verify_session, alt,
                max_tokens=512,
            )
        except Exception as exc:
            log.warning(
                "verify pass on %s failed (%s); keeping primary answer",
                alt.id, exc,
            )
            return primary_answer, False, None
        # Clean up the ephemeral verifier session state — same
        # leak class fixed for OpenAI-compat in fffdc1e.
        try:
            self.cleanup_ephemeral_session(verify_sess_id)
        except Exception:
            pass

        verifier_text = (verify_response.content or "").strip()
        # Confirmation detection. Models routinely add casing /
        # punctuation / leading remarks to the literal VERIFIED token
        # we asked for ("Verified.", "verified", "yes — VERIFIED",
        # "## VERIFIED"). Treat a SHORT response that contains
        # VERIFIED as a confirmation; reserve "long, substantive
        # text" for the corrected-answer branch. Threshold is
        # forgiving but bounded — a 30-char response with VERIFIED
        # in it isn't trying to deliver a corrected answer.
        if (
            verifier_text
            and len(verifier_text) <= 30
            and "VERIFIED" in verifier_text.upper()
        ):
            return primary_answer, False, alt.id
        # If the verifier returned an empty-text fallback placeholder
        # or an obvious failure, don't replace a working answer with
        # nothing.
        if not verifier_text or (
            verify_response.metadata or {}
        ).get("empty_text_fallback"):
            return primary_answer, False, alt.id
        return verifier_text, True, alt.id

    async def _ensemble_dispatch(
        self,
        session_id: str,
        question: str,
        user_session: Any = None,
        identity_override: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], str]:
        """Fan the same prompt to every idle capable worker in parallel.

        When ``user_session`` is provided, each fan-out worker sees
        the full conversation history (cloned into an ephemeral
        session), not just the latest user message — so multi-turn
        questions don't lose context when the operator opts into
        ensemble mode. ``question`` is still the latest user turn
        and is used for the synthesis prompt.

        Returns ``(arbitrated_answer, contributions)`` where
        ``contributions`` is a list of ``{worker_id, answer, ms,
        error}`` dicts — one per worker that responded (or errored).

        Coordinator-side arbitration: when 2+ workers contributed,
        the local agent synthesizes a final answer. Falls back to
        longest-non-empty on synthesis failure.

        Falls back gracefully:
        - 0 workers idle → returns ``("", [])`` and the caller falls
          through to single-worker dispatch
        - 1 worker idle → just that worker's answer (still useful;
          ensemble=true on a tiny fleet still works)
        - all workers timeout/error → returns the longest captured
          error string, or "" if none came back at all
        """
        from datetime import UTC
        from datetime import datetime as _dt

        from towel.agent.conversation import Conversation
        from towel.agent.conversation import Role as _Role
        from towel.gateway.sessions import Session as _Session

        # Build the candidate pool: every enabled, non-draining,
        # non-stuck worker WITH the INFERENCE role assigned. Skip
        # busy workers — fan-out wants real concurrency, not queue
        # thrash. (Operators who want to serialize against busy
        # workers should use verify= instead.)
        #
        # The INFERENCE filter excludes classifier-only workers —
        # those exist for cheap-token tasks (intent / task-type
        # routing) and aren't sized to produce substantive answers.
        # Including them would waste their compute and pollute the
        # arbiter with low-effort responses.
        from towel.nodes.roles import NodeRole as _NodeRole
        now = _dt.now(UTC)
        stuck_threshold_secs = 300.0
        candidates: list[WorkerInfo] = []
        for w in self._workers.list():
            if not w.enabled or w.draining or w.busy:
                continue
            if w.busy_since is not None and (
                (now - w.busy_since).total_seconds() >= stuck_threshold_secs
            ):
                continue
            # Role filter: skip workers built only for cheap-token
            # tasks (classifier-only). A worker with NO role info is
            # treated as eligible — that path covers freshly-registered
            # workers and test fixtures. A worker WITH role info but
            # missing INFERENCE+GENERAL is the classifier-only case
            # we want to exclude.
            worker_roles = self._node_roles.get(w.id)
            if worker_roles is not None and len(worker_roles) > 0:
                if not (
                    _NodeRole.INFERENCE in worker_roles
                    or _NodeRole.GENERAL in worker_roles
                ):
                    continue
            candidates.append(w)

        if not candidates:
            return "", [], "none"

        # uuid suffix so concurrent ensemble runs on the same
        # user-facing session_id don't collide on the ephemeral
        # `_ens_{session_id}_{worker_id}` key. Without this, two
        # concurrent /api/ask?ensemble=true requests on the same
        # session_id would both try to use the same per-worker
        # ephemeral session, interleave their conversations, and
        # corrupt each other's prompts.
        import uuid as _uuid
        run_id = _uuid.uuid4().hex[:8]

        async def _ask_one(worker: WorkerInfo) -> dict[str, Any]:
            """One-shot inference on a fresh ephemeral session."""
            import time as _time
            sess_id = f"_ens_{session_id}_{worker.id}_{run_id}"
            conv = Conversation(id=sess_id)
            # Clone the user's conversation so each worker sees prior
            # turns. Without this, multi-turn questions ("but make it
            # smaller") lost their referent — workers saw only the
            # latest message in isolation. The user's session itself
            # is untouched; we only mutate the ephemeral copy.
            if user_session is not None:
                for m in user_session.conversation.messages:
                    conv.add(m.role, m.content)
            else:
                conv.add(_Role.USER, question)
            session = _Session(id=sess_id, conversation=conv)
            t0 = _time.monotonic()
            try:
                resp = await self._quick_remote_infer(
                    sess_id, session, worker, max_tokens=512,
                    identity_override=identity_override,
                )
                ms = round((_time.monotonic() - t0) * 1000.0, 1)
                meta = resp.metadata or {}
                # An empty-text fallback placeholder is not a real
                # contribution; treat it as a "no answer" so
                # arbitration doesn't pick the placeholder over a
                # real response.
                if meta.get("empty_text_fallback"):
                    return {
                        "worker_id": worker.id, "answer": "",
                        "ms": ms, "error": "empty_text",
                    }
                return {
                    "worker_id": worker.id,
                    "answer": resp.content or "",
                    "ms": ms,
                    "error": None,
                }
            except Exception as exc:
                ms = round((_time.monotonic() - t0) * 1000.0, 1)
                return {
                    "worker_id": worker.id, "answer": "",
                    "ms": ms, "error": _err_str(exc),
                }
            finally:
                try:
                    self.cleanup_ephemeral_session(sess_id)
                except Exception:
                    pass

        # Fire all candidates in parallel with an outer deadline.
        # Each _ask_one already has its own per-worker timeout via
        # _quick_remote_infer (chat_fast_timeout, default 60s) — the
        # outer asyncio.wait deadline is a safety net so one wedged
        # worker can't extend the ensemble run beyond the slowest
        # honest worker. Stragglers past the deadline are recorded as
        # timeout contributions so the operator can see who lagged.
        ensemble_timeout = float(
            getattr(self.config, "chat_fast_timeout", 60.0) or 60.0
        ) * 1.5  # 50% slack over the per-worker bound
        tasks = [asyncio.create_task(_ask_one(w)) for w in candidates]
        try:
            done, pending = await asyncio.wait(tasks, timeout=ensemble_timeout)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        contributions: list[dict[str, Any]] = []
        for t in tasks:
            if t in done:
                try:
                    contributions.append(t.result())
                except Exception as exc:
                    # _ask_one wraps everything but be defensive.
                    contributions.append({
                        "worker_id": "?", "answer": "",
                        "ms": 0.0, "error": _err_str(exc),
                    })
            else:
                # Straggler — cancel and record as timeout.
                t.cancel()
                contributions.append({
                    "worker_id": candidates[tasks.index(t)].id,
                    "answer": "",
                    "ms": round(ensemble_timeout * 1000.0, 1),
                    "error": "ensemble_timeout",
                })

        # Arbitration:
        # - 0 real answers → caller falls through to single-worker dispatch
        # - 1 real answer → return it directly (nothing to arbitrate)
        # - 2+ real answers, near-consensus → skip synthesis, return
        #   the longest (saves ~30s of local-agent work for cases
        #   where the workers basically agreed on the answer)
        # - 2+ real answers, divergent → use the coordinator's local
        #   agent as LLM-as-judge to synthesize. This is the
        #   "coordinator pieces it together" half of the
        #   collaboration model — the workers each contribute an
        #   independent attempt, the coordinator reconciles them.
        real_answers = [c for c in contributions if c["answer"]]
        if not real_answers:
            return "", contributions, "none"
        if len(real_answers) == 1:
            return real_answers[0]["answer"], contributions, "single"
        # Trivial-agreement short-circuit: if every answer is the
        # same string (case-folded, stripped, trailing punctuation
        # removed), there's literally nothing to arbitrate. Catches
        # the case Jaccard misses — very short answers ("42",
        # "Berlin.", "yes") where the tokens-≥3-chars filter strips
        # everything. Trailing punctuation is stripped too so "42."
        # vs "42" don't waste 30s on synthesis when the answers were
        # cosmetically different but factually identical.
        import re as _re_trivial
        def _trivial_form(s: str) -> str:
            return _re_trivial.sub(r"[^\w]+$", "", s.strip()).lower()
        normalized = {_trivial_form(a["answer"]) for a in real_answers}
        if len(normalized) == 1:
            return real_answers[0]["answer"], contributions, "consensus"
        if _answers_in_near_consensus(real_answers):
            # Workers basically agreed — skip the synthesis call.
            # The longest answer usually has the most detail; pick it.
            best = max(real_answers, key=lambda c: len(c["answer"]))
            return best["answer"], contributions, "consensus"

        # Track synthesis timing so the response can surface how long
        # the local-agent arbitration took. Stuffed into each
        # contribution so it travels with the other timing info.
        synth_timing: dict[str, Any] = {}
        synthesized = await self._synthesize_ensemble(
            question, real_answers, timing_sink=synth_timing,
        )
        # Tag every contribution with the synthesis time + timeout
        # flag so the caller has them next to the per-worker `ms`
        # values. The timeout flag is what disambiguates "synthesis
        # ran fast and was useful" from "synthesis bailed and we
        # fell back to longest" — both have a synthesis_ms but only
        # the latter has synthesis_timeout=True.
        if "synthesis_ms" in synth_timing:
            for c in contributions:
                c["synthesis_ms"] = synth_timing["synthesis_ms"]
                if synth_timing.get("synthesis_timeout"):
                    c["synthesis_timeout"] = True
        if synthesized:
            return synthesized, contributions, "synthesis"
        # Synthesis fell through (local agent unavailable, error, or
        # empty output). Don't lose the run — surface the longest
        # contribution as a deterministic fallback.
        best = max(real_answers, key=lambda c: len(c["answer"]))
        return best["answer"], contributions, "longest_fallback"

    async def _synthesize_ensemble(
        self,
        question: str,
        contributions: list[dict[str, Any]],
        timing_sink: dict[str, Any] | None = None,
    ) -> str:
        """Reconcile N worker answers into one via the local agent.

        The coordinator runs its own local model (loaded into memory
        at startup) and is the natural arbiter for ensemble runs —
        it has visibility into every worker's contribution and can
        decide whether they agree, disagree, or partially overlap.

        Returns the synthesized answer text, or "" on failure
        (caller will fall back to a deterministic pick).
        """
        # Shuffle the contributions before labeling. LLM-as-judge has
        # measurable primacy/recency bias — the model tends to favor
        # answers in certain positions regardless of content. Workers
        # arrive in `_ensemble_dispatch` in completion order (fastest
        # first), which means a fast-but-shallow worker would
        # consistently get the privileged "Worker A" slot. Random
        # ordering removes that bias from the arbitration signal.
        import random as _random

        from towel.agent.conversation import Conversation
        from towel.agent.conversation import Role as _Role
        ordered = list(contributions)
        _random.shuffle(ordered)

        # Build a single-turn synthesis prompt. The worker answers
        # are tagged A/B/C/... so the model can refer to them without
        # being primed by raw worker_ids (which encode hardware
        # specs the model shouldn't anchor on).
        labels = "ABCDEFGHIJKLMN"
        # Prompt that biases toward concrete grounded text: prefer
        # specifics over averaging, copy whichever phrasing is more
        # precise rather than rewriting in the arbiter's voice, and
        # keep length proportional to the workers' (don't expand
        # short factual answers into prose). The "do NOT average"
        # line targets a common LLM-as-judge failure mode — blending
        # two specific answers into one vague answer that has neither
        # set of facts.
        lines = [
            "You are arbitrating between answers from multiple AI workers "
            "to a single question. Produce the best final answer for the "
            "user.",
            "",
            "Rules:",
            "1. Where the workers agree, keep that. Where they disagree, "
            "pick the answer with concrete specifics over vague "
            "generalities. Do NOT average — averaging two precise "
            "answers gives one vague answer.",
            "2. If one answer is clearly wrong or incomplete, drop it.",
            "3. Match the workers' format and length. Don't expand "
            "a short factual answer into prose; don't trim a code "
            "block into a one-liner.",
            "4. Output ONLY the final answer — no preamble, no "
            "'Worker A said...' framing, no meta-commentary.",
            "",
            # Cap the embedded question + each worker answer so a
            # user paste of an enormous document can't blow past the
            # local agent's context window. The arbiter judges from
            # a representative sample; if a 64KB-per-field prompt
            # isn't enough to compare answers, the workers already
            # had context overflow problems upstream.
            f"Question: {question[:64000]}{'…' if len(question) > 64000 else ''}",
            "",
        ]
        for i, c in enumerate(ordered[: len(labels)]):
            ans = c["answer"]
            if len(ans) > 64000:
                ans = ans[:64000] + "…"
            lines.append(f"Worker {labels[i]} answered:")
            lines.append(ans)
            lines.append("")
        lines.append("Final answer:")

        synth_conv = Conversation(id="_synth_ensemble")
        synth_conv.add(_Role.USER, "\n".join(lines))
        try:
            # Use generate() instead of step() so a tool-call emitted
            # by the synthesis model doesn't actually run. The
            # synthesizer should be producing TEXT — if the model
            # decides to call e.g. memory.remember on the synthesis
            # context, that's a side-effect bug, not a feature. Strip
            # any tool-call shaped output via parse_tool_calls and
            # keep only the prose.
            import time as _time

            from towel.agent.tool_parser import parse_tool_calls
            # Bound synthesis time so a stuck local agent can't
            # extend an ensemble run forever. 90s is generous —
            # synthesis is short (one focused prompt → one focused
            # answer). Anything past that is the model wedged on the
            # synthesis step itself.
            synth_timeout = float(
                getattr(self.config, "chat_fast_timeout", 60.0) or 60.0
            ) * 1.5
            t0 = _time.monotonic()
            result = await asyncio.wait_for(
                self.agent.generate(synth_conv), timeout=synth_timeout,
            )
            if timing_sink is not None:
                timing_sink["synthesis_ms"] = round(
                    (_time.monotonic() - t0) * 1000.0, 1,
                )
            _tool_calls, remaining_text = parse_tool_calls(result.text)
            return (remaining_text or "").strip()
        except asyncio.TimeoutError:
            log.warning(
                "Ensemble synthesis timed out after %.0fs; "
                "falling back to deterministic pick",
                synth_timeout,
            )
            if timing_sink is not None:
                timing_sink["synthesis_ms"] = round(synth_timeout * 1000.0, 1)
                timing_sink["synthesis_timeout"] = True
            return ""
        except Exception as exc:
            log.warning(
                "Ensemble synthesis on local agent failed (%s); "
                "falling back to deterministic pick",
                exc,
            )
            return ""

    def cleanup_ephemeral_session(self, session_id: str) -> None:
        """Drop in-memory state for a one-shot session.

        OpenAI-compat creates a fresh session_id per /v1/chat/completions
        request (``openai-<random>``) — it's never reused, but every
        call left behind:

        - an entry in ``_session_workers`` (affinity tied to a
          throwaway id)
        - an open context slot on the routed worker

        Both accumulated forever, inflating ``context_pressure`` on
        the workers serving OpenAI traffic. Public so the OpenAI
        compat module (a separate file) can call it after each
        request — both streaming and non-streaming paths.
        """
        worker_id = self._session_workers.pop(session_id, None)
        if worker_id is not None:
            self._node_tracker.close_context_slot(worker_id, session_id)
        # Drop the in-memory Session too — it was never going to be
        # used again, and the SessionManager doesn't persist
        # openai-prefixed sessions (no on-disk store call from
        # openai_compat) so this is purely a memory clean-up.
        self.sessions.remove(session_id)

    def _maybe_set_auto_title(self, session: Any) -> None:
        """Generate and stamp a title on a session that doesn't have one.

        Shared between the WebSocket path and HTTP entry points
        (/api/ask) so api-channel conversations get the same
        auto-titling as chat-channel ones. Previously only the WS
        path titled — every /api/ask session ended up with title=""
        on disk and the conversations list rendered them blank.

        Safe to call repeatedly: only generates when title is empty
        AND there's at least one full exchange to title from.
        """
        if session.conversation.title:
            return
        if len(session.conversation) < 2:
            return
        from towel.agent.titler import generate_title

        first_user = next(
            (
                m.content
                for m in session.conversation.messages
                if m.role == Role.USER
            ),
            "",
        )
        title = generate_title(first_user)
        if title:
            session.conversation.title = title

    def _pick_alternate_chat_worker(
        self, exclude: set[str]
    ) -> WorkerInfo | None:
        """Pick a different worker capable of answering a chat.

        Used by /api/ask and /v1/chat/completions when the routed
        worker returns empty text. Prefers an idle worker, but
        will fall back to a busy non-excluded worker — the WebSocket
        queues serialize requests anyway, and the alternative is
        the diagnostic placeholder, so a slow real answer is
        better than no real answer.

        Picks the largest worker by total_vram_mb so the retry has
        the best shot at a real response. Returns None if no
        qualified alternate exists (all enabled non-draining workers
        are in `exclude`).

        Skips workers that are stuck (busy for more than 5 minutes).
        A stuck worker is by definition not making progress, so
        queuing the retry behind it would just inherit the wedge.
        """
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        stuck_threshold_secs = 300.0
        candidates: list[WorkerInfo] = []
        busy_candidates: list[WorkerInfo] = []
        for w in self._workers.list():
            if w.id in exclude:
                continue
            if not w.enabled or w.draining:
                continue
            if w.busy:
                # Filter out stuck workers — a worker that's been busy
                # for 5+ minutes is wedged on a previous request and
                # won't service the retry any faster than the primary.
                if w.busy_since is not None and (
                    (now - w.busy_since).total_seconds() >= stuck_threshold_secs
                ):
                    continue
                busy_candidates.append(w)
            else:
                candidates.append(w)
        # Prefer idle, but accept busy if that's all we've got. Both
        # buckets sort by VRAM descending so the largest worker wins
        # within its bucket.
        for bucket in (candidates, busy_candidates):
            if bucket:
                bucket.sort(
                    key=lambda w: w.capabilities.get("total_vram_mb", 0),
                    reverse=True,
                )
                return bucket[0]
        return None

    async def _route_by_role(
        self, message: str, session_id: str
    ) -> tuple[WorkerInfo | None, str]:
        """Route a message to the best worker based on task type.

        Classifies the message into an intent + ``TaskType`` and then delegates
        to :class:`towel.gateway.dispatcher.Dispatcher` for the actual worker
        selection (pin → affinity → task-match → role-match → capability
        fallback → idle preempt). Returns ``(worker, intent)``; worker is
        ``None`` when the coordinator should handle the request itself.
        """
        # Step 1: classify into a TaskType (cheap heuristic, then LLM fallback)
        task_type = classify_task_type(message)
        if task_type is None:
            classifier = self._worker_for_role(NodeRole.CLASSIFIER, session_id)
            if classifier:
                task_type = await self._classify_task_on_worker(message, classifier)

        if task_type is not None:
            intent = self._TASK_TO_INTENT.get(task_type, "task")
        else:
            intent = classify_message_intent(message) or "task"

        # Step 2: hand off the selection to the dispatcher.
        assert self._dispatcher is not None
        decision = await self._dispatcher.async_select_for_session(
            session_id,
            intent=intent,
            task_type=task_type,
        )

        if decision.worker is not None:
            # If this session was previously affinitied to a different
            # worker, close the slot on the old worker — otherwise the
            # old slot persists indefinitely on a worker that no longer
            # owns this session. Same leak class as the delete path
            # (see commit 6ca89a0).
            prior = self._session_workers.get(session_id)
            if prior is not None and prior != decision.worker.id:
                self._node_tracker.close_context_slot(prior, session_id)
            self._session_workers[session_id] = decision.worker.id
            return decision.worker, decision.intent

        # No worker → coordinator handles the request locally.
        return None, decision.intent

    async def _quick_remote_infer(
        self,
        session_id: str,
        session: Any,
        worker: WorkerInfo,
        max_tokens: int = 256,
        temperature: float = 0.7,
        identity_override: str | None = None,
    ) -> Any:
        """Lightweight inference on a worker — no tool loop, capped tokens.

        Used for chat/greetings where we want a fast, short response without
        the overhead of the full agent tool loop.
        """
        from towel.agent.conversation import Message

        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._session_jobs[session_id] = job_id
        self._workers.assign(worker.id, job_id, session_id)
        # Coordinator-measured start. Used as a fallback when the
        # worker doesn't report total_ms (older worker code, empty
        # responses, errors). Without this the dispatch log shows
        # "no timing" for any worker that produced 0 tokens, which
        # is exactly the case operators most want to diagnose.
        coord_start = time.monotonic()

        # Send as "infer" with capped tokens — worker uses generate_from_request
        messages = [
            {"role": m.role.value, "content": m.content}
            for m in session.conversation.messages
        ]
        # Determine the worker's inference mode from its capabilities.
        # Defensive against a buggy worker registering with `modes:
        # "llama_chat"` (a bare string instead of a list) — modes[0]
        # on a string would return "l" and silently send a garbage
        # mode to the worker. isinstance check skips that.
        modes = worker.capabilities.get("modes")
        mode = modes[0] if isinstance(modes, list) and modes else "llama_chat"

        # Send a minimal system prompt — empirically, some smaller
        # chat-tuned models (gemma-2B/4B variants observed on the
        # SparklesMint/k-Precision fleet) emit zero tokens when handed
        # a bare yes/no question with no system instruction at all.
        # A one-line directive is plenty to unblock them without
        # adding meaningful tokens to the prompt.
        # Per-request `identity_override` (e.g. /api/ask `system` field)
        # wins over the coordinator's default. Without this parameter
        # the override had to be applied by mutating self.config.identity
        # before the call — which raced badly with concurrent /api/ask
        # requests using different overrides.
        identity = (
            identity_override
            or getattr(self.config, "identity", "")
            or "You are a helpful assistant. Answer concisely."
        )
        # Inject coordinator memory into the worker's system prompt
        # (see _conv_dict_with_memory for the rationale — workers have
        # empty local memory stores so any /api/ask question that
        # needs stored context loses it on the way to the worker).
        # The "infer" payload uses a top-level `system` field rather
        # than embedding the memory in the conversation, so we
        # concatenate onto `identity` here rather than going through
        # the dict helper.
        memory = getattr(self.agent, "memory", None)
        if memory is not None:
            last_user_msg = next(
                (m.content for m in reversed(session.conversation.messages)
                 if m.role == Role.USER),
                "",
            )
            try:
                memory_block = memory.to_prompt_block(query=last_user_msg)
            except Exception as exc:
                log.warning("memory.to_prompt_block failed: %s", exc)
                memory_block = ""
            if memory_block:
                identity = f"{identity}\n\n{memory_block}"
        await worker.ws.send(
            json.dumps(
                {
                    "type": "infer",
                    "job_id": job_id,
                    "session": session_id,
                    "stream": False,
                    "request": {
                        "mode": mode,
                        "model": worker.capabilities.get("model", ""),
                        "system": identity,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "reasoning_effort": "none",
                    },
                }
            )
        )

        chat_fast_timeout = float(
            getattr(self.config, "chat_fast_timeout", 60.0) or 60.0
        )
        # Track whether the worker reached a terminal state on its own.
        # If we exit any other way (timeout, asyncio cancel, Starlette
        # client-disconnect propagating CancelledError), the finally
        # block sends `cancel_job` so the worker stops generating a
        # response nobody will read.
        completed_normally = False
        try:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=chat_fast_timeout)
            except TimeoutError as exc:
                # asyncio.TimeoutError stringifies to "" — bubble up a
                # message the API caller can actually act on. Otherwise
                # simple_ask's generic except returns
                # `{"error": ""}` HTTP 500 with no indication that
                # the worker timed out.
                raise RuntimeError(
                    f"worker {worker.id} did not respond within "
                    f"{chat_fast_timeout:.0f}s"
                ) from exc
            if msg.get("type") == "job_done":
                result = msg.get("result", {}) or {}
                text = result.get("text", "")
                # Defensive: an empty response makes the API look
                # broken. Workers running pre-fix code may emit
                # empty text when the model produces tool calls.
                # Replace with a diagnostic string so the caller
                # sees SOMETHING and operators can spot it. Mark
                # the metadata so the caller (simple_ask) can decide
                # whether to retry on a different worker.
                empty_text_fallback = False
                if not text:
                    empty_text_fallback = True
                    # User-facing placeholder. The previous text
                    # ("the worker returned no text") read like an
                    # operator error message — confusing for end
                    # users who don't know what a worker is and
                    # blamed the model instead of suggesting a path
                    # forward. The retry-on-empty path may replace
                    # this with the alt worker's response; the
                    # operator-facing diagnostic lives in the
                    # `empty_text_fallback: True` metadata flag, not
                    # the prose.
                    text = (
                        "I wasn't able to put together a response that "
                        "turn — try rephrasing or asking again."
                    )
                # Hoist the worker's reported tokens/tps/etc. into the
                # Message metadata so API callers and the UI see real
                # numbers — previously this path discarded everything
                # but the text, leaving tokens=0 tps=0 in responses.
                remote_meta = result.get("metadata", {}) or {}
                if empty_text_fallback:
                    remote_meta = {**remote_meta, "empty_text_fallback": True}
                # Stamp timing onto the dispatch decision so the
                # operator can see both routing + latency in one view.
                # Fall back to coordinator-measured total when the
                # worker didn't report one (older worker code, empty
                # response). Empty-response cases are exactly the
                # ones operators most need timing for.
                coord_total_ms = (time.monotonic() - coord_start) * 1000.0
                if "total_ms" not in remote_meta:
                    remote_meta = {**remote_meta, "total_ms": round(coord_total_ms, 1)}
                if self._dispatcher is not None:
                    decision = self._dispatcher.last_decision_for_session(session_id)
                    if decision is not None:
                        decision.record_completion(
                            ttft_ms=remote_meta.get("ttft_ms"),
                            total_ms=remote_meta.get("total_ms"),
                        )
                response = Message(
                    role=Role.ASSISTANT,
                    content=text,
                    metadata={
                        "remote_worker": worker.id,
                        "quick_infer": True,
                        **remote_meta,
                    },
                )
                session.conversation.messages.append(response)
                completed_normally = True
                return response
            elif msg.get("type") == "job_error":
                # Worker reported failure — it's done on its own;
                # no cancel needed.
                completed_normally = True
                raise RuntimeError(msg.get("message", "Worker failed"))
            # Any other message type isn't a terminal state. Fall
            # through to finally — `completed_normally` stays False
            # and we cancel the job.
        finally:
            if not completed_normally:
                # Stamp the dispatch decision with how long the
                # failing attempt actually took, so operators looking
                # at /dispatch/recent can tell "primary timed out at
                # 60s" apart from "primary errored instantly". Without
                # this the primary decision shows no total_ms when
                # _quick_remote_infer raises.
                if self._dispatcher is not None:
                    decision = self._dispatcher.last_decision_for_session(session_id)
                    if decision is not None and decision.total_ms is None:
                        coord_total_ms = (time.monotonic() - coord_start) * 1000.0
                        decision.record_completion(
                            ttft_ms=None,
                            total_ms=round(coord_total_ms, 1),
                        )
                try:
                    await worker.ws.send(
                        json.dumps(
                            {"type": "cancel_job", "job_id": job_id, "session": session_id}
                        )
                    )
                except Exception as exc:
                    log.debug(
                        "Failed to send cancel_job to %s for %s: %s",
                        worker.id, job_id, exc,
                    )
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    def pin_session_worker(self, session_id: str, worker_id: str) -> bool:
        """Pin a session to a specific worker if that worker exists."""
        if not self._workers.get(worker_id):
            return False
        self._session_pins[session_id] = worker_id
        # If the session was previously affinitied to a different
        # worker, close the slot on the old worker — otherwise the
        # slot persists indefinitely on a worker that no longer owns
        # this session. Same fix as _route_by_role's migration path.
        prior = self._session_workers.get(session_id)
        if prior is not None and prior != worker_id:
            self._node_tracker.close_context_slot(prior, session_id)
        self._session_workers[session_id] = worker_id
        self.sessions.get_or_create(session_id)
        self._save_pins()
        return True

    def unpin_session_worker(self, session_id: str) -> bool:
        """Remove an explicit worker pin from a session."""
        removed = session_id in self._session_pins
        self._session_pins.pop(session_id, None)
        self._save_pins()
        return removed

    def _save_pins(self) -> None:
        """Persist current session worker pins."""
        self.pin_store.save(self._session_pins)

    def _save_worker_states(self) -> None:
        """Persist current worker operational state and manual task overrides.

        Three things are merged into the on-disk file:
        - live enabled/draining flags for currently-connected workers,
        - the prior on-disk entries for workers that aren't connected right
          now (so we don't lose state for a worker that's temporarily down),
        - operator-set manual task overrides keyed by worker_id, which may
          reference disconnected workers too.
        """
        current = self.worker_state_store.load()
        current.update(self._workers.state_snapshot())
        # Layer manual task overrides on top. An entry may exist for a
        # worker that has never connected during this coordinator run, so
        # ensure the dict exists before assigning tasks.
        for worker_id, tasks in self._manual_tasks.items():
            entry = current.setdefault(
                worker_id, {"enabled": True, "draining": False}
            )
            entry["tasks"] = [t.value for t in tasks]
        # Clear tasks for any worker that no longer has an override.
        for worker_id, entry in current.items():
            if worker_id not in self._manual_tasks and "tasks" in entry:
                entry.pop("tasks", None)
        self._worker_states = current
        self.worker_state_store.save(current)

    async def _broadcast_memory_sync(self, exclude_worker_id: str = "") -> None:
        """Broadcast pending memory mutations to all connected workers."""
        if not self._cluster_memory:
            return
        for worker in self._workers.list():
            if worker.id == exclude_worker_id:
                continue
            msg = self._cluster_memory.build_sync_message(target_worker_id=worker.id)
            if msg:
                try:
                    await worker.ws.send(json.dumps(msg))
                except Exception:
                    pass

    async def _replace_worker_impl(
        self,
        target_id: str,
        launcher_url: str,
        launcher_token: str,
        worker_payload: dict[str, Any],
        reason: str = "replace-worker",
    ) -> tuple[dict[str, Any], int]:
        """Core replace-worker flow shared by single-replace and rolling-replace.

        Returns ``(response_body, http_status)`` matching the
        ``/fleet/replace-worker`` endpoint contract: 404 if the worker is
        unknown, 502 if the launcher can't be reached, 200/502 mirroring the
        launcher's own status otherwise.
        """
        existing = self._workers.get(target_id)
        if existing is None:
            return ({"error": f"unknown worker: {target_id}"}, 404)

        # Drain → migrate sessions off this worker.
        self._workers.set_draining(target_id, True)
        await self._initiate_handoffs_for_worker(
            target_id, HandoffReason.WORKER_DRAINING
        )

        # Ask the worker to exit gracefully.
        shutdown_sent = False
        try:
            await existing.ws.send(
                json.dumps({"type": "shutdown", "reason": reason})
            )
            shutdown_sent = True
        except Exception as exc:
            log.warning(
                "replace-worker: shutdown for %s failed: %s", target_id, exc
            )

        # Forward to the launcher.
        worker_payload.setdefault(
            "controller",
            f"ws://{self.config.gateway.host}:{self.config.gateway.port}",
        )
        headers = {"Content-Type": "application/json"}
        if launcher_token:
            headers["Authorization"] = f"Bearer {launcher_token}"

        import httpx

        target = launcher_url.rstrip("/") + "/launch"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(target, json=worker_payload, headers=headers)
        except httpx.RequestError as exc:
            log.warning(
                "replace-worker: launcher %s unreachable (%s)", target, exc
            )
            return (
                {
                    "error": f"launcher unreachable: {exc}",
                    "drained_worker_id": target_id,
                    "shutdown_sent": shutdown_sent,
                },
                502,
            )
        try:
            forwarded = resp.json()
        except ValueError:
            forwarded = {"text": resp.text}
        log.info(
            "replace-worker: %s replaced via %s status=%d shutdown_sent=%s",
            target_id,
            target,
            resp.status_code,
            shutdown_sent,
        )
        return (
            {
                "replaced_worker_id": target_id,
                "shutdown_sent": shutdown_sent,
                "launcher_status": resp.status_code,
                "launcher_response": forwarded,
                "controller_used": worker_payload["controller"],
            },
            200 if resp.is_success else 502,
        )

    async def _initiate_handoffs_for_worker(
        self, worker_id: str, reason: HandoffReason
    ) -> list[str]:
        """Start handoffs for all sessions on a draining/disconnecting worker.

        Returns session IDs that were handed off.
        """
        sessions = self._handoff_manager.sessions_needing_handoff(
            worker_id, self._session_workers
        )
        handed_off = []
        for sid in sessions:
            session = self.sessions.get_or_create(sid)
            token_estimate = sum(
                count_tokens_fallback(m.content) for m in session.conversation.messages
            )
            self._handoff_manager.plan_handoff(
                sid, worker_id, reason,
                conversation_messages=len(session.conversation),
                estimated_tokens=token_estimate,
            )
            # Clear old affinity
            self._session_workers.pop(sid, None)
            self._context_sync.clear_worker(worker_id)

            # Pick a replacement worker. The dispatcher's handoff path skips
            # pin/affinity (those still point at the worker being drained) and
            # goes straight to capability-fallback selection, so we always find
            # a target as long as one suitable worker exists.
            assert self._dispatcher is not None  # set in __post_init__
            decision = self._dispatcher.select_for_handoff(
                sid,
                estimated_tokens=token_estimate,
                exclude={worker_id},
            )
            if decision.worker is not None:
                self._handoff_manager.assign_target(sid, decision.worker.id)
                self._node_tracker.open_context_slot(decision.worker.id, sid, token_estimate)
                self._handoff_manager.complete_handoff(sid, success=True)
            else:
                self._handoff_manager.complete_handoff(
                    sid,
                    success=False,
                    error="No suitable replacement worker available",
                )
            handed_off.append(sid)
        return handed_off

    def _estimate_conversation_tokens(self, session_id: str) -> int:
        """Estimate token count for a session's conversation."""
        session = self.sessions.get_or_create(session_id)
        return sum(count_tokens_fallback(m.content) for m in session.conversation.messages)

    def _desired_worker_capabilities(
        self, *, needs_tools: bool = False
    ) -> dict[str, Any]:
        """Describe the worker shape for scoring, with dynamic tool requirements."""
        cls = self.agent.__class__.__name__
        if cls == "ClaudeCodeRuntime":
            preferred_backend = "claude"
            preferred_mode = "anthropic_messages"
        elif cls == "OllamaRuntime":
            preferred_backend = "ollama"
            preferred_mode = "ollama_chat"
        else:
            preferred_backend = "mlx"
            preferred_mode = "mlx_prompt"
        return {
            "preferred_backend": preferred_backend,
            "preferred_mode": preferred_mode,
            "model": getattr(self.config.model, "name", ""),
            "tools": needs_tools,
        }

    async def _cancel_remote_job(self, session_id: str) -> None:
        job_id = self._session_jobs.get(session_id)
        worker_id = self._session_workers.get(session_id)
        if not job_id or not worker_id:
            return
        worker = self._workers.get(worker_id)
        if not worker:
            return
        await worker.ws.send(
            json.dumps({"type": "cancel_job", "job_id": job_id, "session": session_id})
        )

    async def _dispatch_idle_task(self, worker: WorkerInfo) -> None:
        """Send a background idle task to an idle worker.

        The task runs as a normal agent job. If a real request comes in,
        _preempt_idle_task cancels it first.
        """
        has_tools = bool(worker.capabilities.get("tools", False))
        assigned = self._node_tasks.get(worker.id)
        task = self._idle_manager.next_task_for_worker(worker.id, has_tools, assigned)
        if task is None:
            return

        prompt = IDLE_TASK_PROMPTS[task]
        job_id = uuid.uuid4().hex[:12]
        session_id = f"_idle_{worker.id}_{task.value}"

        from towel.agent.conversation import Conversation

        conv = Conversation(id=session_id)
        conv.add(Role.USER, prompt)

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._workers.assign(worker.id, job_id, session_id)
        self._idle_manager.start_task(worker.id, task)

        log.info("Idle task %s dispatched to %s (job %s)", task, worker.id, job_id)

        payload: dict[str, Any] = {
            "type": "run",
            "job_id": job_id,
            "session": session_id,
            "stream": False,
            "conversation": conv.to_dict(),
        }
        # Include project context if available
        from towel.agent.project import load_project_context
        project_ctx = load_project_context()
        if project_ctx:
            payload["project_context"] = project_ctx

        await worker.ws.send(json.dumps(payload))

        # Collect result in background — don't block
        async def _collect() -> None:
            completed_normally = False
            try:
                while True:
                    msg = await asyncio.wait_for(queue.get(), timeout=300.0)
                    msg_type = msg.get("type")
                    if msg_type == "job_done":
                        result = msg.get("result", {})
                        text = result.get("text", "")
                        # Also check response field for non-streaming
                        if not text:
                            resp = msg.get("response", {})
                            text = resp.get("content", "")
                        self._idle_manager.complete_task(worker.id, text)
                        completed_normally = True
                        break
                    elif msg_type == "job_error":
                        self._idle_manager.complete_task(
                            worker.id, msg.get("message", "error"), error=True
                        )
                        # Worker errored on its own; no cancel needed.
                        completed_normally = True
                        break
                    # job_event — ignore streaming tokens for idle tasks
            except TimeoutError:
                self._idle_manager.complete_task(worker.id, "Timed out", error=True)
            except asyncio.CancelledError:
                self._idle_manager.cancel_task(worker.id)
            finally:
                if not completed_normally:
                    # Tell the worker to drop its in-flight idle job —
                    # otherwise on a coord-side timeout/cancel it keeps
                    # generating output nobody will read. Same pattern
                    # as the inference and classification paths.
                    try:
                        await worker.ws.send(
                            json.dumps(
                                {"type": "cancel_job", "job_id": job_id, "session": session_id}
                            )
                        )
                    except Exception as cancel_exc:
                        log.debug(
                            "Failed to send cancel_job (idle) to %s: %s",
                            worker.id, cancel_exc,
                        )
                self._job_queues.pop(job_id, None)
                self._workers.release(worker.id)
                # After finishing, try to dispatch another idle task
                if not worker.busy and worker.enabled and not worker.draining:
                    await self._dispatch_idle_task(worker)

        asyncio.create_task(_collect())

    async def _schedule_idle_work(self) -> None:
        """Check all idle workers and dispatch background tasks."""
        for worker in self._workers.list():
            if (
                not worker.busy
                and worker.enabled
                and not worker.draining
                and not self._idle_manager.is_idle_task(worker.id)
            ):
                await self._dispatch_idle_task(worker)

    async def _preempt_idle_task(self, worker: WorkerInfo) -> None:
        """Cancel any idle task running on a worker to free it for real work."""
        if not self._idle_manager.is_idle_task(worker.id):
            return
        task = self._idle_manager.cancel_task(worker.id)
        log.info("Preempting idle task %s on %s for real request", task, worker.id)
        job_id = worker.current_job_id
        if job_id:
            await worker.ws.send(
                json.dumps(
                    {
                        "type": "cancel_job",
                        "job_id": job_id,
                        "session": worker.current_session_id or "",
                    }
                )
            )
            self._job_queues.pop(job_id, None)
            self._workers.release(worker.id)

    async def _remote_generate(
        self,
        session_id: str,
        conversation: Any,
        worker: WorkerInfo,
        *,
        stream: bool,
        client_ws: ServerConnection | None = None,
    ) -> dict[str, Any]:
        """Run one inference pass on a remote worker.

        Uses delta sync when the worker has seen this session before,
        falling back to full conversation transfer for first contact.
        """
        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._session_jobs[session_id] = job_id
        self._workers.assign(worker.id, job_id, session_id)

        # Track context usage on the node
        token_estimate = sum(
            count_tokens_fallback(m.content) for m in conversation.messages
        )
        self._node_tracker.open_context_slot(worker.id, session_id, token_estimate)

        # Load project context from coordinator's CWD so workers get it
        from towel.agent.project import load_project_context

        project_ctx = load_project_context()

        # Delta sync: only send new messages if worker has seen this session
        delta = self._context_sync.compute_delta(worker.id, session_id, conversation)
        # Inject coordinator memory into the payload (see _conv_dict_with_memory).
        conv_dict = self._conv_dict_with_memory(conversation)
        if delta.is_full_sync:
            # First time or structural change — send full conversation
            payload: dict[str, Any] = {
                "type": "run",
                "job_id": job_id,
                "session": session_id,
                "stream": stream,
                "conversation": conv_dict,
            }
        else:
            # Incremental: send only the delta
            payload = {
                "type": "run",
                "job_id": job_id,
                "session": session_id,
                "stream": stream,
                "conversation": conv_dict,
                "delta": delta.to_dict(),
            }

        if project_ctx:
            payload["project_context"] = project_ctx

        await worker.ws.send(json.dumps(payload))

        accumulated_text = ""
        # Inference-chunk timeout: how long we'll wait for the next
        # token / event from the worker. Operator-tunable so a cold
        # 30B model doesn't trip a 120s default before its first
        # token. Falls back to 300s if config is absent.
        chunk_timeout = float(getattr(self.config, "worker_inference_timeout", 300.0) or 300.0)
        # Same cancel-on-disconnect pattern as the streaming siblings:
        # only the worker's own terminal event counts as "completed
        # normally"; anything else (timeout, CancelledError, client
        # disconnect propagating up) triggers a cancel_job so the
        # worker doesn't keep producing for nobody.
        completed_normally = False
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=chunk_timeout)
                except TimeoutError as exc:
                    # asyncio.TimeoutError stringifies to "" — convert to
                    # a RuntimeError carrying the worker id and timeout
                    # so the caller's generic except renders something
                    # the API user can act on. Mirrors the chat-fast
                    # path's fix in 2bab006.
                    raise RuntimeError(
                        f"worker {worker.id} stalled mid-stream "
                        f"after {chunk_timeout:.0f}s"
                    ) from exc
                msg_type = msg.get("type")
                if msg_type == "job_event":
                    event = msg.get("event", {})
                    # Accumulate token text for callers that need the full response
                    if event.get("type") == "token":
                        accumulated_text += event.get("text", "")
                    if client_ws is not None:
                        await client_ws.send(json.dumps(event))
                elif msg_type == "job_done":
                    self._context_sync.advance_cursor(worker.id, session_id, conversation)
                    # "run" job_done carries response/conversation, not result
                    result = msg.get("result") or {}
                    if not result.get("text") and accumulated_text:
                        result = {"text": accumulated_text, "metadata": {}}
                    elif not result.get("text"):
                        # Non-streaming: extract last assistant message
                        resp = msg.get("response", {})
                        result = {
                            "text": resp.get("content", ""),
                            "metadata": resp.get("metadata", {}),
                        }
                    completed_normally = True
                    return result
                elif msg_type == "job_error":
                    # Worker already failed on its own; no cancel needed.
                    completed_normally = True
                    raise RuntimeError(msg.get("message", "Remote worker failed"))
        finally:
            if not completed_normally:
                try:
                    await worker.ws.send(
                        json.dumps(
                            {"type": "cancel_job", "job_id": job_id, "session": session_id}
                        )
                    )
                except Exception as exc:
                    log.debug(
                        "Failed to send cancel_job to %s for %s: %s",
                        worker.id, job_id, exc,
                    )
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    async def _step_remote_inference(
        self, session_id: str, session: Any, worker: WorkerInfo
    ) -> Any:
        """Run the local tool loop while outsourcing each generation pass."""
        # Coordinator-measured start, used as a fallback when the
        # worker reports no total_ms (same rationale as _quick_remote_infer).
        coord_start = time.monotonic()

        try:
            return await self._step_remote_inference_inner(
                session_id, session, worker, coord_start,
            )
        except Exception:
            # On exception from _remote_generate, stamp the dispatch
            # decision so operators can see "agent loop failed after Xs"
            # in /dispatch/recent. Without this the original dispatch
            # decision shows no total_ms.
            if self._dispatcher is not None:
                decision = self._dispatcher.last_decision_for_session(session_id)
                if decision is not None and decision.total_ms is None:
                    coord_total_ms = (time.monotonic() - coord_start) * 1000.0
                    decision.record_completion(
                        ttft_ms=None,
                        total_ms=round(coord_total_ms, 1),
                    )
            raise

    async def _step_remote_inference_inner(
        self,
        session_id: str,
        session: Any,
        worker: WorkerInfo,
        coord_start: float,
    ) -> Any:
        """Inner body of `_step_remote_inference` — see wrapper for
        rationale. Split so the outer can stamp dispatch timing on
        exception without rewriting the loop body.
        """
        total_tokens = 0
        last_metadata: dict[str, Any] = {"remote_worker": worker.id}
        remaining_text = ""
        # Loop-detection state. Track the last few tool-call fingerprints;
        # if the same (name, args) repeats too many times consecutively
        # the worker is stuck — break out before we hit the
        # MAX_TOOL_ITERATIONS cap, which on a 20s-per-call worker would
        # take ~5 hours to reach. 3 identical calls in a row is a
        # generous threshold (real tool loops occasionally repeat a
        # lookup) without being so loose it lets the loop run wild.
        last_call_fingerprints: list[str] = []
        LOOP_REPEAT_LIMIT = 3
        loop_detected = False

        for _ in range(MAX_TOOL_ITERATIONS):
            result = await self._remote_generate(
                session_id,
                session.conversation,
                worker,
                stream=False,
            )
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            total_tokens += metadata.get("tokens", metadata.get("output_tokens", 0))
            last_metadata = {"remote_worker": worker.id, **metadata}

            tool_calls, remaining_text = parse_tool_calls(text)
            if not tool_calls:
                from towel.agent.conversation import Message

                # Stamp the dispatch decision with the final-iteration
                # timing so the dispatch log reflects this session's
                # actual end-to-end latency — not just the routing
                # decision that triggered it.
                coord_total_ms = (time.monotonic() - coord_start) * 1000.0
                stamped_total = metadata.get("total_ms")
                if stamped_total is None:
                    stamped_total = round(coord_total_ms, 1)
                if self._dispatcher is not None:
                    decision = self._dispatcher.last_decision_for_session(session_id)
                    if decision is not None:
                        decision.record_completion(
                            ttft_ms=metadata.get("ttft_ms"),
                            total_ms=stamped_total,
                        )
                # Also stamp the coordinator-measured total_ms onto
                # the response metadata so callers (/api/ask) see real
                # latency even when the worker didn't report it.
                response_meta = last_metadata | {"tokens": total_tokens}
                if "total_ms" not in response_meta:
                    response_meta["total_ms"] = stamped_total
                response = Message(
                    role=Role.ASSISTANT,
                    content=text,
                    metadata=response_meta,
                )
                session.conversation.messages.append(response)
                return response

            if remaining_text:
                session.conversation.add(Role.ASSISTANT, remaining_text)

            # Loop-detection: compute a fingerprint for this iteration's
            # tool calls. When the same fingerprint appears LOOP_REPEAT_LIMIT
            # times consecutively, the worker is stuck — break the loop
            # so we don't burn hours of inference on a request a real
            # model would have given up on in 2 iterations.
            iter_fingerprint = json.dumps(
                [(tc.name, tc.arguments) for tc in tool_calls],
                sort_keys=True, default=str,
            )
            last_call_fingerprints.append(iter_fingerprint)
            if len(last_call_fingerprints) > LOOP_REPEAT_LIMIT:
                last_call_fingerprints.pop(0)
            if (
                len(last_call_fingerprints) == LOOP_REPEAT_LIMIT
                and len(set(last_call_fingerprints)) == 1
            ):
                log.warning(
                    "Tool-call loop detected on session %s (same call %r "
                    "repeated %d times); breaking out",
                    session_id, tool_calls[0].name, LOOP_REPEAT_LIMIT,
                )
                loop_detected = True
                break

            for tc in tool_calls:
                try:
                    tool_result = await self.agent.skills.execute_tool(tc.name, tc.arguments)
                    result_str = tool_result if isinstance(tool_result, str) else str(tool_result)
                    is_error = tool_result_is_error(result_str)
                except Exception as exc:
                    result_str = f"Error executing {tc.name}: {exc}"
                    is_error = True
                    log.error(result_str)

                session.conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )

        from towel.agent.conversation import Message

        if loop_detected:
            stuck_msg = (
                f"I got stuck calling {tool_calls[0].name!r} repeatedly. "
                "Stopping to avoid burning more time on this turn."
            )
            response = Message(
                role=Role.ASSISTANT,
                content=remaining_text + ("\n\n" + stuck_msg if remaining_text else stuck_msg),
                metadata=last_metadata | {"tokens": total_tokens, "loop_detected": True},
            )
            session.conversation.messages.append(response)
            return response

        response = Message(
            role=Role.ASSISTANT,
            content=remaining_text or "I've reached my tool execution limit for this turn.",
            metadata=last_metadata | {"tokens": total_tokens, "max_iterations": True},
        )
        session.conversation.messages.append(response)
        return response

    async def _stream_remote_inference(
        self,
        ws: ServerConnection,
        session_id: str,
        session: Any,
        worker: WorkerInfo,
    ) -> None:
        """Run the local streaming tool loop while outsourcing generation."""
        total_tokens = 0
        remaining_text = ""
        # Loop-detection state (mirrors _step_remote_inference_inner).
        # See 4f5a63e for rationale — without this a stuck worker
        # loops 999 times, ~5h of compute on a 20s/call worker.
        last_call_fingerprints: list[str] = []
        LOOP_REPEAT_LIMIT = 3
        loop_detected_call_name: str | None = None

        for _ in range(MAX_TOOL_ITERATIONS):
            result = await self._remote_generate(
                session_id,
                session.conversation,
                worker,
                stream=True,
                client_ws=ws,
            )
            full_text = result.get("text", "")

            if self.agent.is_cancelled:
                if full_text.strip():
                    session.conversation.add(Role.ASSISTANT, full_text)
                await ws.send(
                    json.dumps(
                        AgentEvent.cancelled(
                            full_text,
                            metadata={"tokens": total_tokens, "reason": "user_cancelled"},
                        ).to_ws_message(session_id)
                    )
                )
                return

            tokenizer = getattr(self.agent, "_tokenizer", None)
            if tokenizer:
                token_count = len(tokenizer.encode(full_text))
            else:
                token_count = len(full_text.split())
            total_tokens += token_count

            tool_calls, remaining_text = parse_tool_calls(full_text)
            if not tool_calls:
                session.conversation.add(Role.ASSISTANT, full_text)
                await ws.send(
                    json.dumps(
                        AgentEvent.complete(
                            full_text,
                            metadata={"tokens": total_tokens, "remote_worker": worker.id},
                        ).to_ws_message(session_id)
                    )
                )
                return

            if remaining_text:
                session.conversation.add(Role.ASSISTANT, remaining_text)

            # Loop-detection check — see _step_remote_inference_inner
            # for rationale (commit 4f5a63e).
            iter_fingerprint = json.dumps(
                [(tc.name, tc.arguments) for tc in tool_calls],
                sort_keys=True, default=str,
            )
            last_call_fingerprints.append(iter_fingerprint)
            if len(last_call_fingerprints) > LOOP_REPEAT_LIMIT:
                last_call_fingerprints.pop(0)
            if (
                len(last_call_fingerprints) == LOOP_REPEAT_LIMIT
                and len(set(last_call_fingerprints)) == 1
            ):
                log.warning(
                    "Tool-call loop detected on streaming session %s "
                    "(same call %r repeated %d times); breaking out",
                    session_id, tool_calls[0].name, LOOP_REPEAT_LIMIT,
                )
                loop_detected_call_name = tool_calls[0].name
                break

            for tc in tool_calls:
                event_msg = AgentEvent.tool_call(tc.name, tc.arguments).to_ws_message(session_id)
                await ws.send(json.dumps(event_msg))
                try:
                    tool_result = await self.agent.skills.execute_tool(tc.name, tc.arguments)
                    result_str = tool_result if isinstance(tool_result, str) else str(tool_result)
                    is_error = tool_result_is_error(result_str)
                except Exception as exc:
                    result_str = f"Error executing {tc.name}: {exc}"
                    is_error = True
                    log.error(result_str)

                result_msg = AgentEvent.tool_result(tc.name, result_str).to_ws_message(session_id)
                await ws.send(json.dumps(result_msg))
                session.conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )

        # Decide which terminal message to emit based on why we exited
        # the loop. Loop-detected gets its own structured note; running
        # out of iterations naturally is the existing fallback.
        if loop_detected_call_name is not None:
            stuck_msg = (
                f"I got stuck calling {loop_detected_call_name!r} repeatedly. "
                "Stopping to avoid burning more time on this turn."
            )
            # Persist the stuck message so the next turn (and any
            # /conversations replay) sees the same text the WS client
            # just got. The last iteration's `remaining_text` (if any)
            # was already appended to the conversation in the for-loop,
            # so we only need to add the stuck_msg itself. Without
            # this, the WS client saw the loop-detected complete event
            # but the persisted transcript ended with the last tool
            # result — replay showed no model response.
            session.conversation.add(Role.ASSISTANT, stuck_msg)
            await ws.send(
                json.dumps(
                    AgentEvent.complete(
                        (remaining_text + "\n\n" + stuck_msg) if remaining_text else stuck_msg,
                        metadata={
                            "tokens": total_tokens,
                            "remote_worker": worker.id,
                            "loop_detected": True,
                        },
                    ).to_ws_message(session_id)
                )
            )
            return
        # Same persistence symmetry for the max-iterations fall-off
        # path — the natural exit-by-iteration-limit must land in the
        # conversation too, not just the WS event stream. When
        # remaining_text is non-empty it was already added in the
        # loop body, so emit only the iteration-limit notice.
        max_iter_msg = "I've reached my tool execution limit for this turn."
        if not remaining_text:
            session.conversation.add(Role.ASSISTANT, max_iter_msg)
        await ws.send(
            json.dumps(
                AgentEvent.complete(
                    remaining_text or max_iter_msg,
                    metadata={
                        "tokens": total_tokens,
                        "remote_worker": worker.id,
                        "max_iterations": True,
                    },
                ).to_ws_message(session_id)
            )
        )

    async def _stream_remote_response(
        self,
        ws: ServerConnection,
        session_id: str,
        session: Any,
        worker: WorkerInfo,
    ) -> None:
        """Run a streaming response on a remote worker and forward its events."""
        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._session_jobs[session_id] = job_id
        self._workers.assign(worker.id, job_id, session_id)

        await worker.ws.send(
            json.dumps(
                {
                    "type": "run",
                    "job_id": job_id,
                    "session": session_id,
                    "stream": True,
                    "conversation": self._conv_dict_with_memory(session.conversation),
                }
            )
        )

        # Chunk timeout: how long to wait for the next event from the
        # worker. Without this an unannounced worker DC mid-stream
        # leaves the queue.get() hanging forever — the client websocket
        # stays open, the session never releases the worker assignment,
        # and the operator has no way to unwedge it short of killing
        # the coordinator.
        chunk_timeout = float(
            getattr(self.config, "worker_inference_timeout", 300.0) or 300.0
        )
        # Track whether we exited via the worker's own terminal event.
        # On any other exit (client disconnect, timeout, cancellation)
        # we send `cancel_job` so the worker stops emitting tokens
        # that nobody is reading.
        completed_normally = False
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=chunk_timeout)
                except asyncio.TimeoutError:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "session": session_id,
                                "message": (
                                    f"timeout waiting for next stream event from "
                                    f"{worker.id} after {chunk_timeout:.0f}s"
                                ),
                            }
                        )
                    )
                    break
                msg_type = msg.get("type")
                if msg_type == "job_event":
                    event = msg.get("event", {})
                    await ws.send(json.dumps(event))
                elif msg_type == "job_done":
                    conversation = msg.get("conversation")
                    if conversation:
                        session.conversation = session.conversation.from_dict(conversation)
                    completed_normally = True
                    break
                elif msg_type == "job_error":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "session": session_id,
                                "message": msg.get("message", "Remote worker failed"),
                            }
                        )
                    )
                    # Worker already finished (with an error). No
                    # cancel needed.
                    completed_normally = True
                    break
        finally:
            if not completed_normally:
                try:
                    await worker.ws.send(
                        json.dumps(
                            {"type": "cancel_job", "job_id": job_id, "session": session_id}
                        )
                    )
                except Exception as exc:
                    log.debug(
                        "Failed to send cancel_job to %s for %s: %s",
                        worker.id, job_id, exc,
                    )
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    async def iter_remote_tokens(
        self,
        session_id: str,
        session: Any,
        worker: WorkerInfo,
    ):
        """Async-iterate token text from a remote worker for a session.

        Used by the OpenAI-compat SSE endpoint: spins up a run job
        with stream=True, then yields each ``event.text`` from the
        worker's job_event token messages. Cleans up the queue +
        session/worker assignments in a finally block so a client
        disconnect doesn't leak state.
        """
        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._session_jobs[session_id] = job_id
        self._workers.assign(worker.id, job_id, session_id)
        # Track whether we exited via the worker's own job_done — if
        # we did, no need to send a cancel. If we exit any other way
        # (SSE client disconnect, asyncio cancellation, timeout), the
        # finally block sends a cancel_job so the worker stops
        # generating tokens nobody will read.
        completed_normally = False
        try:
            await worker.ws.send(
                json.dumps(
                    {
                        "type": "run",
                        "job_id": job_id,
                        "session": session_id,
                        "stream": True,
                        "conversation": self._conv_dict_with_memory(session.conversation),
                    }
                )
            )
            # Chunk timeout: bound how long we'll wait for the next
            # event. A silent worker DC mid-stream would otherwise
            # leave this generator hanging — the SSE client would keep
            # the HTTP connection open, the session would stay assigned
            # to the dead worker, and the operator would have no way
            # to recover short of restarting the coordinator.
            chunk_timeout = float(
                getattr(self.config, "worker_inference_timeout", 300.0) or 300.0
            )
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=chunk_timeout)
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"timeout waiting for next stream event from "
                        f"{worker.id} after {chunk_timeout:.0f}s"
                    )
                msg_type = msg.get("type")
                if msg_type == "job_event":
                    event = msg.get("event", {})
                    # AgentEvent.token serializes with key "content"
                    # under EventType.TOKEN. Probe a few likely names
                    # so future event-schema tweaks don't silently
                    # break the stream.
                    if event.get("type") == "token":
                        text = (
                            event.get("content")
                            or event.get("text")
                            or event.get("token")
                            or ""
                        )
                        if text:
                            yield text
                elif msg_type == "job_done":
                    conv = msg.get("conversation")
                    if conv:
                        session.conversation = session.conversation.from_dict(conv)
                    completed_normally = True
                    break
                elif msg_type == "job_error":
                    # Worker reported failure — it's done generating
                    # on its own; no cancel needed.
                    completed_normally = True
                    raise RuntimeError(msg.get("message", "remote worker failed"))
        finally:
            if not completed_normally:
                # Client gave up (SSE disconnect, asyncio cancel) or we
                # hit a coordinator-side timeout. Tell the worker to
                # stop so it doesn't burn cycles generating tokens
                # nobody will read.
                try:
                    await worker.ws.send(
                        json.dumps(
                            {"type": "cancel_job", "job_id": job_id, "session": session_id}
                        )
                    )
                except Exception as exc:
                    log.debug(
                        "Failed to send cancel_job to %s for %s: %s",
                        worker.id, job_id, exc,
                    )
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    async def _step_remote(self, session_id: str, session: Any, worker: WorkerInfo) -> Any:
        """Run a non-streaming response on a remote worker."""
        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._session_jobs[session_id] = job_id
        self._workers.assign(worker.id, job_id, session_id)

        await worker.ws.send(
            json.dumps(
                {
                    "type": "run",
                    "job_id": job_id,
                    "session": session_id,
                    "stream": False,
                    "conversation": session.conversation.to_dict(),
                }
            )
        )

        try:
            while True:
                msg = await queue.get()
                msg_type = msg.get("type")
                if msg_type == "job_done":
                    conversation = msg.get("conversation")
                    if conversation:
                        session.conversation = session.conversation.from_dict(conversation)
                    response = msg.get("response", {})
                    from towel.agent.conversation import Message

                    return Message(
                        role=Role.ASSISTANT,
                        content=response.get("content", ""),
                        metadata=response.get("metadata", {}),
                    )
                if msg_type == "job_error":
                    raise RuntimeError(msg.get("message", "Remote worker failed"))
        finally:
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    def _build_http_app(self) -> Starlette:
        """Build the HTTP API + web UI app."""
        web_dir = Path(__file__).parent.parent / "web"

        async def health(_request: Any) -> JSONResponse:
            import socket as _socket
            model_name = getattr(self.config.model, "name", "")
            backend = getattr(self.config, "_backend", None) or (
                "claude" if "claude" in model_name.lower() else
                "ollama" if "ollama" in model_name.lower() else
                "mlx"
            )
            try:
                from towel import __version__ as _coord_version
            except Exception:
                _coord_version = "0.0.0"
            return JSONResponse(
                {
                    "status": "hoopy",
                    "version": "0.1.0",
                    "motto": "Don't Panic.",
                    "connections": len(self._connections),
                    "sessions": len(self.sessions),
                    "workers": self._workers.stats(),
                    "coordinator": {
                        "hostname": _socket.gethostname(),
                        "model": model_name,
                        "backend": backend,
                        "context_window": getattr(self.config.model, "context_window", 0),
                        "max_tokens": getattr(self.config.model, "max_tokens", 0),
                        # Real package version so worker drift detection
                        # has something to compare against.
                        "version": _coord_version,
                        "gateway_ws": (
                            f"ws://{self.config.gateway.host}:{self.config.gateway.port}"
                        ),
                        "gateway_http": (
                            f"http://{self.config.gateway.host}:"
                            f"{self.config.gateway.port + 1}"
                        ),
                    },
                }
            )

        async def sessions_list(request: Request) -> JSONResponse:
            """GET /sessions — live in-memory active sessions.

            Query params:
              ``channel`` — narrow to sessions on this channel.
              ``worker``  — narrow to sessions routed to this worker.
              ``pinned_to`` — narrow to sessions pinned to this worker.

            Same channel-filter semantics as /api/sessions and
            /conversations. The worker/pinned_to filters are unique
            to this endpoint because it carries live routing state.
            """
            channel_filter = request.query_params.get("channel")
            worker_filter = request.query_params.get("worker")
            pinned_filter = request.query_params.get("pinned_to")
            for name, val in (
                ("channel", channel_filter),
                ("worker", worker_filter),
                ("pinned_to", pinned_filter),
            ):
                if val is not None and len(val) > 256:
                    return JSONResponse(
                        {"error": f"{name} must be 256 chars or fewer"},
                        status_code=400,
                    )
            sessions = self.sessions.all()
            if channel_filter is not None:
                sessions = [s for s in sessions if s.conversation.channel == channel_filter]
            if worker_filter is not None:
                sessions = [
                    s for s in sessions
                    if self._session_workers.get(s.id) == worker_filter
                ]
            if pinned_filter is not None:
                sessions = [
                    s for s in sessions
                    if self._session_pins.get(s.id) == pinned_filter
                ]
            return JSONResponse(
                {
                    "sessions": [
                        {
                            "id": s.id,
                            "channel": s.conversation.channel,
                            # `messages` was the original key (in-memory
                            # active sessions). Sister endpoints
                            # /api/sessions and /conversations return
                            # `message_count` for the same datum —
                            # clients hitting both had to special-case
                            # the field name. Expose both here so the
                            # uniform name works everywhere; keep
                            # `messages` for backwards compatibility
                            # with existing web-UI / CLI callers.
                            "messages": len(s.conversation),
                            "message_count": len(s.conversation),
                            "created_at": s.conversation.created_at.isoformat(),
                            "worker_id": self._session_workers.get(s.id),
                            "pinned_worker_id": self._session_pins.get(s.id),
                        }
                        for s in sessions
                    ]
                }
            )

        async def workers_list(_request: Any) -> JSONResponse:
            workers_data = []
            live_ids: set[str] = set()
            for worker in self._workers.matching():
                live_ids.add(worker.id)
                wd = worker.to_dict()
                wd["roles"] = [str(r) for r in self._node_roles.get(worker.id, [])]
                wd["assigned_tasks"] = [str(t) for t in self._node_tasks.get(worker.id, [])]
                # Flag whether the current task list came from an operator
                # override (persisted across restarts) vs the auto-assignment
                # derived from capabilities. The UI uses this to render a
                # badge so it's clear what survives a restart.
                wd["tasks_overridden"] = worker.id in self._manual_tasks
                # Derived quality bucket — same rules the dispatcher uses for
                # task gating, surfaced so operators see "low/medium/high" at
                # a glance without having to inspect capabilities by hand.
                wd["quality_tier"] = worker_quality_tier(worker.capabilities or {})
                workers_data.append(wd)

            # Workers that have persisted state but aren't currently
            # connected. Without surfacing these, operators can't see or
            # clear an override for a worker that's offline — they'd have
            # to edit worker_state.json by hand or wait for it to come back.
            offline: list[dict[str, Any]] = []
            seen: set[str] = set()
            for worker_id, state in self._worker_states.items():
                if worker_id in live_ids or worker_id in seen:
                    continue
                seen.add(worker_id)
                offline.append({
                    "id": worker_id,
                    "enabled": bool(state.get("enabled", True)),
                    "draining": bool(state.get("draining", False)),
                    "manual_tasks": [
                        str(t) for t in self._manual_tasks.get(worker_id, [])
                    ],
                })
            # Catch overrides that exist in memory but never made it to the
            # persisted file (shouldn't happen post-fix, but defend in depth).
            for worker_id, tasks in self._manual_tasks.items():
                if worker_id in live_ids or worker_id in seen:
                    continue
                seen.add(worker_id)
                offline.append({
                    "id": worker_id,
                    "enabled": True,
                    "draining": False,
                    "manual_tasks": [str(t) for t in tasks],
                })

            return JSONResponse(
                {
                    "workers": workers_data,
                    "offline_persisted": offline,
                    "all_tasks": [str(t) for t in TaskType],
                    "requirements": self._desired_worker_capabilities(),
                    "pins": dict(self._session_pins),
                }
            )

        async def worker_tasks_update(request: Request) -> JSONResponse:
            """Override the auto-assigned tasks for a worker.

            Accepts requests for offline workers too, but only to clear an
            existing override (empty list). Setting a non-empty override on
            an unknown worker is rejected to prevent typos from creating
            phantom entries.
            """
            worker_id = request.path_params["worker_id"]
            worker = self._workers.get(worker_id)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "body must be a JSON object"}, status_code=400,
                )

            task_names = body.get("tasks")
            if task_names is None:
                return JSONResponse({"error": "tasks list required"}, status_code=400)
            # Strict type check: a string here would iterate as characters
            # and emit confusing per-char errors ("'c' is not a valid
            # TaskType"). The intent is unambiguously a list.
            if not isinstance(task_names, list):
                return JSONResponse(
                    {"error": f"tasks must be a list (got {type(task_names).__name__})"},
                    status_code=400,
                )
            if not all(isinstance(t, str) for t in task_names):
                return JSONResponse(
                    {"error": "tasks must be a list of strings"},
                    status_code=400,
                )
            # Reject obviously bogus task names early — TaskType values
            # are short identifiers (chat, inference, classification,
            # …). Without this cap, a 2000-char task name would be
            # echoed verbatim in the error response and any access log,
            # because the downstream ValueError stringifies the full
            # bad value.
            for t in task_names:
                if len(t) > 64:
                    return JSONResponse(
                        {"error": "task names must be 64 chars or fewer"},
                        status_code=400,
                    )

            # Validate task names
            try:
                tasks = [TaskType(t) for t in task_names]
            except ValueError as e:
                return JSONResponse({"error": f"Invalid task: {e}"}, status_code=400)

            # Live worker required for setting a new override, but clearing
            # one is allowed for any known persisted-state entry so operators
            # can wipe overrides for currently-offline workers.
            if worker is None:
                if tasks:
                    return JSONResponse(
                        {"error": "Worker not found (cannot set override on offline worker)"},
                        status_code=404,
                    )
                had_override = worker_id in self._manual_tasks
                self._manual_tasks.pop(worker_id, None)
                self._node_tasks.pop(worker_id, None)
                if had_override:
                    self._save_worker_states()
                return JSONResponse({
                    "worker_id": worker_id,
                    "assigned_tasks": [],
                    "tasks_overridden": False,
                    "cleared_offline": had_override,
                })

            # Stash the override so a reconnect doesn't wipe the operator's
            # choice. An empty list explicitly removes the override and
            # immediately re-derives the auto-assigned defaults so the UI
            # doesn't briefly render an empty task list between the click
            # and the next worker register.
            if tasks:
                self._manual_tasks[worker_id] = tasks
                self._node_tasks[worker_id] = tasks
                effective = tasks
            else:
                self._manual_tasks.pop(worker_id, None)
                roles = self._node_roles.get(worker_id, [])
                effective = assign_tasks(worker.capabilities or {}, roles)
                self._node_tasks[worker_id] = effective
            # Persist so the override survives a coordinator restart, not
            # just a worker reconnect.
            self._save_worker_states()
            log.info(
                "Worker %s tasks manually set: %s",
                worker_id,
                ", ".join(str(t) for t in effective),
            )
            return JSONResponse({
                "worker_id": worker_id,
                "assigned_tasks": [str(t) for t in effective],
                "tasks_overridden": worker_id in self._manual_tasks,
            })

        async def cluster_nodes(_request: Any) -> JSONResponse:
            # Merge operator state (enabled / draining / busy) onto
            # each node entry so operators debugging "why isn't this
            # worker being used" don't have to cross-reference
            # /workers in another tab. The NodeTracker tracks
            # hardware + context-slot state; the WorkerRegistry
            # tracks operator-set flags.
            #
            # `busy_since` / `busy_for_seconds` / `current_job_id` /
            # `current_session_id` come along for the ride so the
            # cluster panel can tell "long task running" apart from
            # "wedged for 15 minutes" without bouncing to /workers
            # (the field set the fleet UI also reads).
            from datetime import UTC, datetime
            data = self._node_tracker.to_dict()
            nodes = data.get("nodes", {}) or {}
            now = datetime.now(UTC)
            for worker_id, node_entry in nodes.items():
                worker = self._workers.get(worker_id)
                if worker is not None:
                    node_entry["enabled"] = worker.enabled
                    node_entry["draining"] = worker.draining
                    node_entry["busy"] = worker.busy
                    node_entry["current_job_id"] = worker.current_job_id
                    node_entry["current_session_id"] = worker.current_session_id
                    node_entry["busy_since"] = (
                        worker.busy_since.isoformat()
                        if worker.busy_since is not None else None
                    )
                    node_entry["busy_for_seconds"] = (
                        (now - worker.busy_since).total_seconds()
                        if worker.busy and worker.busy_since is not None
                        else None
                    )
                else:
                    # Node-tracker has an entry the registry no longer
                    # knows about. Mark explicitly so the UI can show
                    # the discrepancy.
                    node_entry["enabled"] = None
                    node_entry["draining"] = None
                    node_entry["busy"] = None
                    node_entry["current_job_id"] = None
                    node_entry["current_session_id"] = None
                    node_entry["busy_since"] = None
                    node_entry["busy_for_seconds"] = None
                # `roles` + `assigned_tasks` were on /workers but not
                # /cluster/nodes — operators debugging "why didn't
                # this task land on this worker" had to cross
                # reference two endpoints to compare hardware
                # capabilities (in cluster/nodes) against routing
                # eligibility (in /workers). Surface both here too so
                # the cluster panel is self-contained.
                node_entry["roles"] = [
                    str(r) for r in self._node_roles.get(worker_id, [])
                ]
                node_entry["assigned_tasks"] = [
                    str(t) for t in self._node_tasks.get(worker_id, [])
                ]
                # `quality_tier` rounds out the parity with /workers —
                # the same low/medium/high bucket the dispatcher uses
                # for task gating. Without it, the cluster panel had
                # the hardware capabilities (vram, context) but not
                # the derived "is this worker capable of CODE_REVIEW"
                # tier, which is what operators actually want to see.
                if worker is not None and worker.capabilities:
                    node_entry["quality_tier"] = worker_quality_tier(
                        worker.capabilities
                    )
                else:
                    # Synthesize the caps the tier function needs from
                    # the node-tracker entry — the field names differ
                    # (resources.vram_total_mb here vs total_vram_mb in
                    # worker.capabilities), so map them at the boundary
                    # so stale tracker entries still get a meaningful
                    # tier instead of "unknown".
                    res = node_entry.get("resources") or {}
                    synthetic_caps = {
                        "total_vram_mb": res.get("vram_total_mb", 0),
                        "context_window": node_entry.get("context_window", 0),
                        "backend": node_entry.get("backend", ""),
                    }
                    node_entry["quality_tier"] = worker_quality_tier(
                        synthetic_caps
                    )
            return JSONResponse(data)

        async def dispatch_explain(request: Request) -> JSONResponse:
            """Preview where a request would be routed for a session.

            Operator-facing introspection — answers "what would happen if a
            new request for session X with intent Y landed right now?" without
            actually consuming a worker or polluting the recent-decisions log.

            Query params:
              session_id (required)
              intent       — chat | tool | task (default: task)
              task_type    — optional TaskType value
              estimated_tokens — int, default 0
            """
            sid = request.query_params.get("session_id")
            if not sid:
                return JSONResponse({"error": "session_id required"}, status_code=400)
            # Match the length + control-char rules /api/ask applies
            # to session_id. Without this, a 100KB session_id slipped
            # through to the dispatcher (which builds a decision
            # carrying the id), got echoed back in the response, and
            # surfaced verbatim in any access log entry. Control
            # characters embedded in the id break log readability.
            if len(sid) > 256:
                return JSONResponse(
                    {"error": "session_id must be 256 chars or fewer"},
                    status_code=400,
                )
            if any(ord(c) < 0x20 or ord(c) == 0x7F for c in sid):
                return JSONResponse(
                    {"error": "session_id must not contain control characters"},
                    status_code=400,
                )
            intent = request.query_params.get("intent", "task")
            # Only the three intent codes the dispatcher actually
            # branches on. Without this, a typo like ?intent=tools
            # silently fell into the "task" path (the layered selector
            # treats unknown intents as task) so the preview was wrong
            # in a way the operator couldn't see.
            if intent not in ("chat", "tool", "task"):
                return JSONResponse(
                    {"error": "intent must be one of: chat, tool, task"},
                    status_code=400,
                )
            task_type_raw = request.query_params.get("task_type")
            task_type: TaskType | None = None
            if task_type_raw:
                # Cap before TaskType() so a 2000-char bogus value
                # can't get echoed verbatim in the error response (same
                # rule as /workers/{id}/tasks).
                if len(task_type_raw) > 64:
                    return JSONResponse(
                        {"error": "task_type must be 64 chars or fewer"},
                        status_code=400,
                    )
                try:
                    task_type = TaskType(task_type_raw)
                except ValueError:
                    return JSONResponse(
                        {"error": f"Unknown task_type: {task_type_raw}"}, status_code=400
                    )
            try:
                estimated_tokens = int(request.query_params.get("estimated_tokens", "0"))
            except ValueError:
                return JSONResponse(
                    {"error": "estimated_tokens must be an integer"}, status_code=400
                )
            # Sanity bounds. Negative tokens make no sense; absurd
            # large values would propagate to selector checks like
            # `estimated_tokens > worker.context_window` and trip
            # quality_degraded even for tiny real requests routed
            # alongside the bogus preview. Cap at 10M — far above
            # any real context window — and floor at 0.
            if estimated_tokens < 0:
                return JSONResponse(
                    {"error": "estimated_tokens must be ≥ 0"},
                    status_code=400,
                )
            if estimated_tokens > 10_000_000:
                return JSONResponse(
                    {"error": "estimated_tokens must be ≤ 10000000"},
                    status_code=400,
                )
            assert self._dispatcher is not None
            decision = self._dispatcher.explain_for_session(
                sid,
                intent=intent,
                task_type=task_type,
                estimated_tokens=estimated_tokens,
            )
            return JSONResponse(decision.to_dict())

        async def dispatch_recent(request: Request) -> JSONResponse:
            """Return the most recent dispatch decisions for operator debugging.

            Each entry shows which worker was picked, the reason code, and how
            many candidates were considered — so an operator can see *why* a
            given session landed on a given worker (or didn't land anywhere
            and got handled by the coordinator).

            Optional filters narrow the window so operators can answer
            specific questions ("show me only degraded routes for session X"):

              ``?reason=<code>``       — exact reason-code match
              ``?worker=<id>``         — only decisions that picked this worker
              ``?session=<id>``        — only decisions for this session
              ``?only_degraded=1``     — only quality_degraded decisions
              ``?only_affinity_missed=1`` — only affinity_missed decisions
              ``?only_pin_missed=1``   — only pin_missed decisions
              ``?min_total_ms=N``      — only decisions with total_ms ≥ N
              ``?intent=chat|tool|task`` — only decisions of this intent class
              ``?limit=N``             — cap the response (default 20)

            Filters apply *before* the limit so a tight limit doesn't hide
            matches that would have surfaced from earlier in the buffer.
            """
            try:
                limit = max(1, min(int(request.query_params.get("limit", "20")), 500))
            except ValueError:
                return JSONResponse({"error": "limit must be an integer"}, status_code=400)
            reason = request.query_params.get("reason")
            worker_filter = request.query_params.get("worker")
            session_filter = request.query_params.get("session")
            # `intent` filter: chat | tool | task. Matches the intent
            # field on each decision so operators can ask "show me
            # only chat traffic" or "only tool dispatches". Bogus
            # values reject with 400 — a typo like `?intent=tools`
            # would otherwise silently match nothing and look like
            # an empty log (same defensive shape /dispatch/explain
            # used since 2026-04).
            intent_filter = request.query_params.get("intent")
            if intent_filter is not None and intent_filter not in (
                "chat", "tool", "task",
            ):
                return JSONResponse(
                    {"error": "intent must be one of: chat, tool, task"},
                    status_code=400,
                )
            only_degraded = request.query_params.get("only_degraded") in {"1", "true"}
            only_affinity_missed = request.query_params.get("only_affinity_missed") in {"1", "true"}
            # `only_pin_missed=1` surfaces decisions where the
            # session had an explicit pin but it was silently
            # bypassed (pinned worker was busy/draining/disabled).
            # Without this filter operators had no fast way to spot
            # "my pin is being ignored" — the bypassed decision
            # looked like a normal route in the log.
            only_pin_missed = request.query_params.get("only_pin_missed") in {"1", "true"}
            # `min_total_ms=<N>` surfaces only decisions whose
            # measured wall-time is >= N ms — operators triaging slow
            # / timed-out requests previously had to eyeball the
            # whole list for high `total_ms` values. Pairs naturally
            # with the worker_inference_timeout (default 300s) and
            # chat_fast_timeout (60s) so a quick `?min_total_ms=60000`
            # lights up every long tail. Entries without total_ms
            # (e.g. recorded mid-flight) are excluded — they haven't
            # yet been stamped with timing.
            try:
                min_total_ms_raw = request.query_params.get("min_total_ms")
                min_total_ms = (
                    float(min_total_ms_raw) if min_total_ms_raw is not None else None
                )
            except ValueError:
                return JSONResponse(
                    {"error": "min_total_ms must be a number"}, status_code=400,
                )
            if min_total_ms is not None and min_total_ms < 0:
                return JSONResponse(
                    {"error": "min_total_ms must be ≥ 0"}, status_code=400,
                )

            assert self._dispatcher is not None
            entries = [d.to_dict() for d in self._dispatcher.history()]
            # Hide ephemeral collaboration sessions by default — each
            # ensemble run records one decision per fan-out worker,
            # which would otherwise dominate the recent-decisions
            # view and confuse operators looking for their actual
            # user-facing sessions. Opt-in via `?include_ephemeral=1`
            # so the audit trail remains accessible.
            include_ephemeral = request.query_params.get(
                "include_ephemeral",
            ) in {"1", "true"}
            if not include_ephemeral:
                entries = [
                    e for e in entries
                    if not str(e.get("session_id") or "").startswith(
                        ("_ens_", "_verify_", "_synth_")
                    )
                ]
            if reason:
                entries = [e for e in entries if e.get("reason") == reason]
            if worker_filter:
                entries = [e for e in entries if e.get("worker_id") == worker_filter]
            if session_filter:
                entries = [e for e in entries if e.get("session_id") == session_filter]
            if intent_filter:
                entries = [e for e in entries if e.get("intent") == intent_filter]
            if only_degraded:
                entries = [e for e in entries if e.get("quality_degraded")]
            if only_affinity_missed:
                entries = [e for e in entries if e.get("affinity_missed")]
            if only_pin_missed:
                entries = [e for e in entries if e.get("pin_missed")]
            if min_total_ms is not None:
                entries = [
                    e for e in entries
                    if isinstance(e.get("total_ms"), (int, float))
                    and e["total_ms"] >= min_total_ms
                ]

            # Log freshness: oldest-entry age + cap occupancy so the
            # UI can warn when the operator is looking at a saturated
            # ring buffer (audit data is being dropped) or a stale
            # window (no recent activity).
            history_size = int(getattr(self.config, "dispatch_history_size", 500) or 500)
            log_status: dict[str, Any] = {
                "size": len(self._dispatcher.history()),
                "cap": history_size,
            }
            full_history = self._dispatcher.history()
            if full_history:
                # `oldest_age_seconds` previously looked for an
                # `ts` field in ISO string form, but to_dict() emits
                # `timestamp` as a Unix float (time.time()). The
                # mismatch made datetime.fromisoformat("") raise
                # ValueError, which the bare except swallowed —
                # `oldest_age_seconds` was silently missing from
                # every log_status payload. The UI's "log freshness"
                # warning had no signal to act on.
                try:
                    import time as _time
                    oldest_ts = full_history[0].timestamp
                    if isinstance(oldest_ts, (int, float)):
                        age_s = _time.time() - oldest_ts
                        log_status["oldest_age_seconds"] = max(0, int(age_s))
                except Exception:
                    pass
                log_status["saturated"] = len(full_history) >= history_size
            return JSONResponse(
                {
                    "decisions": entries[-limit:],
                    "total_matching": len(entries),
                    "no_workers_reason": REASON_NO_WORKERS,
                    "log_status": log_status,
                }
            )

        async def memory_list(request: Request) -> JSONResponse:
            """Return the agent's persistent memories.

            Operators (and curious users) currently have no easy way to see
            what the agent has remembered across sessions — only the prompt
            block injection on each turn proves anything is stored. This
            endpoint returns the raw entries as JSON, with optional filters:

              ``?type=fact``  — restrict to one memory type
              ``?q=<text>``   — substring search across key and content
              ``?limit=N``    — cap the response (default 200)
            """
            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"memories": [], "count": 0})
            try:
                limit = max(1, min(int(request.query_params.get("limit", "200")), 1000))
            except ValueError:
                return JSONResponse(
                    {"error": "limit must be an integer"}, status_code=400
                )
            mem_type = request.query_params.get("type") or None
            query = request.query_params.get("q") or None
            tag = request.query_params.get("tag") or None
            # Strip + length + control-char guard on `q` so a 2000-char
            # FTS scan / null byte / embedded newline can't slip into
            # the substring fallback path. Matches the /search rule.
            if query is not None:
                query = query.strip() or None
            if query is not None:
                if len(query) > 256:
                    return JSONResponse(
                        {"error": "q must be 256 chars or fewer"},
                        status_code=400,
                    )
                if any(ord(c) < 0x20 or ord(c) == 0x7F for c in query):
                    return JSONResponse(
                        {"error": "q must not contain control characters"},
                        status_code=400,
                    )
            # Same length + control-char guard on `tag`. The tag flows
            # into a LIKE pattern (memory/store.py: `tags LIKE
            # f'%"{tag}"%'`) which trips SQLITE_MAX_LIKE_PATTERN_LENGTH
            # on a 1MB tag — the same crash class as the long-query bug
            # fixed in eb86631.
            if tag is not None:
                tag = tag.strip() or None
            if tag is not None:
                if len(tag) > 256:
                    return JSONResponse(
                        {"error": "tag must be 256 chars or fewer"},
                        status_code=400,
                    )
                if any(ord(c) < 0x20 or ord(c) == 0x7F for c in tag):
                    return JSONResponse(
                        {"error": "tag must not contain control characters"},
                        status_code=400,
                    )
            # scope semantics on this endpoint:
            #   absent           — honor the store's default (project + global)
            #   "__all__"        — no filter (audit across every project)
            #   ""               — global only
            #   "proj:..."       — that scope only
            scope_raw = request.query_params.get("scope")
            # When the caller asks for "__all__" we sidestep the
            # runtime store's default_scope by opening a fresh
            # connection to the same file with no default. Cheaper
            # than threading another sentinel through recall_all().
            if scope_raw == "__all__":
                from towel.memory.store import MemoryStore as _MS
                view = _MS(store_dir=memory.store_dir)
                scope_arg: str | None = None
            else:
                view = memory
                scope_arg = scope_raw if scope_raw is not None else None
            try:
                if query:
                    entries = view.search(query, scope=scope_arg)
                    if mem_type:
                        entries = [e for e in entries if e.memory_type == mem_type]
                    if tag:
                        entries = [e for e in entries if tag in (e.tags or [])]
                else:
                    entries = view.recall_all(
                        memory_type=mem_type, tag=tag, scope=scope_arg,
                    )
            except Exception as exc:
                log.exception("Memory listing failed: %s", exc)
                return JSONResponse(
                    {"error": f"Memory backend error: {exc}"}, status_code=500
                )
            # Newest first by updated_at, oldest last.
            entries = sorted(entries, key=lambda e: e.updated_at, reverse=True)
            return JSONResponse(
                {
                    "memories": [_memory_entry_dict(e) for e in entries[:limit]],
                    "count": len(entries),
                    "truncated": len(entries) > limit,
                }
            )

        async def memory_recalls(request: Request) -> JSONResponse:
            """List recent recall events (what was queried, what came back).

            Query params:
              ``limit``   default 50, capped at 500
              ``hours``   restrict to recalls within this window
              ``key``     substring filter on returned keys + query text
            """
            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"recalls": []})
            try:
                limit = int(request.query_params.get("limit", "50"))
                limit = max(1, min(limit, 500))
            except ValueError:
                return JSONResponse({"error": "limit must be int"}, status_code=400)
            hours_raw = request.query_params.get("hours")
            since_hours: float | None = None
            if hours_raw is not None:
                try:
                    since_hours = float(hours_raw)
                except ValueError:
                    return JSONResponse({"error": "hours must be numeric"}, status_code=400)
                # Reject NaN, inf, -inf — float() parses these as valid
                # floats but downstream SQL bound comparisons yield
                # silent zero-row results that look identical to
                # "no recent activity." Better to fail loud.
                import math
                if not math.isfinite(since_hours):
                    return JSONResponse(
                        {"error": "hours must be a finite number"},
                        status_code=400,
                    )
                # Negative hours never make semantic sense — would scan
                # the future, which is empty. Fail loud rather than
                # returning a misleading empty result.
                if since_hours < 0:
                    return JSONResponse(
                        {"error": "hours must be ≥ 0"}, status_code=400,
                    )
            key_filter = request.query_params.get("key") or None
            # Strip + length + control-char guard on key_filter so a
            # 1000-char value or embedded null byte can't slip into the
            # substring scan. Matches /search and /memory rules.
            if key_filter is not None:
                key_filter = key_filter.strip() or None
            if key_filter is not None:
                if len(key_filter) > 256:
                    return JSONResponse(
                        {"error": "key must be 256 chars or fewer"},
                        status_code=400,
                    )
                if any(ord(c) < 0x20 or ord(c) == 0x7F for c in key_filter):
                    return JSONResponse(
                        {"error": "key must not contain control characters"},
                        status_code=400,
                    )
            try:
                rows = memory.recent_recalls(
                    limit=limit, since_hours=since_hours, key_filter=key_filter,
                )
            except Exception as exc:
                log.exception("memory.recent_recalls failed: %s", exc)
                return JSONResponse({"error": _err_str(exc)}, status_code=500)
            return JSONResponse({"recalls": rows, "count": len(rows)})

        async def memory_activity(request: Request) -> JSONResponse:
            """Time-bucketed counts of memory writes for the recent window.

            Query params:
              ``hours``        window size, default 24, capped at 168 (1 week)
              ``bucket_hours`` size of each bucket, default 1
              ``column``       created_at (default) or updated_at

            Returns a list of buckets oldest→newest with count and
            by-source breakdown so the UI can render a sparkline +
            tooltip without further roundtrips.
            """
            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"buckets": []})
            try:
                hours = float(request.query_params.get("hours", "24"))
                hours = max(0.5, min(hours, 168.0))
                bucket_hours = float(request.query_params.get("bucket_hours", "1"))
                bucket_hours = max(0.1, min(bucket_hours, hours))
            except ValueError:
                return JSONResponse(
                    {"error": "hours / bucket_hours must be numeric"},
                    status_code=400,
                )
            column = request.query_params.get("column", "created_at")
            if column not in ("created_at", "updated_at"):
                return JSONResponse(
                    {"error": "column must be created_at or updated_at"},
                    status_code=400,
                )
            try:
                buckets = memory.activity(
                    hours=hours, bucket_hours=bucket_hours, column=column,
                )
            except Exception as exc:
                log.exception("memory.activity failed: %s", exc)
                return JSONResponse({"error": _err_str(exc)}, status_code=500)
            return JSONResponse(
                {
                    "buckets": buckets,
                    "hours": hours,
                    "bucket_hours": bucket_hours,
                    "column": column,
                }
            )

        async def memory_inspect(request: Request) -> JSONResponse:
            """Return one memory + its salience, related entries, and freshness.

            The web inspect modal uses this to show context that the
            list endpoint can't carry cheaply per-row: graph neighbors,
            computed salience score, and whether the entry has been
            recently active. Returns 404 when the key is unknown so
            the UI can render a clean "no such memory" state.
            """
            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"error": "memory disabled"}, status_code=503)
            key = request.path_params["key"]
            entry = memory.recall(key)
            if entry is None:
                return JSONResponse(
                    {"error": f"no memory with key {key!r}"}, status_code=404
                )
            try:
                from towel.memory.store import salience as _salience
                score = _salience(entry)
            except Exception:
                score = None
            try:
                related = memory.recall_related(key, limit=5)
            except Exception:
                related = []
            try:
                recent_recalls = memory.recalls_returning(key, limit=5)
            except Exception:
                recent_recalls = []
            return JSONResponse(
                {
                    "entry": _memory_entry_dict(entry),
                    "salience": score,
                    "related": [
                        {"weight": w, **rel.to_dict()} for rel, w in related
                    ],
                    "recent_recalls": recent_recalls,
                }
            )

        async def memory_stats(_request: Request) -> JSONResponse:
            """Aggregate counts and salience signal for the memory panel.

            Returns a small JSON object the UI can use to render a
            dashboard without pulling the entire corpus client-side.
            """
            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse(
                    {"total": 0, "recalled": 0, "by_type": {}, "by_source": {}}
                )
            try:
                entries = memory.recall_all()
            except Exception as exc:
                log.exception("Memory stats failed: %s", exc)
                return JSONResponse(
                    {"error": f"Memory backend error: {exc}"}, status_code=500
                )
            total = len(entries)
            recalled = sum(1 for e in entries if e.recall_count > 0)
            by_type: dict[str, int] = {}
            by_source: dict[str, int] = {}
            by_scope: dict[str, int] = {}
            # Per-pattern health: captures + recalled for each named
            # auto_capture pattern. Lets the web panel show which
            # heuristics are pulling weight vs. generating noise.
            per_pattern: dict[str, dict[str, int]] = {}
            for e in entries:
                by_type[e.memory_type] = by_type.get(e.memory_type, 0) + 1
                src = (getattr(e, "source", "") or "") or "operator"
                by_source[src] = by_source.get(src, 0) + 1
                sc = (getattr(e, "scope", "") or "") or "global"
                by_scope[sc] = by_scope.get(sc, 0) + 1
                if src.startswith("auto_capture:"):
                    label = src.split(":", 1)[1]
                    bucket = per_pattern.setdefault(
                        label, {"captures": 0, "recalled": 0}
                    )
                    bucket["captures"] += 1
                    if e.recall_count > 0:
                        bucket["recalled"] += 1
            # Recent unvalidated captures: surfaces likely heuristic
            # false-positives so operators can spot-check.
            pending = sorted(
                (e for e in entries if e.recall_count == 0),
                key=lambda e: e.created_at,
                reverse=True,
            )[:5]
            return JSONResponse(
                {
                    "total": total,
                    "recalled": recalled,
                    "total_recall_events": sum(e.recall_count for e in entries),
                    "by_type": by_type,
                    "by_source": by_source,
                    "by_scope": by_scope,
                    "auto_capture_patterns": per_pattern,
                    "recent_unvalidated": [_memory_entry_dict(e) for e in pending],
                }
            )

        async def fleet_inventory(_request: Request) -> JSONResponse:
            """Aggregate ``available_models`` across the whole fleet.

            Answers "where can I find model X?" and "which models are most
            broadly cached?" with a single GET. Useful for picking rollout
            targets when you have a model in mind but don't know which
            workers already have it.

            Returns::

                {
                  "models": [
                    {
                      "name": "qwen3.6:27b",
                      "workers": ["mac-studio", "rtx5090"],
                      "cached_count": 2
                    },
                    ...
                  ],
                  "total_unique": 14,
                  "total_workers": 4,
                  "fleet_max_param_b": 70.0
                }

            Sorted by ``cached_count`` desc, then name asc.
            """
            inventory: dict[str, list[str]] = {}
            max_param_b = 0.0
            for worker in self._workers.list():
                caps = worker.capabilities or {}
                # available_models is what the worker reports it has
                # cached on disk; the currently-loaded `model` is
                # always cached too (it's loaded in RAM, must be on
                # disk). Some llama-server builds don't expose
                # /v1/models so available_models comes back empty —
                # if we trust that alone, an actively-serving worker
                # contributes nothing to the inventory. Merge both
                # sources, deduplicated per worker.
                names: set[str] = set()
                for name in caps.get("available_models") or []:
                    if isinstance(name, str) and name:
                        names.add(name)
                current_model = caps.get("model")
                if isinstance(current_model, str) and current_model:
                    names.add(current_model)
                for name in names:
                    inventory.setdefault(name, []).append(worker.id)
                cap = caps.get("max_param_b_est")
                if isinstance(cap, (int, float)) and cap > max_param_b:
                    max_param_b = float(cap)
            entries = [
                {"name": name, "workers": sorted(workers), "cached_count": len(workers)}
                for name, workers in inventory.items()
            ]
            entries.sort(key=lambda e: (-e["cached_count"], e["name"]))
            return JSONResponse(
                {
                    "models": entries,
                    "total_unique": len(entries),
                    "total_workers": len(self._workers.list()),
                    "fleet_max_param_b": max_param_b,
                }
            )

        async def fleet_suggest_targets(request: Request) -> JSONResponse:
            """For a given model name, classify each worker's readiness.

            Lets the operator pick rollout targets sensibly in a
            heterogeneous fleet — a 70B model can't fit on a Pi, and a
            target a worker already has cached doesn't pay the
            download cost on /fleet/spawn.

            Body::

                {"model": "qwen3.6:27b"}        # required
                {"model": "...", "min_param_b": 7.0}  # only show workers
                                                       # that can fit ≥7B

            Returns one entry per connected worker with:

              - ``has_model_cached`` — exact match in ``available_models``
              - ``fits`` — ``max_param_b_est`` >= the model's apparent size
                (heuristic from the name, defaults to "unknown -> True")
              - ``quality_tier`` — the same low/medium/high bucket the UI shows
              - ``max_param_b_est`` and ``available_model_count`` for context

            Plus a ``recommended`` list of worker ids that have the model
            cached, ordered by quality tier desc.
            """
            try:
                body = await request.json()
            except Exception as exc:
                return JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "payload must be a JSON object"}, status_code=400
                )
            model = _stripped_str(body.get("model"))
            if not model:
                return JSONResponse(
                    {"error": "model is required"}, status_code=400
                )
            try:
                min_param_b = float(body.get("min_param_b", 0))
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "min_param_b must be a number"}, status_code=400
                )

            # Rough param-count guess from the model name — e.g. "qwen3-7b"
            # => 7.0, "Llama-3.3-70B-Instruct-4bit" => 70.0. When we can't
            # tell, treat the model as "small enough to maybe fit" so we
            # don't over-eagerly reject workers.
            est_size = _guess_model_param_b(model)

            tier_rank = {"high": 0, "medium": 1, "low": 2}
            # Rough download size at 4-bit quant: ≈0.6 GB / B params.
            est_download_gb = round(est_size * 0.6, 1) if est_size else None
            analyses: list[dict[str, Any]] = []
            for worker in self._workers.list():
                caps = worker.capabilities or {}
                inventory = caps.get("available_models") or []
                # Treat the currently-loaded model as "cached" — it
                # has to be on disk to be loaded in RAM. Without this
                # check, a worker actively serving the requested model
                # comes back has_model_cached=false, which misleads
                # replace-worker UIs into recommending a fresh download.
                current_model = caps.get("model")
                cached = (
                    model in inventory
                    or (isinstance(current_model, str) and current_model == model)
                )
                worker_cap = float(caps.get("max_param_b_est") or 0.0)
                fits = est_size is None or worker_cap >= est_size
                tier = worker_quality_tier(caps)
                if min_param_b and worker_cap < min_param_b:
                    continue
                # Disk-fit check — only relevant for workers that would
                # need to download. Cached workers don't pay the cost.
                live = caps.get("live_resources") or {}
                disk_free_gb = live.get("disk_free_gb")
                disk_fits: bool | None
                if cached or est_download_gb is None or not isinstance(
                    disk_free_gb, (int, float)
                ):
                    # Can't tell, or doesn't apply — leave as None so the
                    # client can treat "unknown" differently from a hard no.
                    disk_fits = None
                else:
                    # Need at least the download size plus a 2 GB working margin.
                    disk_fits = bool(disk_free_gb >= est_download_gb + 2.0)
                analyses.append(
                    {
                        "worker_id": worker.id,
                        "has_model_cached": cached,
                        "fits": fits,
                        "quality_tier": tier,
                        "max_param_b_est": worker_cap,
                        "available_model_count": len(inventory),
                        "backend": caps.get("backend"),
                        "disk_free_gb": disk_free_gb,
                        "disk_fits": disk_fits,
                    }
                )

            # Cached + fits + highest tier first.
            analyses.sort(
                key=lambda a: (
                    not a["has_model_cached"],
                    not a["fits"],
                    tier_rank.get(a["quality_tier"], 9),
                    a["worker_id"],
                )
            )
            recommended = [a["worker_id"] for a in analyses if a["has_model_cached"] and a["fits"]]
            return JSONResponse(
                {
                    "model": model,
                    "estimated_param_b": est_size,
                    "estimated_download_gb": est_download_gb,
                    "workers": analyses,
                    "recommended": recommended,
                }
            )

        async def fleet_upgrade(request: Request) -> JSONResponse:
            """Proxy a software-upgrade command to a remote launcher.

            Pairs with /fleet/replace-worker: upgrade first, then replace, so
            the new spawn picks up the new code. Operator passes the launcher
            URL + token plus either a built-in strategy (``pip`` / ``git-pull``
            / ``uv``) or a custom argv. The launcher runs it synchronously
            and returns the exit code + stdout/stderr.

            Body::

                {"launcher_url": ..., "launcher_token": ...,
                 "strategy": "pip"}                    # built-in
                # or
                {"launcher_url": ..., "launcher_token": ...,
                 "command": ["sh", "-c", "git pull && pip install -e ."]}
            """
            try:
                body = await request.json()
            except Exception as exc:
                return JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "payload must be a JSON object"}, status_code=400
                )
            launcher_url = _stripped_str(body.get("launcher_url"))
            if not launcher_url:
                return JSONResponse(
                    {"error": "launcher_url is required"}, status_code=400
                )
            # Forward strategy/command verbatim — the launcher does the
            # validation and we re-surface its 400 if the operator passed
            # something the launcher doesn't recognise.
            upgrade_body: dict[str, Any] = {}
            if "strategy" in body:
                upgrade_body["strategy"] = body["strategy"]
            if "command" in body:
                upgrade_body["command"] = body["command"]
            if not upgrade_body:
                upgrade_body["strategy"] = "pip"

            launcher_token = _stripped_str(body.get("launcher_token"))
            headers = {"Content-Type": "application/json"}
            if launcher_token:
                headers["Authorization"] = f"Bearer {launcher_token}"

            import httpx

            target = launcher_url.rstrip("/") + "/upgrade"
            # 6 min coordinator timeout — slightly longer than the launcher's
            # own 5-minute pip cap so we get the structured 504 from the
            # launcher rather than a generic httpx timeout.
            try:
                async with httpx.AsyncClient(timeout=360.0) as client:
                    resp = await client.post(target, json=upgrade_body, headers=headers)
            except httpx.RequestError as exc:
                log.warning(
                    "fleet/upgrade: launcher %s unreachable (%s)", target, exc
                )
                return JSONResponse(
                    {"error": f"launcher unreachable: {exc}"}, status_code=502
                )
            try:
                forwarded = resp.json()
            except ValueError:
                forwarded = {"text": resp.text}
            log.info(
                "fleet/upgrade: launcher=%s status=%d strategy=%s",
                target,
                resp.status_code,
                upgrade_body.get("strategy", "custom"),
            )
            return JSONResponse(
                {
                    "launcher_url": launcher_url,
                    "launcher_status": resp.status_code,
                    "launcher_response": forwarded,
                },
                status_code=200 if resp.is_success else 502,
            )

        async def worker_self_upgrade(request: Request) -> JSONResponse:
            """Tell a connected worker to upgrade itself in place.

            One-click upgrade path that bypasses the launcher daemon: we
            send a ``self_upgrade`` message down the worker's existing
            WebSocket connection. The worker runs the upgrade strategy
            (pip / git-pull / uv) and re-execs; the reconnect loop comes
            back online on the new code automatically.

            Request body (all optional)::

                {"strategy": "pip" | "git-pull" | "uv"}   # default: pip

            Returns ``{"ok": true, "worker_id": ..., "strategy": ...}``
            once the message has been sent. We don't wait for completion
            because the worker disconnects mid-upgrade; the caller can
            poll the fleet list to see it re-register.
            """
            worker_id = request.path_params["worker_id"]
            try:
                body = await request.json() if await request.body() else {}
            except Exception:
                return JSONResponse(
                    {"error": "Invalid JSON body"}, status_code=400,
                )
            # Reject non-dict bodies loud rather than silently coercing
            # to `{}` — an operator passing `[1,2,3]` or `"pip"` as the
            # body almost certainly meant something specific and would
            # be confused if we treated it as "no body, use default".
            # Empty body still falls through to default `strategy=pip`
            # via the existing `or {}` path above.
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "body must be a JSON object"}, status_code=400,
                )
            strategy = _stripped_str(body.get("strategy"), "pip")
            # Allowlist of upgrade strategies. Anything else gets
            # rejected with a clear error — the worker side dispatches
            # on this string and an unknown value would either no-op
            # silently or surface a worker-side error long after the
            # operator's request returned 200. Keeping the allowlist
            # here means typos fail loud at the coordinator.
            if strategy not in ("pip", "git-pull", "uv"):
                return JSONResponse(
                    {
                        "error": (
                            "strategy must be one of: pip, git-pull, uv "
                            f"(got {strategy!r})"
                        )
                    },
                    status_code=400,
                )

            worker = self._workers.get(worker_id)
            if worker is None or worker.ws is None:
                return JSONResponse(
                    {"error": f"worker {worker_id!r} not connected"},
                    status_code=404,
                )
            try:
                await worker.ws.send(
                    json.dumps({"type": "self_upgrade", "strategy": strategy})
                )
            except Exception as exc:
                log.warning(
                    "worker self-upgrade: send to %s failed: %s", worker_id, exc
                )
                return JSONResponse(
                    {"error": f"send failed: {exc}"}, status_code=502
                )
            log.info(
                "worker self-upgrade: dispatched to %s (strategy=%s)",
                worker_id, strategy,
            )
            return JSONResponse(
                {"ok": True, "worker_id": worker_id, "strategy": strategy}
            )

        async def fleet_replace_worker(request: Request) -> JSONResponse:
            """Atomically replace a running worker with a new spawn.

            Used to swap a worker onto a different model / backend / config
            (or upgrade its software) without leaving zombies. Steps:

              1. Look up the named worker; 404 if unknown.
              2. Mark it draining so the dispatcher stops sending new work;
                 the existing handoff manager migrates active sessions.
              3. Send ``{"type":"shutdown"}`` over its WebSocket; the worker's
                 reconnect loop sees the flag and exits cleanly rather than
                 retrying.
              4. POST to the supplied launcher with the new worker spec.

            Body mirrors /fleet/spawn but adds ``target_worker_id``.
            """
            try:
                body = await request.json()
            except Exception as exc:
                return JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "payload must be a JSON object"}, status_code=400
                )
            target_id = _stripped_str(body.get("target_worker_id"))
            if not target_id:
                return JSONResponse(
                    {"error": "target_worker_id is required"}, status_code=400
                )
            launcher_url = _stripped_str(body.get("launcher_url"))
            if not launcher_url:
                return JSONResponse(
                    {"error": "launcher_url is required"}, status_code=400
                )
            worker_payload = body.get("worker") or {}
            if not isinstance(worker_payload, dict):
                return JSONResponse(
                    {"error": "worker must be a JSON object"}, status_code=400
                )

            launcher_token = _stripped_str(body.get("launcher_token"))
            reason = body.get("reason") or "replace-worker"
            result, status = await self._replace_worker_impl(
                target_id=target_id,
                launcher_url=launcher_url,
                launcher_token=launcher_token,
                worker_payload=worker_payload,
                reason=reason,
            )
            return JSONResponse(result, status_code=status)

        async def fleet_rolling_replace(request: Request) -> JSONResponse:
            """Walk through a list of workers and replace each sequentially.

            Used to roll a model change or code upgrade across the fleet
            without taking everything down at once. The coordinator drains
            and replaces one worker, optionally waits a configurable delay
            so the new worker has time to connect and pick up traffic, then
            proceeds to the next target.

            Body::

                {
                  "targets": [
                    {"target_worker_id": "w1", "launcher_url": "http://h1:18751"},
                    {"target_worker_id": "w2", "launcher_url": "http://h2:18751"}
                  ],
                  "launcher_token": "shared-bearer",  // optional, applied to all
                  "worker": {"backend": "mlx", "model": "new/model"},
                  "delay_between_seconds": 5
                }

            Returns ``{"results": [...]}`` with one entry per target in order,
            each carrying the per-worker replace result. Continues through
            partial failures rather than aborting — operators can rerun for
            just the failed workers.
            """
            try:
                body = await request.json()
            except Exception as exc:
                return JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "payload must be a JSON object"}, status_code=400
                )
            targets = body.get("targets")
            if not isinstance(targets, list) or not targets:
                return JSONResponse(
                    {"error": "targets must be a non-empty list"}, status_code=400
                )
            shared_token = _stripped_str(body.get("launcher_token"))
            worker_template = body.get("worker") or {}
            if not isinstance(worker_template, dict):
                return JSONResponse(
                    {"error": "worker must be a JSON object"}, status_code=400
                )
            try:
                delay = float(body.get("delay_between_seconds", 5))
            except (TypeError, ValueError):
                return JSONResponse(
                    {"error": "delay_between_seconds must be a number"},
                    status_code=400,
                )
            if delay < 0 or delay > 600:
                return JSONResponse(
                    {"error": "delay_between_seconds must be in [0, 600]"},
                    status_code=400,
                )

            results: list[dict[str, Any]] = []
            for idx, target in enumerate(targets):
                if not isinstance(target, dict):
                    results.append(
                        {"index": idx, "ok": False, "error": "target must be an object"}
                    )
                    continue
                tid = _stripped_str(target.get("target_worker_id"))
                lurl = _stripped_str(target.get("launcher_url"))
                if not tid or not lurl:
                    results.append(
                        {
                            "index": idx,
                            "ok": False,
                            "error": "target needs target_worker_id and launcher_url",
                            "target_worker_id": tid or None,
                        }
                    )
                    continue
                tok = _stripped_str(target.get("launcher_token")) or shared_token
                # Each target gets a fresh copy of the worker template — we
                # don't want one worker's modification (controller auto-fill,
                # worker_id) to leak into the next.
                worker_payload = dict(worker_template)
                # Keep the worker id stable across the replace.
                worker_payload.setdefault("worker_id", tid)
                result, status = await self._replace_worker_impl(
                    target_id=tid,
                    launcher_url=lurl,
                    launcher_token=tok,
                    worker_payload=worker_payload,
                    reason="rolling-replace",
                )
                results.append(
                    {
                        "index": idx,
                        "ok": 200 <= status < 300,
                        "status": status,
                        **result,
                    }
                )
                # Wait before the next target so the new worker has a chance
                # to connect and pick up traffic. Skip the delay after the
                # last target.
                if delay > 0 and idx < len(targets) - 1:
                    await asyncio.sleep(delay)

            overall_ok = all(r.get("ok") for r in results)
            return JSONResponse(
                {"results": results, "ok": overall_ok},
                status_code=200 if overall_ok else 502,
            )

        async def fleet_spawn(request: Request) -> JSONResponse:
            """Proxy a worker-spawn request to a remote launcher.

            Operators run ``towel launcher`` on each candidate worker host;
            this endpoint forwards a launch request to a named launcher and
            auto-fills the ``controller`` URL with this coordinator's own WS
            address so the operator doesn't have to repeat themselves.

            Body shape::

                {
                  "launcher_url":  "http://host:18751",
                  "launcher_token": "<bearer>",        # optional, forwarded as Authorization
                  "worker": {
                    "backend": "ollama",               # everything else passes through
                    "ollama_url": "http://localhost:11434",
                    "worker_id": "gpu-box-1"
                  }
                }

            Returns the launcher's response body and HTTP status code.
            """
            try:
                body = await request.json()
            except Exception as exc:
                return JSONResponse(
                    {"error": f"invalid JSON: {exc}"}, status_code=400
                )
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "payload must be a JSON object"}, status_code=400
                )
            launcher_url = _stripped_str(body.get("launcher_url"))
            if not launcher_url:
                return JSONResponse(
                    {"error": "launcher_url is required"}, status_code=400
                )
            worker_payload = body.get("worker") or {}
            if not isinstance(worker_payload, dict):
                return JSONResponse(
                    {"error": "worker must be a JSON object"}, status_code=400
                )
            # Auto-fill controller with this coordinator's own WS URL when the
            # operator hasn't pre-set one.
            worker_payload.setdefault(
                "controller",
                f"ws://{self.config.gateway.host}:{self.config.gateway.port}",
            )

            launcher_token = _stripped_str(body.get("launcher_token"))
            headers = {"Content-Type": "application/json"}
            if launcher_token:
                headers["Authorization"] = f"Bearer {launcher_token}"

            import httpx

            target = launcher_url.rstrip("/") + "/launch"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(target, json=worker_payload, headers=headers)
            except httpx.RequestError as exc:
                log.warning("fleet/spawn: launcher %s unreachable (%s)", target, exc)
                return JSONResponse(
                    {"error": f"launcher unreachable: {exc}"}, status_code=502
                )
            try:
                forwarded = resp.json()
            except ValueError:
                forwarded = {"text": resp.text}
            log.info(
                "fleet/spawn → %s status=%d worker_id=%s",
                target,
                resp.status_code,
                worker_payload.get("worker_id"),
            )
            return JSONResponse(
                {
                    "launcher_url": launcher_url,
                    "launcher_status": resp.status_code,
                    "launcher_response": forwarded,
                    "controller_used": worker_payload["controller"],
                },
                status_code=200 if resp.is_success else 502,
            )

        async def memory_create(request: Request) -> JSONResponse:
            """Create a new memory. Refuses if the key already exists.

            Body shape::

                {"key": "...", "content": "...", "type": "fact",
                 "tags": [...], "scope": "..."}

            Returns 409 if the key already exists — the operator can
            then choose PATCH for update or another key. Keeps create
            and update semantics distinct so the web UI's "+ new"
            button can't silently clobber existing data.
            """
            from towel.memory.store import MEMORY_TYPES

            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"error": "no memory backend"}, status_code=503)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse({"error": "body must be a JSON object"}, status_code=400)
            raw_key = body.get("key")
            content = body.get("content")
            if not (isinstance(raw_key, str) and raw_key.strip()):
                return JSONResponse({"error": "key required"}, status_code=400)
            # Strip leading/trailing whitespace. Operators submitting
            # `"trailing  "` almost never mean a different key from
            # `"trailing"` — and the as-stored value was invisible in
            # URL bars and CLI output, making the entry effectively
            # unrecallable without exact whitespace replay.
            key = raw_key.strip()
            # Hard cap key length. Memory keys appear in URL paths
            # (/memory/{key}/inspect, /nudge, ...), in dispatch logs,
            # and in the web UI. Anything past a couple hundred chars
            # is almost certainly accidental input that will produce
            # absurd URLs. 256 mirrors the practical filesystem and
            # URL-segment limits.
            if len(key) > 256:
                return JSONResponse(
                    {"error": "key must be 256 chars or fewer"}, status_code=400,
                )
            # Reject control chars (newlines, NULs, etc.) — they break
            # URL routing and log readability. Spaces and printable
            # punctuation are fine.
            if any(ord(c) < 0x20 or ord(c) == 0x7F for c in key):
                return JSONResponse(
                    {"error": "key must not contain control characters"},
                    status_code=400,
                )
            if not (isinstance(content, str) and content.strip()):
                return JSONResponse({"error": "content required"}, status_code=400)
            mtype = body.get("type", "fact")
            if mtype not in MEMORY_TYPES:
                return JSONResponse(
                    {"error": f"type must be one of {list(MEMORY_TYPES)}"},
                    status_code=400,
                )
            tags = body.get("tags")
            if tags is not None and not (
                isinstance(tags, list) and all(isinstance(t, str) for t in tags)
            ):
                return JSONResponse({"error": "tags must be a list of strings"}, status_code=400)
            scope = body.get("scope")
            if scope is not None and not isinstance(scope, str):
                return JSONResponse({"error": "scope must be a string"}, status_code=400)

            if memory.recall(key) is not None:
                return JSONResponse(
                    {"error": f"key {key!r} already exists; use PATCH to update"},
                    status_code=409,
                )
            try:
                entry = memory.remember(
                    key, content, memory_type=mtype,
                    source="api", tags=tags, scope=scope,
                )
            except Exception as exc:
                log.exception("memory.create(%r) failed: %s", key, exc)
                return JSONResponse({"error": _err_str(exc)}, status_code=500)
            return JSONResponse(_memory_entry_dict(entry), status_code=201)

        async def memory_nudge(request: Request) -> JSONResponse:
            """Manually bump an entry's recall_count by 1.

            Operator-driven counter to ``forget``: mark an entry as
            useful so the salience score (and the doctor cold-pattern
            check) stop treating it as never-recalled. Uses the same
            _bump_recall path the retrieval code uses, so the effect
            is indistinguishable from a real surface in the prompt.
            """
            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"error": "no memory backend"}, status_code=503)
            key = request.path_params["key"]
            if memory.recall(key) is None:
                return JSONResponse(
                    {"error": f"no memory with key {key!r}"}, status_code=404
                )
            try:
                memory._bump_recall([key])
            except Exception as exc:
                log.exception("memory.nudge(%r) failed: %s", key, exc)
                return JSONResponse({"error": _err_str(exc)}, status_code=500)
            entry = memory.recall(key)
            return JSONResponse(_memory_entry_dict(entry))

        async def memory_edit(request: Request) -> JSONResponse:
            """Update an existing memory's content, type, tags, or scope.

            Body shape (all fields optional, only changed ones supplied)::

                {"content": "...", "type": "fact", "tags": [...], "scope": "..."}

            Returns 404 if the key doesn't exist. An omitted `content`
            leaves the existing content unchanged; an explicit empty
            string is rejected (400) — that path silently destroyed
            memories before, with no good use case for clobbering to
            empty. tags here REPLACES the list (the CLI add/remove
            tag flow is for additive edits).
            """
            from towel.memory.store import MEMORY_TYPES

            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"error": "no memory backend"}, status_code=503)
            key = request.path_params["key"]
            existing = memory.recall(key)
            if existing is None:
                return JSONResponse(
                    {"error": f"no memory with key {key!r}"}, status_code=404
                )
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON body"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse({"error": "body must be a JSON object"}, status_code=400)

            new_content = body.get("content")
            if not isinstance(new_content, (str, type(None))):
                return JSONResponse({"error": "content must be a string"}, status_code=400)
            # Explicit empty/whitespace content is rejected. POST already
            # requires non-empty content; allowing PATCH to clobber to ""
            # silently destroys the memory. Operators who want to drop a
            # memory should call DELETE.
            if isinstance(new_content, str) and new_content != "" and not new_content.strip():
                return JSONResponse(
                    {"error": "content must not be whitespace-only"}, status_code=400,
                )
            if new_content == "":
                return JSONResponse(
                    {"error": "content must not be empty; DELETE the key to remove it"},
                    status_code=400,
                )
            content = new_content if new_content is not None else existing.content

            new_type = body.get("type") or existing.memory_type
            if new_type not in MEMORY_TYPES:
                return JSONResponse(
                    {"error": f"type must be one of {list(MEMORY_TYPES)}"},
                    status_code=400,
                )

            new_tags = body.get("tags")
            if new_tags is not None and not (
                isinstance(new_tags, list) and all(isinstance(t, str) for t in new_tags)
            ):
                return JSONResponse({"error": "tags must be a list of strings"}, status_code=400)

            new_scope = body.get("scope")
            if new_scope is not None and not isinstance(new_scope, str):
                return JSONResponse({"error": "scope must be a string"}, status_code=400)

            # remember() merges tags into the existing set; PATCH
            # semantics call for REPLACE. So delete-then-reinsert
            # iff tags were specified, otherwise leave them alone.
            try:
                if new_tags is not None:
                    # Drop and re-add to get clean tag replacement.
                    # forget() cascades graph links; we accept the
                    # small loss because editing tags wholesale is
                    # rare. Operators who want additive tag changes
                    # already have memory.add_tag / remove_tag.
                    memory.forget(key)
                    updated = memory.remember(
                        key, content, memory_type=new_type,
                        source=existing.source,
                        tags=new_tags,
                        scope=new_scope if new_scope is not None else existing.scope,
                    )
                else:
                    updated = memory.remember(
                        key, content, memory_type=new_type,
                        source=existing.source,
                        scope=new_scope if new_scope is not None else None,
                    )
            except Exception as exc:
                log.exception("memory.edit(%r) failed: %s", key, exc)
                return JSONResponse({"error": _err_str(exc)}, status_code=500)
            return JSONResponse(_memory_entry_dict(updated))

        async def memory_forget(request: Request) -> JSONResponse:
            """Delete a single memory by key.

            Pairs with ``GET /memory`` so operators (or the chat UI) can
            curate what the agent carries between sessions without dropping
            into a REPL. Returns 404 if the key didn't exist so callers can
            distinguish "already gone" from "successfully removed".
            """
            memory = getattr(self.agent, "memory", None)
            if memory is None:
                return JSONResponse({"error": "no memory backend"}, status_code=503)
            key = request.path_params["key"]
            try:
                removed = memory.forget(key)
            except Exception as exc:
                log.exception("memory.forget(%r) failed: %s", key, exc)
                return JSONResponse({"error": _err_str(exc)}, status_code=500)
            if not removed:
                return JSONResponse({"error": f"no memory with key {key!r}"}, status_code=404)
            return JSONResponse({"ok": True, "key": key})

        async def skills_list(_request: Request) -> JSONResponse:
            """Return the skills loaded on this coordinator and their tools.

            Operator-facing introspection — answers "which skills did the
            agent discover at startup, and what tools does each expose?". A
            common failure mode is the model not calling a tool the operator
            *thought* was available; this endpoint shows the ground truth.
            """
            registry = getattr(self.agent, "skills", None)
            if registry is None:
                return JSONResponse({"skills": [], "total_tools": 0})
            skills_data = []
            for skill_name in registry.list_skills():
                skill = registry.get_skill(skill_name)
                if skill is None:
                    continue
                tools = []
                for tool in skill.tools():
                    params = tool.parameters or {}
                    props = (params.get("properties") or {}) if isinstance(params, dict) else {}
                    tools.append(
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": list(props.keys()),
                        }
                    )
                skills_data.append(
                    {
                        "name": skill_name,
                        "description": skill.description,
                        "tool_count": len(tools),
                        "tools": tools,
                    }
                )
            return JSONResponse(
                {
                    "skills": skills_data,
                    "total_tools": len(registry.tool_names()),
                }
            )

        async def cluster_handoffs(request: Request) -> JSONResponse:
            """Cluster handoff stats + recent records.

            Query params:
              ``limit`` — cap on recent records (default 20, max 200).
                          The earlier signature ignored the URL entirely
                          and always returned 20 — operators triaging
                          a recent disconnect storm couldn't see past
                          the most recent twenty handoffs.
              ``only_failed=1`` — narrow to failed handoffs so the
                          operator can ask "what's going wrong?"
                          without grepping success: false client-side.
              ``reason`` — narrow to one HandoffReason (e.g.
                          worker_draining, worker_disconnected,
                          manual_rebalance). Bogus values reject 400.
            """
            try:
                limit = int(request.query_params.get("limit", "20"))
            except ValueError:
                return JSONResponse(
                    {"error": "limit must be an integer"}, status_code=400,
                )
            limit = max(1, min(limit, 200))
            only_failed = request.query_params.get("only_failed") in {"1", "true"}
            reason_filter = request.query_params.get("reason")
            if reason_filter is not None:
                # The stats output's by_reason map keys are the same
                # set we accept here. Compare against the actual
                # HandoffReason values rather than hardcoding the
                # list so adding a new enum member doesn't drift.
                valid_reasons = {r.value for r in HandoffReason}
                if reason_filter not in valid_reasons:
                    return JSONResponse(
                        {
                            "error": (
                                "reason must be one of: "
                                f"{', '.join(sorted(valid_reasons))}"
                            ),
                        },
                        status_code=400,
                    )
            recent = self._handoff_manager.recent_handoffs(limit=limit)
            if only_failed:
                recent = [r for r in recent if not r.get("success")]
            if reason_filter is not None:
                recent = [r for r in recent if r.get("reason") == reason_filter]
            # Pending handoffs: in-progress migrations. The stats
            # output exposes a bare count under `pending`; expose the
            # records themselves too so an operator triaging "what's
            # stuck right now?" doesn't have to guess. Filter the
            # same way as recent records for consistency.
            pending = self._handoff_manager.pending_handoffs()
            if reason_filter is not None:
                pending = [p for p in pending if p.get("reason") == reason_filter]
            return JSONResponse(
                {
                    "stats": self._handoff_manager.stats(),
                    "recent": recent,
                    "pending": pending,
                }
            )

        async def idle_tasks_status(_request: Any) -> JSONResponse:
            """Return idle task results and active background work."""
            return JSONResponse({
                "results": self._idle_manager.all_results(),
                "active": self._idle_manager.active_tasks(),
            })

        async def worker_state_update(request: Request) -> JSONResponse:
            worker_id = request.path_params["worker_id"]
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "body must be a JSON object"}, status_code=400,
                )

            worker = self._workers.get(worker_id)
            if not worker:
                return JSONResponse({"error": "Worker not found"}, status_code=404)

            enabled = body.get("enabled")
            draining = body.get("draining")
            if enabled is None and draining is None:
                return JSONResponse(
                    {"error": "enabled or draining required"},
                    status_code=400,
                )
            # Strict boolean check. The previous code passed values
            # through `bool(...)` which made any non-empty string
            # truthy — `{"draining": "yes"}` actually drained the
            # worker, and `{"draining": "false"}` did too (the string
            # "false" is truthy in Python). Dangerous for an operator-
            # facing endpoint.
            for field, value in (("enabled", enabled), ("draining", draining)):
                if value is not None and not isinstance(value, bool):
                    return JSONResponse(
                        {"error": f"{field} must be true or false (got {type(value).__name__})"},
                        status_code=400,
                    )

            if enabled is not None:
                self._workers.set_enabled(worker_id, enabled)
            if draining is not None:
                self._workers.set_draining(worker_id, draining)
                # Trigger handoffs when a worker starts draining
                if draining:
                    await self._initiate_handoffs_for_worker(
                        worker_id, HandoffReason.WORKER_DRAINING
                    )
            self._save_worker_states()

            updated = self._workers.get(worker_id)
            assert updated is not None
            # WorkerInfo.to_dict directly — the previous code mistakenly
            # routed this through `_memory_entry_dict`, which is the
            # memory-store response shaper and added spurious
            # `tags`/`source`/`scope`/`last_recalled_at` fields to the
            # worker payload. Operators saw "tags": [] on a worker
            # and reasonably wondered if memory entries were attached.
            return JSONResponse(updated.to_dict())

        async def session_pin_worker(request: Request) -> JSONResponse:
            session_id = request.path_params["session_id"]
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            # Reject non-dict bodies loud — the previous catch-all
            # except clause turned `null` / `[1,2]` / `"foo"` into a
            # misleading "Invalid JSON body" (the JSON parsed fine; it
            # was the shape that was wrong).
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "body must be a JSON object"}, status_code=400,
                )
            raw_worker_id = body.get("worker_id", "")
            # Non-string worker_id (int 42, list, dict) previously
            # crashed on .strip() and surfaced as "Invalid JSON body".
            # Fail with a clear message instead.
            if not isinstance(raw_worker_id, str):
                return JSONResponse(
                    {"error": "worker_id must be a string"}, status_code=400,
                )
            worker_id = raw_worker_id.strip()
            if not worker_id:
                return JSONResponse({"error": "worker_id required"}, status_code=400)
            if not self.pin_session_worker(session_id, worker_id):
                return JSONResponse({"error": "Worker not found"}, status_code=404)
            # The pin was set, but the worker's current state may make
            # it unusable for the next request — surface a non-fatal
            # warning so the operator who just pinned a draining /
            # disabled worker isn't surprised when their next request
            # gets pin_missed-routed elsewhere. The pin itself still
            # took effect (will fire if the worker becomes routable).
            pinned_worker = self._workers.get(worker_id)
            warning: str | None = None
            if pinned_worker is not None:
                if not pinned_worker.enabled:
                    warning = (
                        "pinned worker is currently disabled — "
                        "requests will route elsewhere until you re-enable it"
                    )
                elif pinned_worker.draining:
                    warning = (
                        "pinned worker is currently draining — "
                        "requests will route elsewhere until it returns to service"
                    )
            body_out: dict[str, Any] = {
                "session_id": session_id,
                "worker_id": worker_id,
                "pinned": True,
            }
            if warning is not None:
                body_out["warning"] = warning
            return JSONResponse(body_out)

        async def session_unpin_worker(request: Request) -> JSONResponse:
            session_id = request.path_params["session_id"]
            removed = self.unpin_session_worker(session_id)
            return JSONResponse(
                {
                    "session_id": session_id,
                    "pinned": False,
                    "removed": removed,
                }
            )

        async def webchat(_request: Any) -> HTMLResponse | FileResponse:
            index = web_dir / "index.html"
            if index.exists():
                return FileResponse(index)
            return HTMLResponse("<h1>Towel</h1><p>Web UI not found.</p>", status_code=404)

        async def search_conversations(request: Request) -> JSONResponse:
            query = request.query_params.get("q", "")
            # Strip first, then check — whitespace-only `q` (e.g.
            # `?q=%20%20`) used to flow into the store's regex builder
            # as `re.escape("  ")` which matches any string containing
            # two adjacent spaces, returning essentially every message
            # in the archive. Treat whitespace-only as "missing query".
            query = query.strip()
            if not query:
                return JSONResponse({"error": "Missing ?q= parameter"}, status_code=400)
            # Cap query length. A 2000-char query wastes CPU on a
            # full-archive FTS scan, bloats the echoed response and
            # any access log, and is almost never a legitimate
            # search ("paste the entire stack trace" isn't what this
            # endpoint is for). Same 256-char rule as session_id /
            # memory key.
            if len(query) > 256:
                return JSONResponse(
                    {"error": "q must be 256 chars or fewer"},
                    status_code=400,
                )
            # Reject control characters — null bytes and embedded
            # newlines break log readability and would surface in
            # the echoed `query` field of the JSON response.
            if any(ord(c) < 0x20 or ord(c) == 0x7F for c in query):
                return JSONResponse(
                    {"error": "q must not contain control characters"},
                    status_code=400,
                )
            # Sanitize limit: int() raises on non-numeric, and an
            # uncapped value could scan an entire archive on a busy
            # coordinator. Clamp to 1..200.
            try:
                limit = int(request.query_params.get("limit", "20"))
            except ValueError:
                return JSONResponse(
                    {"error": "limit must be an integer"}, status_code=400
                )
            limit = max(1, min(limit, 200))
            # Optional `?role=user|assistant|tool|system` — narrows
            # results to messages from that role. The store already
            # supports the filter; the gateway just wasn't exposing
            # it. Useful for "show me what the user asked about X"
            # vs "show me everywhere the assistant mentioned X".
            role_raw = request.query_params.get("role")
            role_filter: Role | None = None
            if role_raw is not None:
                try:
                    role_filter = Role(role_raw)
                except ValueError:
                    return JSONResponse(
                        {
                            "error": (
                                "role must be one of: "
                                f"{', '.join(r.value for r in Role)}"
                            ),
                        },
                        status_code=400,
                    )
            # Optional `?regex=1` — treat the query as a regex
            # pattern. Defaults to substring (re.escape'd) which is
            # what most users want; opt-in to regex when the operator
            # actually needs it. Invalid patterns return [] from the
            # store; surface as 400 here so the caller can fix the
            # typo instead of staring at empty results.
            regex_raw = request.query_params.get("regex")
            use_regex = regex_raw in {"1", "true"}
            store = self.sessions.store
            if not store:
                return JSONResponse({"results": []})
            if use_regex:
                # Pre-validate the pattern at the gateway so a bad
                # regex fails fast and loud rather than the store's
                # silent "return [] on re.error" path.
                import re as _re
                try:
                    _re.compile(query)
                except _re.error as exc:
                    return JSONResponse(
                        {"error": f"invalid regex: {exc}"}, status_code=400,
                    )
            results = store.search(
                query, limit=limit, role_filter=role_filter, regex=use_regex,
            )
            return JSONResponse(
                {
                    "query": query,
                    "results": [
                        {
                            "conversation_id": r.conversation_id,
                            # Title rides alongside conversation_id
                            # so search-result UIs can show the
                            # human-readable name. Without this,
                            # results panels fell back to the
                            # session_id (e.g. "openai-chatcmpl-…")
                            # which is meaningless for browsing.
                            "title": r.title,
                            "channel": r.channel,
                            "created_at": r.created_at,
                            "summary": r.summary,
                            "match_count": len(r.matches),
                            "matches": [
                                {
                                    "role": m.role,
                                    "snippet": m.snippet,
                                    "timestamp": m.timestamp,
                                }
                                for m in r.matches[:5]
                            ],
                        }
                        for r in results
                    ],
                }
            )

        async def conversations_list(request: Request) -> JSONResponse:
            """List all persisted conversations (not just active ones).

            Query params:
              ``limit`` — cap on entries scanned from disk (default 50,
                          max 500). Filters apply *after* the scan, so
                          channel/tag filters on a large archive may
                          return fewer entries than ``limit``.
              ``channel`` — only return conversations created on this
                            channel (api / cli / webchat / unknown).
              ``tag``     — only return conversations carrying this tag.
            """
            try:
                limit = int(request.query_params.get("limit", "50"))
            except ValueError:
                return JSONResponse(
                    {"error": "limit must be an integer"}, status_code=400
                )
            limit = max(1, min(limit, 500))
            # Optional channel + tag narrowing. The store doesn't
            # support these directly, so filter the already-loaded
            # summaries — fine because the scan is capped at `limit`
            # anyway and operators using these filters typically know
            # what they're looking for. Cap each filter at 64 chars to
            # match other filter-param rules.
            channel_filter = request.query_params.get("channel")
            if channel_filter is not None:
                if len(channel_filter) > 64:
                    return JSONResponse(
                        {"error": "channel must be 64 chars or fewer"},
                        status_code=400,
                    )
            tag_filter = request.query_params.get("tag")
            if tag_filter is not None:
                if len(tag_filter) > 64:
                    return JSONResponse(
                        {"error": "tag must be 64 chars or fewer"},
                        status_code=400,
                    )
            store = self.sessions.store
            if not store:
                return JSONResponse({"conversations": []})
            summaries = store.list_conversations(limit=limit)
            if channel_filter is not None:
                summaries = [s for s in summaries if s.channel == channel_filter]
            if tag_filter is not None:
                summaries = [s for s in summaries if tag_filter in (s.tags or [])]
            return JSONResponse(
                {
                    "conversations": [
                        {
                            "id": s.id,
                            "title": s.title,
                            "channel": s.channel,
                            "created_at": s.created_at,
                            "message_count": s.message_count,
                            "summary": s.summary,
                            # Tags were on /api/sessions but not here —
                            # two list endpoints with different shapes
                            # confused API clients. Both now expose
                            # the same shape so callers can use either.
                            "tags": list(s.tags),
                        }
                        for s in summaries
                    ]
                }
            )

        async def conversation_detail(request: Request) -> JSONResponse:
            """Load a full conversation by ID."""
            conv_id = request.path_params["conv_id"]
            store = self.sessions.store
            if not store:
                return JSONResponse({"error": "No store"}, status_code=500)
            conv = store.load(conv_id)
            if not conv:
                return JSONResponse({"error": "Not found"}, status_code=404)
            # Always emit `title` and `tags` (possibly empty) so the
            # detail shape matches the list view. `Conversation.to_dict`
            # conditionally omits these when empty — fine for storage,
            # but it forced API clients to special-case detail responses
            # vs list responses (KeyError on `data["title"]` for an
            # untitled conversation).
            d = conv.to_dict()
            d.setdefault("title", "")
            d.setdefault("tags", [])
            # Surface live routing state alongside the persisted
            # conversation so a UI that opens a single conversation
            # sees pin/affinity in one fetch — parity with
            # /api/sessions and /sessions, which started exposing
            # these fields in the recent observability round. Both
            # values are None when the session isn't actively routed
            # or pinned (most of the conversation archive).
            d["worker_id"] = self._session_workers.get(conv_id)
            d["pinned_worker_id"] = self._session_pins.get(conv_id)
            return JSONResponse(d)

        async def conversation_rename(request: Request) -> JSONResponse:
            """Rename a conversation."""
            conv_id = request.path_params["conv_id"]
            store = self.sessions.store
            if not store:
                return JSONResponse({"error": "No store"}, status_code=500)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "body must be a JSON object"}, status_code=400,
                )
            raw_title = body.get("title", "")
            if not isinstance(raw_title, str):
                return JSONResponse(
                    {"error": "title must be a string"}, status_code=400,
                )
            title = raw_title.strip()
            if not title:
                return JSONResponse({"error": "Title required"}, status_code=400)
            # Titles surface in the web UI sidebar, /api/sessions output,
            # and dispatch logs. A 10k-char title destroys layouts and
            # bloats every list response. 200 chars is generous; the
            # auto-generated title via `display_title` is ~50.
            if len(title) > 200:
                return JSONResponse(
                    {"error": "title must be 200 chars or fewer"},
                    status_code=400,
                )
            # Strip control characters (newlines, NULs, tabs). Multi-line
            # titles break list-view rendering and log readability.
            if any(ord(c) < 0x20 or ord(c) == 0x7F for c in title):
                return JSONResponse(
                    {"error": "title must not contain control characters"},
                    status_code=400,
                )
            if store.rename(conv_id, title):
                # Also patch the in-memory session if one is loaded —
                # otherwise its next save() (on a follow-up /api/ask)
                # would clobber the rename with its stale title.
                in_mem = self.sessions.get(conv_id)
                if in_mem is not None:
                    in_mem.conversation.title = title
                return JSONResponse({"id": conv_id, "title": title})
            return JSONResponse({"error": "Not found"}, status_code=404)

        async def conversation_delete(request: Request) -> JSONResponse:
            """Delete a conversation."""
            conv_id = request.path_params["conv_id"]
            store = self.sessions.store
            if not store:
                return JSONResponse({"error": "No store"}, status_code=500)
            deleted = store.delete(conv_id)
            self.sessions.remove(conv_id)
            # Clean up in-memory routing state for this session too.
            # `conversations_delete_all` already does this; the single
            # delete handler missed it and leaked an entry per
            # delete-then-recreate cycle. Not catastrophic — these
            # dicts hold small entries — but the operator-visible
            # symptom was /sessions showing a stale `worker_id` for
            # a deleted-and-recreated session_id.
            prior_worker = self._session_workers.pop(conv_id, None)
            self._session_jobs.pop(conv_id, None)
            # Close the worker-side context slot too. Without this,
            # NodeTracker accumulates ghost slots for every deleted
            # session — visible as inflated `active_sessions` and
            # `context_pressure` on /cluster/nodes, and tracked all
            # the way to pressure=1.0 after enough probes. The
            # dispatcher's context-aware routing then avoids these
            # workers even though they're genuinely idle.
            if prior_worker is not None:
                self._node_tracker.close_context_slot(prior_worker, conv_id)
            # Also drop a worker pin if one was set — a pin on a
            # deleted conversation can't be honored and would leak
            # into the persisted SessionPinStore on the next save.
            if conv_id in self._session_pins:
                self._session_pins.pop(conv_id, None)
                self.pin_store.save(self._session_pins)
            return JSONResponse({"deleted": deleted})

        async def conversations_delete_all(request: Request) -> JSONResponse:
            """Delete all conversations.

            Requires ``?confirm=yes`` to actually perform the delete.
            Without it, returns a 400 with the live count + the URL the
            caller would need to hit. This is a footgun guard: a stale
            curl in shell history or a misclicked button shouldn't
            silently wipe an operator's entire conversation archive.
            """
            store = self.sessions.store
            if not store:
                return JSONResponse({"error": "No store"}, status_code=500)
            if request.query_params.get("confirm") != "yes":
                return JSONResponse(
                    {
                        "error": (
                            "this would delete ALL conversations; "
                            "re-issue with ?confirm=yes to proceed"
                        ),
                        "would_delete": store.count,
                    },
                    status_code=400,
                )
            count = store.delete_all()
            self.sessions.clear()
            # Close every node tracker context slot, since every
            # conversation that owned one is now gone. Without this,
            # context_pressure stays inflated until the workers
            # disconnect or restart.
            for sid, worker_id in list(self._session_workers.items()):
                self._node_tracker.close_context_slot(worker_id, sid)
            self._session_workers.clear()
            # Also wipe the worker-pin map. The per-conversation
            # delete path drops pins (commit explaining the leak path
            # there), but delete-all left them in memory — next
            # /sessions/<id>/pin-worker save() re-persisted them as
            # ghost entries pointing at conversations that no longer
            # exist. Operators who used Delete-All to reset state and
            # then re-pinned a NEW session found stale entries
            # reappearing on disk.
            if self._session_pins:
                self._session_pins.clear()
                self.pin_store.save(self._session_pins)
            return JSONResponse({"deleted": count})

        async def conversation_export(request: Request) -> HTMLResponse:
            """Export a conversation to markdown / json / text / html."""
            from starlette.responses import Response

            from towel.persistence.export import (
                export_html,
                export_json,
                export_markdown,
                export_text,
            )

            conv_id = request.path_params["conv_id"]
            fmt = request.query_params.get("format", "markdown")
            # Reject unknown formats explicitly rather than silently
            # falling back to markdown — a client passing `format=evil`
            # got markdown back with no indication of the typo, which
            # made it hard to spot config errors.
            if fmt not in ("markdown", "json", "text", "html"):
                return JSONResponse(
                    {"error": "format must be one of: markdown, json, text, html"},
                    status_code=400,
                )
            store = self.sessions.store
            if not store:
                return JSONResponse({"error": "No store"}, status_code=500)
            conv = store.load(conv_id)
            if not conv:
                return JSONResponse({"error": "Not found"}, status_code=404)

            if fmt == "json":
                # Default pretty (indent=2) for human reading via
                # browser/curl. Opt into compact (`?pretty=0`) for
                # piping into jq, log shipping, or any other use that
                # cares about bytes more than readability.
                pretty_raw = request.query_params.get("pretty")
                pretty = pretty_raw not in {"0", "false", "no"}
                content = export_json(conv, pretty=pretty)
                media_type = "application/json"
                ext = "json"
            elif fmt == "text":
                content = export_text(conv)
                media_type = "text/plain"
                ext = "txt"
            elif fmt == "html":
                # export_html produces a self-contained dark-themed
                # standalone page with collaboration metadata
                # (verified-by, ensemble) surfaced in line — same
                # source the markdown export uses, just rendered.
                # Operators were already using the function via
                # test_export; the gateway route was the missing
                # piece for sharing a conversation as a single
                # file someone can open in a browser.
                content = export_html(conv, include_metadata=True)
                media_type = "text/html"
                ext = "html"
            else:
                content = export_markdown(conv, include_metadata=True)
                media_type = "text/markdown"
                ext = "md"

            # Sanitize conv_id for the Content-Disposition filename.
            # Session IDs may include `"` (the simple_ask validator
            # only rejects control chars + ≥257 chars), and an
            # unescaped quote breaks the filename="..." quoting —
            # browsers truncate at the inner quote and lose the
            # extension. Same alnum-only rule as the store path
            # sanitizer (persistence/store.py _path_for); safe by
            # construction.
            safe_conv = "".join(
                c for c in conv_id if c.isalnum() or c in "-_"
            ) or "conversation"
            filename = f"towel-{safe_conv[:16]}.{ext}"
            return Response(
                content,
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        async def simple_ask(request: Request) -> JSONResponse:
            """POST /api/ask — simple one-shot question/answer endpoint.

            Body fields (all optional except ``message``)::

                {
                  "message": "...",            // required
                  "session_id": "default",     // resume / pin context
                  "system": null,              // identity override
                  "verify": false,             // second-worker review pass
                  "ensemble": false,           // parallel fan-out + synthesis
                  "max_tokens": 256,           // 1..4096, default 256
                  "temperature": 0.7           // 0..2, default 0.7
                }

            Response: {"response": "...", "session": "...", "tokens": N, "tps": N.N}

            Either ``session`` or ``session_id`` is accepted; the rest of
            the codebase calls it ``session_id`` (path params, internal
            APIs, /api/sessions output), so clients reasonably expect that
            name here too. Previously only ``session`` worked and any
            client passing ``session_id`` was silently merged into
            ``api-default``, sharing context with every other caller.

            Much simpler than /v1/chat/completions for quick integrations.
            """
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)

            # Body must be a JSON object. Without this, an array /
            # string / null body crashes on `body.get(...)` with an
            # AttributeError that Starlette renders as plaintext
            # "Internal Server Error" HTTP 500 — not even JSON, hard
            # for API clients to handle uniformly.
            if not isinstance(body, dict):
                return JSONResponse(
                    {"error": "body must be a JSON object"}, status_code=400,
                )

            raw_message = body.get("message", "")
            if not isinstance(raw_message, str):
                return JSONResponse(
                    {"error": "message must be a string"}, status_code=400,
                )
            message = raw_message.strip()
            if not message:
                return JSONResponse({"error": "message is required"}, status_code=400)

            session_id = body.get("session_id") or body.get("session") or "api-default"
            if not isinstance(session_id, str):
                return JSONResponse(
                    {"error": "session_id must be a string"}, status_code=400,
                )
            # Strip leading/trailing whitespace so `"  sid  "` and
            # `"sid"` map to the same session. The on-disk filename
            # sanitizer drops whitespace anyway, so the two forms were
            # already pointing at the same .json file — but the
            # in-memory session dict keyed by the raw string, splitting
            # them in confusing ways (saves to one key, loads under
            # another). Strip at the boundary so they're unified.
            session_id = session_id.strip()
            if not session_id:
                # All-whitespace input falls through to the default;
                # a literal empty session_id is ambiguous, default is
                # the documented behavior.
                session_id = "api-default"
            # Same length + control-char rules as memory keys
            # (commit 1865e7d). Session IDs flow into dispatch logs,
            # filesystem paths, and URL params; absurd lengths or
            # newlines break log readability and produce broken URLs.
            if len(session_id) > 256:
                return JSONResponse(
                    {"error": "session_id must be 256 chars or fewer"},
                    status_code=400,
                )
            if any(ord(c) < 0x20 or ord(c) == 0x7F for c in session_id):
                return JSONResponse(
                    {"error": "session_id must not contain control characters"},
                    status_code=400,
                )
            system_override = body.get("system")
            # `system` is concatenated into the worker's system prompt
            # via `self.config.identity = system_override`. A non-string
            # value (number, list, dict) would either crash deeper in
            # the dispatch path or corrupt the identity field until
            # the `finally` restored it. Reject early.
            if system_override is not None and not isinstance(system_override, str):
                return JSONResponse(
                    {"error": "system must be a string"}, status_code=400,
                )
            # Opt-in two-worker verifier pass: after the primary
            # response lands, a second worker reviews (question,
            # answer) and either confirms or returns a corrected
            # version. This is genuine multi-worker collaboration on a
            # single user request — not just routing one request to
            # one worker. Trades latency for accuracy; the operator
            # toggles per-request.
            verify_raw = body.get("verify", False)
            if not isinstance(verify_raw, bool):
                return JSONResponse(
                    {"error": "verify must be true or false"}, status_code=400,
                )
            verify = bool(verify_raw)
            # Opt-in ensemble: fan the same prompt to every idle
            # worker in parallel, then arbitrate. Different shape from
            # verify: verify is sequential (draft → review), ensemble
            # is parallel (everyone answers → coordinator picks).
            # ensemble wins on latency (capped by slowest worker, not
            # sum) and on coverage (every model contributes). Costs
            # more compute. Mutually exclusive with verify — they
            # represent two different collaboration models.
            ensemble_raw = body.get("ensemble", False)
            if not isinstance(ensemble_raw, bool):
                return JSONResponse(
                    {"error": "ensemble must be true or false"}, status_code=400,
                )
            ensemble = bool(ensemble_raw)
            if ensemble and verify:
                return JSONResponse(
                    {"error": "ensemble and verify are mutually exclusive"},
                    status_code=400,
                )
            # Optional max_tokens override. Without this, every
            # /api/ask call was hard-capped at 256 — a request like
            # "explain X in detail" silently truncated at ~150 words
            # and there was no client-side knob to raise the ceiling.
            # /v1/chat/completions has always honored max_tokens; this
            # brings /api/ask to parity. Same [1, 4096] clamp the
            # OpenAI-compat path uses so a runaway 100k value can't
            # blow up the worker's generation budget.
            max_tokens_raw = body.get("max_tokens")
            api_ask_max_tokens = 256
            if max_tokens_raw is not None:
                try:
                    max_tokens_int = int(max_tokens_raw)
                except (TypeError, ValueError):
                    return JSONResponse(
                        {"error": "max_tokens must be an integer"},
                        status_code=400,
                    )
                if max_tokens_int < 1:
                    return JSONResponse(
                        {"error": "max_tokens must be ≥ 1"},
                        status_code=400,
                    )
                api_ask_max_tokens = min(max_tokens_int, 4096)
            # Optional temperature override. Same as max_tokens:
            # /v1/chat/completions honors it, /api/ask was hard-pinned
            # at the _quick_remote_infer default (0.7). Clients
            # wanting deterministic outputs (temperature=0) or more
            # creative ones (temperature=1.5+) had no way to express
            # that. Clamp to [0, 2] matching OpenAI's documented
            # range and openai-compat's parser.
            temperature_raw = body.get("temperature")
            api_ask_temperature = 0.7
            if temperature_raw is not None:
                try:
                    api_ask_temperature = float(temperature_raw)
                except (TypeError, ValueError):
                    return JSONResponse(
                        {"error": "temperature must be a number"},
                        status_code=400,
                    )
                if api_ask_temperature < 0 or api_ask_temperature > 2:
                    return JSONResponse(
                        {"error": "temperature must be between 0 and 2"},
                        status_code=400,
                    )

            session = self.sessions.get_or_create(session_id)
            session.conversation.channel = "api"
            session.conversation.add(Role.USER, message)

            # Wall-clock start so the response can report the actual
            # end-to-end request duration, separate from `total_ms`
            # (which reflects only the last worker call's duration).
            # For retry paths this lets operators see "primary took
            # 20s, retry took 15s, total 35s wall" without correlating
            # the worker's metadata to their own client timer.
            request_start = time.monotonic()

            # Per-request identity override. Passed straight through
            # to `_quick_remote_infer` rather than mutating
            # `self.config.identity` — that approach raced badly with
            # concurrent /api/ask calls using different overrides
            # (req A's worker would see req B's `system` because the
            # config field is shared mutable state).
            identity_override = system_override or None

            try:
                # Ensemble short-circuit: when the caller opted in
                # (deep-reasoning task where extra compute is worth
                # it), fan to every idle worker concurrently and let
                # the coordinator arbitrate. Bypasses the dispatcher's
                # single-worker routing entirely — every worker
                # contributes, coordinator picks. Returns early when
                # we got at least one real answer; falls through to
                # the normal single-worker path otherwise.
                ensemble_contributions: list[dict[str, Any]] = []
                ensemble_arb_mode: str = ""
                if ensemble:
                    arbitrated, ensemble_contributions, ensemble_arb_mode = (
                        await self._ensemble_dispatch(
                            session_id, message, user_session=session,
                            identity_override=identity_override,
                        )
                    )
                    # Aggregate dispatch entry so the operator sees
                    # "session X ran ensemble (mode, N workers)" in
                    # /dispatch/recent next to other dispatch events
                    # — the per-worker fan-out decisions live under
                    # the ephemeral _ens_* ids that the recent view
                    # hides by default. Always record when the user
                    # opted in, even with zero contributions, so the
                    # "all workers were busy → silently fell through to
                    # single-dispatch" case is visible. Without this,
                    # operators had no way to see that ensemble was
                    # requested and skipped — the response looked
                    # identical to a normal single-worker dispatch.
                    if self._dispatcher is not None:
                        try:
                            self._dispatcher.record_ensemble(
                                session_id=session_id,
                                contributions=ensemble_contributions,
                                arbitration_mode=ensemble_arb_mode,
                            )
                        except Exception as exc:
                            log.debug(
                                "Failed to record ensemble dispatch: %s", exc,
                            )
                    if arbitrated:
                        from towel.agent.conversation import Message as _Message
                        response = _Message(
                            role=Role.ASSISTANT,
                            content=arbitrated,
                            metadata={
                                "ensemble": True,
                                "ensemble_arbitration": ensemble_arb_mode,
                                "ensemble_contributions": ensemble_contributions,
                                "remote_worker": "ensemble",
                            },
                        )
                        session.conversation.messages.append(response)
                        worker = None
                        intent = "task"
                        # Skip the rest of the normal flow.
                        self._maybe_set_auto_title(session)
                        self.sessions.save(session_id)
                        request_total_ms = round(
                            (time.monotonic() - request_start) * 1000.0, 1,
                        )
                        ens_body: dict[str, Any] = {
                            "response": response.content,
                            "session": session_id,
                            "tokens": count_tokens_fallback(response.content),
                            "tps": 0,
                            "worker": "ensemble",
                            "ensemble": True,
                            # How the arbitration decided: "synthesis"
                            # (LLM-as-judge), "consensus" (workers
                            # agreed), "single" (only one
                            # contribution), "longest_fallback"
                            # (synthesis failed, picked longest).
                            "ensemble_arbitration": ensemble_arb_mode,
                            "ensemble_contributions": ensemble_contributions,
                            "request_total_ms": request_total_ms,
                        }
                        return JSONResponse(ens_body)
                    # Ensemble fan-out returned nothing useful — fall
                    # through to single-worker dispatch as a safety
                    # net. The contributions list still surfaces what
                    # each worker did so the operator can diagnose.

                # Route through cluster workers when available
                worker, intent = await self._route_by_role(message, session_id)
                if worker and intent == "chat":
                    # Wrap primary call so a timeout / worker error
                    # gets a retry on the alternate too, not just the
                    # empty-text case. Operationally these are the same
                    # failure: "worker X didn't give us a useful
                    # answer" — and a second worker is right there.
                    try:
                        response = await self._quick_remote_infer(
                            session_id, session, worker,
                            max_tokens=api_ask_max_tokens,
                            temperature=api_ask_temperature,
                            identity_override=identity_override,
                        )
                        primary_failed = False
                        primary_exc: Exception | None = None
                    except Exception as exc:
                        log.info(
                            "primary worker %s raised %s; will try alternate",
                            worker.id, exc,
                        )
                        primary_failed = True
                        primary_exc = exc
                        response = None  # type: ignore[assignment]
                    # When the chosen worker emitted no real text (small
                    # models like gemma-4-E2B routinely produce tool
                    # calls or empty content for a "hi"), try a SECOND
                    # qualified worker before falling back to the
                    # diagnostic placeholder. The retry walks the
                    # candidate list once, skipping the worker we just
                    # tried, so a fleet with one good worker and one
                    # flaky one still returns useful text. We don't
                    # retry on the local coordinator's own agent
                    # because in practice that path is slow enough
                    # (~85s for MLX on this fleet) to make the diagnostic
                    # placeholder a better UX than a hang.
                    needs_retry = primary_failed or (
                        response is not None
                        and (response.metadata or {}).get("empty_text_fallback")
                    )
                    if needs_retry:
                        alt = self._pick_alternate_chat_worker(exclude={worker.id})
                        if alt is None and primary_failed:
                            # No alt and primary raised — re-raise the
                            # primary exception so the caller's 500
                            # carries the useful message.
                            assert primary_exc is not None
                            raise primary_exc
                        if alt is not None:
                            log.info(
                                "worker %s %s; retrying on %s",
                                worker.id,
                                "failed (" + str(primary_exc) + ")"
                                if primary_failed
                                else "returned empty text",
                                alt.id,
                            )
                            # Record the retry as its own dispatch
                            # decision so /dispatch/recent shows
                            # operators that a fallback happened. The
                            # decision's `previous_worker_id` points
                            # to the failed primary.
                            #
                            # `cause` differentiates "primary failed
                            # (timeout/error)" from "empty text" so
                            # the notes line in /dispatch/recent
                            # reflects the real failure mode. Without
                            # it, a timed-out primary was logged as
                            # "retry after empty response" — wrong
                            # and actively misleading when triaging
                            # latency issues.
                            if self._dispatcher is not None:
                                self._dispatcher.record_retry(
                                    session_id=session_id,
                                    retry_worker=alt,
                                    original_worker_id=worker.id,
                                    intent="chat",
                                    cause=(
                                        f"primary_failed: {primary_exc}"
                                        if primary_failed and primary_exc is not None
                                        else "empty_text"
                                    ),
                                )
                            # Drop the diagnostic placeholder so the
                            # alt worker doesn't see it as its own
                            # prior assistant turn. Only the empty-text
                            # path has a placeholder to pop — the
                            # primary-failed path never appended one.
                            popped: Any = None
                            if (
                                not primary_failed
                                and session.conversation.messages
                                and session.conversation.messages[-1].role
                                == Role.ASSISTANT
                            ):
                                popped = session.conversation.messages.pop()
                            try:
                                retry_response = await self._quick_remote_infer(
                                    session_id, session, alt,
                                    max_tokens=api_ask_max_tokens,
                                    temperature=api_ask_temperature,
                                    identity_override=identity_override,
                                )
                            except Exception as retry_exc:
                                # Retry crashed (timeout, worker DC, etc.).
                                # Restore the original placeholder so the
                                # session record matches what we'll send
                                # back to the caller, then keep the
                                # original `response` if we have one,
                                # otherwise re-raise the primary exc.
                                log.warning(
                                    "retry on %s failed (%s); keeping %s",
                                    alt.id, retry_exc,
                                    "primary exception" if primary_failed
                                    else f"empty-text response from {worker.id}",
                                )
                                if popped is not None:
                                    session.conversation.messages.append(popped)
                                if primary_failed:
                                    assert primary_exc is not None
                                    raise primary_exc
                            else:
                                # Only adopt the retry if it actually
                                # produced text. If the alt worker ALSO
                                # returned empty AND we have an original
                                # response, keep that — no point flapping.
                                # If primary FAILED (no response) and the
                                # alt returned empty, adopt the empty
                                # response with fallback metadata anyway —
                                # the diagnostic placeholder is more useful
                                # than the primary's exception.
                                alt_was_empty = (retry_response.metadata or {}).get(
                                    "empty_text_fallback"
                                )
                                if (not alt_was_empty) or primary_failed:
                                    retry_response.metadata = (
                                        retry_response.metadata or {}
                                    ) | {
                                        "fallback_from_worker": worker.id,
                                        "fallback_reason": (
                                            "primary_failed" if primary_failed
                                            else "empty_text"
                                        ),
                                    }
                                    response = retry_response
                                else:
                                    # Both primary AND alt returned
                                    # empty text. This is a fleet-wide
                                    # signal — the models likely emit
                                    # tool calls for every chat-style
                                    # input, which the user experiences
                                    # as a long wait followed by the
                                    # generic placeholder. Surface a
                                    # warning at log level so the
                                    # operator notices, and tag the
                                    # response metadata so /dispatch
                                    # readers and clients can spot it.
                                    log.warning(
                                        "Dual empty-text on session %s: "
                                        "primary=%s alt=%s — both workers "
                                        "produced no parseable text. The "
                                        "models likely tool-loop on this "
                                        "prompt; consider reviewing the "
                                        "system prompt or worker quality.",
                                        session_id, worker.id, alt.id,
                                    )
                                    response.metadata = (
                                        response.metadata or {}
                                    ) | {
                                        "dual_empty_text": True,
                                        "alt_worker": alt.id,
                                    }
                elif worker:
                    response = await self._step_remote_inference(
                        session_id, session, worker
                    )
                else:
                    response = await self.agent.step(session.conversation)
                    session.conversation.messages.append(response)

                # Two-worker verifier pass (opt-in via verify=true).
                # The primary worker's answer becomes the input to a
                # SECOND worker that either confirms it or replaces it
                # with a corrected version. Genuine collaboration: two
                # workers acting on one request. Skipped when:
                # - operator didn't ask
                # - the primary failed (no answer to verify)
                # - the response is a placeholder from dual-empty
                #   (verifying an "I couldn't respond" doesn't help)
                # - no alternate worker exists
                if verify and worker is not None:
                    primary_meta = response.metadata or {}
                    has_real_answer = bool(
                        response.content
                        and not primary_meta.get("empty_text_fallback")
                    )
                    if has_real_answer:
                        final_answer, was_corrected, verifier_id = (
                            await self._verify_pass(
                                session_id, message,
                                response.content, worker.id,
                            )
                        )
                        # Aggregate dispatch entry — see record_ensemble
                        # comment for the same pattern. Always record
                        # when the user opted in: a `verifier_id=None`
                        # means "no alternate worker available" and
                        # the operator still wants to see the request
                        # was made (symmetric to ensemble: skipped).
                        if self._dispatcher is not None:
                            try:
                                self._dispatcher.record_verify(
                                    session_id=session_id,
                                    verifier_id=verifier_id,
                                    primary_id=worker.id,
                                    was_corrected=was_corrected,
                                )
                            except Exception as exc:
                                log.debug(
                                    "Failed to record verify dispatch: %s", exc,
                                )
                        if was_corrected and final_answer != response.content:
                            log.info(
                                "Verifier %s corrected primary %s answer for session %s",
                                verifier_id, worker.id, session_id,
                            )
                            # Replace the last assistant message with
                            # the corrected version so the persisted
                            # transcript reflects what the user
                            # actually got back.
                            if (
                                session.conversation.messages
                                and session.conversation.messages[-1].role
                                == Role.ASSISTANT
                            ):
                                session.conversation.messages[-1].content = final_answer
                            response.content = final_answer
                            response.metadata = (response.metadata or {}) | {
                                "verified_by": verifier_id,
                                "verifier_corrected": True,
                                "primary_worker": worker.id,
                            }
                        elif verifier_id is not None:
                            response.metadata = (response.metadata or {}) | {
                                "verified_by": verifier_id,
                                "verifier_corrected": False,
                                "primary_worker": worker.id,
                            }
                # Auto-title parity with the WS path so api-channel
                # conversations don't end up with title="" on disk —
                # which was making every /api/ask session render as a
                # blank row in the conversations list.
                self._maybe_set_auto_title(session)
                self.sessions.save(session_id)

                # Surface timing data when the worker reported it.
                # ttft_ms isn't always present (streaming-only path
                # populates it); fall back to total_ms or omit the
                # field when neither is known.
                meta = response.metadata or {}
                # Defensive: workers running pre-fix code occasionally
                # report tokens=0 even with visible content (llama-server
                # builds without `usage`, or the reasoning_content
                # substitution path before that fix shipped). Estimate
                # from the response body so /api/ask doesn't lie about
                # what got generated. Worker's reported count still
                # wins when it's non-zero.
                reported_tokens_raw = meta.get("tokens", 0)
                reported_tokens = (
                    int(reported_tokens_raw)
                    if isinstance(reported_tokens_raw, (int, float))
                    else 0
                )
                if reported_tokens == 0 and response.content:
                    reported_tokens = count_tokens_fallback(response.content)
                # tps may arrive as explicit None from a worker that
                # didn't measure (e.g. job_error before any tokens, or
                # an empty-text response). `round(None, 1)` raises
                # TypeError, which would 500 the whole /api/ask after
                # an otherwise-recoverable empty-text fallback. Coerce
                # to 0 at the boundary.
                tps_raw = meta.get("tps")
                tps_val = float(tps_raw) if isinstance(tps_raw, (int, float)) else 0.0
                body: dict[str, Any] = {
                    "response": response.content,
                    "session": session_id,
                    "tokens": reported_tokens,
                    "tps": round(tps_val, 1),
                    "worker": meta.get("remote_worker", "coordinator"),
                }
                if isinstance(meta.get("ttft_ms"), (int, float)):
                    body["ttft_ms"] = round(meta["ttft_ms"], 1)
                if isinstance(meta.get("total_ms"), (int, float)):
                    body["total_ms"] = round(meta["total_ms"], 1)
                # End-to-end wall-clock time for the whole request,
                # including any retry hops and the coordinator's own
                # work (memory injection, classification, dispatch).
                # Distinct from `total_ms`, which is only the last
                # worker call's duration — operators on the retry
                # path were seeing the alt's time and wondering why
                # their client timer was much higher.
                body["request_total_ms"] = round(
                    (time.monotonic() - request_start) * 1000.0, 1
                )
                if meta.get("empty_text_tool_call_fallback"):
                    body["fallback"] = "empty_text_tool_call"
                if meta.get("fallback_from_worker"):
                    body["fallback_from_worker"] = meta["fallback_from_worker"]
                    body["fallback_reason"] = meta.get("fallback_reason", "")
                # Surface the dual-empty-text signal — when BOTH the
                # primary and the retry returned empty (typically
                # because every worker tool-loops on this prompt
                # shape), the caller currently sees the diagnostic
                # placeholder with no hint that two workers actually
                # tried. Without this field, clients couldn't tell
                # "one slow worker returned empty" from "fleet-wide
                # tool-loop on this prompt" and operators kept asking
                # "did anything else try?". Same metadata flag the
                # WS path has carried since the dual-empty fix; just
                # mirror it into the HTTP body.
                if meta.get("dual_empty_text"):
                    body["dual_empty_text"] = True
                    if meta.get("alt_worker"):
                        body["alt_worker"] = meta["alt_worker"]
                # Surface the verifier pass so the caller can tell
                # the answer went through two-worker collaboration.
                if meta.get("verified_by"):
                    body["verified_by"] = meta["verified_by"]
                    body["verifier_corrected"] = bool(
                        meta.get("verifier_corrected", False)
                    )
                    body["primary_worker"] = meta.get(
                        "primary_worker", body.get("worker", ""),
                    )
                return JSONResponse(body)
            except Exception as e:
                # Persist the partial session even on inference failure
                # so the user message isn't lost — without this, a new
                # /api/ask session that errored before any reply showed
                # up later in /conversations as if the user had never
                # asked anything (the in-memory user turn was added at
                # line 4410, but the save() inside the try was skipped
                # when the inference raised).
                try:
                    self.sessions.save(session_id)
                except Exception as save_exc:
                    log.debug(
                        "Failed to persist session %s after inference error: %s",
                        session_id, save_exc,
                    )
                return JSONResponse({"error": _err_str(e)}, status_code=500)

        async def api_sessions(request: Request) -> JSONResponse:
            """GET /api/sessions — list active and stored sessions with tags.

            Query params:
              ``limit`` — cap on entries scanned (default 50, max 500).
              ``channel`` — narrow to sessions created on this channel.
              ``tag``     — narrow to sessions carrying this tag.

            Channel + tag filters apply *after* the scan, mirroring the
            shape /conversations exposes — the two list endpoints stay
            symmetrical so callers can switch between them without
            field-name surprises.
            """
            store = self.sessions.store
            if not store:
                return JSONResponse({"sessions": []})
            try:
                limit = int(request.query_params.get("limit", "50"))
            except ValueError:
                return JSONResponse(
                    {"error": "limit must be an integer"}, status_code=400
                )
            limit = max(1, min(limit, 500))
            # Filter validation matches /conversations: 64-char cap,
            # 400 on overlong, post-scan filtering so the response
            # shape stays uniform.
            channel_filter = request.query_params.get("channel")
            if channel_filter is not None and len(channel_filter) > 64:
                return JSONResponse(
                    {"error": "channel must be 64 chars or fewer"},
                    status_code=400,
                )
            tag_filter = request.query_params.get("tag")
            if tag_filter is not None and len(tag_filter) > 64:
                return JSONResponse(
                    {"error": "tag must be 64 chars or fewer"},
                    status_code=400,
                )
            summaries = store.list_conversations(limit=limit)
            if channel_filter is not None:
                summaries = [s for s in summaries if s.channel == channel_filter]
            if tag_filter is not None:
                summaries = [s for s in summaries if tag_filter in (s.tags or [])]
            items = []
            for s in summaries:
                # Tags now ride on the summary itself — no second
                # file-read per session needed (was 50 extra disk
                # reads per /api/sessions call before).
                # Also surface the worker_id (current affinity) and
                # pinned_worker_id so /api/sessions is a one-stop
                # answer for "what's this session's routing state?".
                # The sister /sessions endpoint exposes these for live
                # in-memory sessions; /api/sessions covers the full
                # persisted set, and operators using it to triage
                # currently had to cross-reference with /sessions
                # just to see where their sessions were pinned. Both
                # values are None when the session isn't actively
                # routed / pinned to anywhere.
                items.append(
                    {
                        "id": s.id,
                        "title": s.title,
                        "channel": s.channel,
                        "created_at": s.created_at,
                        "message_count": s.message_count,
                        "summary": s.summary,
                        "tags": list(s.tags),
                        "worker_id": self._session_workers.get(s.id),
                        "pinned_worker_id": self._session_pins.get(s.id),
                    }
                )
            return JSONResponse({"sessions": items})

        async def admin_restart(request: Request) -> JSONResponse:
            """POST /admin/restart — gracefully re-exec this process.

            Requires ``?confirm=yes`` to actually restart. Without it,
            returns a 400 — same footgun guard used by
            ``DELETE /conversations``. A stray curl in shell history
            or a misclicked automation shouldn't be one keystroke away
            from dropping all in-memory state (dispatch log, active
            sessions, in-flight worker assignments).

            The web UI's restart button passes the flag automatically.
            """
            import asyncio as _asyncio
            import os as _os
            import sys as _sys

            if request.query_params.get("confirm") != "yes":
                return JSONResponse(
                    {
                        "error": (
                            "restarting drops all in-memory state "
                            "(dispatch log, active sessions, in-flight "
                            "worker assignments); re-issue with "
                            "?confirm=yes to proceed"
                        )
                    },
                    status_code=400,
                )

            async def _do_restart() -> None:
                await _asyncio.sleep(0.3)  # let the HTTP response flush
                _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

            _asyncio.create_task(_do_restart())
            return JSONResponse({"status": "restarting"})

        # OpenAI-compatible API routes
        from towel.gateway.openai_compat import build_openai_routes  # noqa: F401

        # Pass `self` so the openai_compat handler can route through
        # the same worker dispatch as /api/ask. Without this, every
        # /v1/chat/completions call ran on the coordinator's local
        # agent regardless of fleet availability — defeating the whole
        # point of having workers.
        openai_routes = build_openai_routes(self.agent, self.config, gateway=self)

        from towel.agent.streaming_protocol import build_sse_routes
        from towel.setup_server import setup_routes

        sse_routes = build_sse_routes(self.agent, self.config)
        # The setup wizard mounts at /setup and exposes its own JSON API at
        # /api/setup/*. Reachable from the running gateway for live reconfig.
        setup_route_list = setup_routes()

        routes: list[Route | Mount] = [
            Route("/health", health),
            Route("/sessions", sessions_list),
            Route("/sessions/{session_id}/pin-worker", session_pin_worker, methods=["POST"]),
            Route("/sessions/{session_id}/pin-worker", session_unpin_worker, methods=["DELETE"]),
            Route("/workers", workers_list),
            Route("/workers/{worker_id}/state", worker_state_update, methods=["POST"]),
            Route("/workers/{worker_id}/tasks", worker_tasks_update, methods=["POST"]),
            Route("/workers/{worker_id}/upgrade", worker_self_upgrade, methods=["POST"]),
            Route("/cluster/nodes", cluster_nodes),
            Route("/cluster/handoffs", cluster_handoffs),
            Route("/cluster/idle", idle_tasks_status),
            Route("/dispatch/recent", dispatch_recent),
            Route("/dispatch/explain", dispatch_explain),
            Route("/skills", skills_list),
            Route("/memory", memory_list),
            Route("/memory", memory_create, methods=["POST"]),
            Route("/memory/stats", memory_stats),
            Route("/memory/activity", memory_activity),
            Route("/memory/recalls", memory_recalls),
            # Order matters: more-specific routes first so /memory/stats
            # and /memory/{key}/inspect don't get shadowed by /memory/{key}.
            Route("/memory/{key}/inspect", memory_inspect),
            Route("/memory/{key}/nudge", memory_nudge, methods=["POST"]),
            Route("/memory/{key}", memory_edit, methods=["PATCH"]),
            Route("/memory/{key}", memory_forget, methods=["DELETE"]),
            Route("/fleet/spawn", fleet_spawn, methods=["POST"]),
            Route("/fleet/replace-worker", fleet_replace_worker, methods=["POST"]),
            Route("/fleet/upgrade", fleet_upgrade, methods=["POST"]),
            Route("/fleet/suggest-targets", fleet_suggest_targets, methods=["POST"]),
            Route("/fleet/inventory", fleet_inventory),
            Route("/fleet/rolling-replace", fleet_rolling_replace, methods=["POST"]),
            Route("/conversations", conversations_list, methods=["GET"]),
            Route("/conversations", conversations_delete_all, methods=["DELETE"]),
            Route("/conversations/{conv_id}", conversation_detail, methods=["GET"]),
            Route("/conversations/{conv_id}", conversation_delete, methods=["DELETE"]),
            Route("/conversations/{conv_id}/rename", conversation_rename, methods=["POST"]),
            Route("/conversations/{conv_id}/export", conversation_export),
            Route("/search", search_conversations),
            Route("/admin/restart", admin_restart, methods=["POST"]),
            Route("/api/ask", simple_ask, methods=["POST"]),
            Route("/api/sessions", api_sessions, methods=["GET"]),
            *openai_routes,
            *sse_routes,
            *setup_route_list,
            Route("/", webchat),
        ]

        # Serve additional static assets if they exist (css, js, images)
        if web_dir.is_dir():
            routes.append(Mount("/static", StaticFiles(directory=str(web_dir)), name="static"))

        return Starlette(routes=routes)

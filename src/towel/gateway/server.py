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
        try:
            async for raw in ws:
                msg = json.loads(raw)
                msg_type = msg.get("type", "message")

                if msg_type == "register":
                    conn_id = msg.get("id", ws.id.hex[:12])
                    self._connections[conn_id] = ws
                    role = msg.get("role", "channel")
                    capabilities = msg.get("capabilities", {})
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
                        self._workers.heartbeat(conn_id, caps)
                        if caps:
                            self._node_tracker.update_heartbeat(conn_id, caps)
                    continue

                if msg_type == "memory_sync":
                    # Worker is sending memory mutations to the controller
                    if conn_id and self._cluster_memory:
                        mutations = msg.get("mutations", [])
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
                    session = self.sessions.get_or_create(session_id)
                    content = msg.get("content", "")
                    channel = msg.get("channel", "unknown")
                    stream = msg.get("stream", True)

                    session.conversation.add(Role.USER, content, channel=channel)

                    # ── Role-based dispatch ─────────────────────────────
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

                    # Auto-title after first exchange
                    if not session.conversation.title and len(session.conversation) >= 2:
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

                    # Persist conversation after each exchange
                    self.sessions.save(session_id)

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
            # Cancel any running tasks for this connection
            for task in self._active_tasks.values():
                if not task.done():
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

        # Determine the worker's inference mode from its capabilities
        modes = worker.capabilities.get("modes", [])
        mode = modes[0] if modes else "llama_chat"

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

        try:
            msg = await asyncio.wait_for(queue.get(), timeout=5.0)
            if msg.get("type") == "job_done":
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

        try:
            msg = await asyncio.wait_for(queue.get(), timeout=5.0)
            if msg.get("type") == "job_done":
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
        """
        candidates: list[WorkerInfo] = []
        busy_candidates: list[WorkerInfo] = []
        for w in self._workers.list():
            if w.id in exclude:
                continue
            if not w.enabled or w.draining:
                continue
            if w.busy:
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
        # Determine the worker's inference mode from its capabilities
        modes = worker.capabilities.get("modes", [])
        mode = modes[0] if modes else "llama_chat"

        # Send a minimal system prompt — empirically, some smaller
        # chat-tuned models (gemma-2B/4B variants observed on the
        # SparklesMint/k-Precision fleet) emit zero tokens when handed
        # a bare yes/no question with no system instruction at all.
        # A one-line directive is plenty to unblock them without
        # adding meaningful tokens to the prompt.
        identity = (
            getattr(self.config, "identity", "")
            or "You are a helpful assistant. Answer concisely."
        )
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
                        "temperature": 0.7,
                        "reasoning_effort": "none",
                    },
                }
            )
        )

        try:
            msg = await asyncio.wait_for(queue.get(), timeout=60.0)
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
                    text = (
                        "(The worker returned no text — "
                        "likely emitted tool calls instead. "
                        "Update the worker if this repeats.)"
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
                return response
            elif msg.get("type") == "job_error":
                raise RuntimeError(msg.get("message", "Worker failed"))
        finally:
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    def pin_session_worker(self, session_id: str, worker_id: str) -> bool:
        """Pin a session to a specific worker if that worker exists."""
        if not self._workers.get(worker_id):
            return False
        self._session_pins[session_id] = worker_id
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
                        break
                    elif msg_type == "job_error":
                        self._idle_manager.complete_task(
                            worker.id, msg.get("message", "error"), error=True
                        )
                        break
                    # job_event — ignore streaming tokens for idle tasks
            except TimeoutError:
                self._idle_manager.complete_task(worker.id, "Timed out", error=True)
            except asyncio.CancelledError:
                self._idle_manager.cancel_task(worker.id)
            finally:
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
        if delta.is_full_sync:
            # First time or structural change — send full conversation
            payload: dict[str, Any] = {
                "type": "run",
                "job_id": job_id,
                "session": session_id,
                "stream": stream,
                "conversation": conversation.to_dict(),
            }
        else:
            # Incremental: send only the delta
            payload = {
                "type": "run",
                "job_id": job_id,
                "session": session_id,
                "stream": stream,
                "conversation": conversation.to_dict(),
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
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=chunk_timeout)
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
                    return result
                elif msg_type == "job_error":
                    raise RuntimeError(msg.get("message", "Remote worker failed"))
        finally:
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    async def _step_remote_inference(
        self, session_id: str, session: Any, worker: WorkerInfo
    ) -> Any:
        """Run the local tool loop while outsourcing each generation pass."""
        total_tokens = 0
        last_metadata: dict[str, Any] = {"remote_worker": worker.id}
        remaining_text = ""
        # Coordinator-measured start, used as a fallback when the
        # worker reports no total_ms (same rationale as _quick_remote_infer).
        coord_start = time.monotonic()

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

        await ws.send(
            json.dumps(
                AgentEvent.complete(
                    remaining_text or "I've reached my tool execution limit for this turn.",
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
                    "conversation": session.conversation.to_dict(),
                }
            )
        )

        try:
            while True:
                msg = await queue.get()
                msg_type = msg.get("type")
                if msg_type == "job_event":
                    event = msg.get("event", {})
                    await ws.send(json.dumps(event))
                elif msg_type == "job_done":
                    conversation = msg.get("conversation")
                    if conversation:
                        session.conversation = session.conversation.from_dict(conversation)
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
                    break
        finally:
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
        try:
            await worker.ws.send(
                json.dumps(
                    {
                        "type": "run",
                        "job_id": job_id,
                        "session": session_id,
                        "stream": True,
                        "conversation": session.conversation.to_dict(),
                    }
                )
            )
            while True:
                msg = await queue.get()
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
                    break
                elif msg_type == "job_error":
                    raise RuntimeError(msg.get("message", "remote worker failed"))
        finally:
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

        async def sessions_list(_request: Any) -> JSONResponse:
            return JSONResponse(
                {
                    "sessions": [
                        {
                            "id": s.id,
                            "channel": s.conversation.channel,
                            "messages": len(s.conversation),
                            "created_at": s.conversation.created_at.isoformat(),
                            "worker_id": self._session_workers.get(s.id),
                            "pinned_worker_id": self._session_pins.get(s.id),
                        }
                        for s in self.sessions.all()
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

            task_names = body.get("tasks")
            if task_names is None:
                return JSONResponse({"error": "tasks list required"}, status_code=400)

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
            return JSONResponse(self._node_tracker.to_dict())

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
            intent = request.query_params.get("intent", "task")
            task_type_raw = request.query_params.get("task_type")
            task_type: TaskType | None = None
            if task_type_raw:
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
            only_degraded = request.query_params.get("only_degraded") in {"1", "true"}
            only_affinity_missed = request.query_params.get("only_affinity_missed") in {"1", "true"}

            assert self._dispatcher is not None
            entries = [d.to_dict() for d in self._dispatcher.history()]
            if reason:
                entries = [e for e in entries if e.get("reason") == reason]
            if worker_filter:
                entries = [e for e in entries if e.get("worker_id") == worker_filter]
            if session_filter:
                entries = [e for e in entries if e.get("session_id") == session_filter]
            if only_degraded:
                entries = [e for e in entries if e.get("quality_degraded")]
            if only_affinity_missed:
                entries = [e for e in entries if e.get("affinity_missed")]

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
                from datetime import datetime, UTC
                try:
                    oldest_ts = datetime.fromisoformat(
                        full_history[0].to_dict().get("ts", "")
                    )
                    age_s = (datetime.now(UTC) - oldest_ts).total_seconds()
                    log_status["oldest_age_seconds"] = int(age_s)
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
                    "memories": [e.to_dict() for e in entries[:limit]],
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
            key_filter = request.query_params.get("key") or None
            try:
                rows = memory.recent_recalls(
                    limit=limit, since_hours=since_hours, key_filter=key_filter,
                )
            except Exception as exc:
                log.exception("memory.recent_recalls failed: %s", exc)
                return JSONResponse({"error": str(exc)}, status_code=500)
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
                return JSONResponse({"error": str(exc)}, status_code=500)
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
                    "entry": entry.to_dict(),
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
                    "recent_unvalidated": [e.to_dict() for e in pending],
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
            model = (body.get("model") or "").strip()
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
            launcher_url = (body.get("launcher_url") or "").strip()
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

            launcher_token = body.get("launcher_token") or ""
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
                body = {}
            if not isinstance(body, dict):
                body = {}
            strategy = (body.get("strategy") or "pip").strip()

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
            target_id = (body.get("target_worker_id") or "").strip()
            if not target_id:
                return JSONResponse(
                    {"error": "target_worker_id is required"}, status_code=400
                )
            launcher_url = (body.get("launcher_url") or "").strip()
            if not launcher_url:
                return JSONResponse(
                    {"error": "launcher_url is required"}, status_code=400
                )
            worker_payload = body.get("worker") or {}
            if not isinstance(worker_payload, dict):
                return JSONResponse(
                    {"error": "worker must be a JSON object"}, status_code=400
                )

            launcher_token = body.get("launcher_token") or ""
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
            shared_token = body.get("launcher_token") or ""
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
                tid = (target.get("target_worker_id") or "").strip()
                lurl = (target.get("launcher_url") or "").strip()
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
                tok = target.get("launcher_token") or shared_token
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
            launcher_url = (body.get("launcher_url") or "").strip()
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

            launcher_token = body.get("launcher_token") or ""
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
            key = body.get("key")
            content = body.get("content")
            if not (isinstance(key, str) and key.strip()):
                return JSONResponse({"error": "key required"}, status_code=400)
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
                return JSONResponse({"error": str(exc)}, status_code=500)
            return JSONResponse(entry.to_dict(), status_code=201)

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
                return JSONResponse({"error": str(exc)}, status_code=500)
            entry = memory.recall(key)
            return JSONResponse(entry.to_dict())

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
                    saved_links_warning = False  # placeholder for telemetry
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
                return JSONResponse({"error": str(exc)}, status_code=500)
            return JSONResponse(updated.to_dict())

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
                return JSONResponse({"error": str(exc)}, status_code=500)
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

        async def cluster_handoffs(_request: Any) -> JSONResponse:
            return JSONResponse(
                {
                    "stats": self._handoff_manager.stats(),
                    "recent": self._handoff_manager.recent_handoffs(),
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
            return JSONResponse(updated.to_dict())

        async def session_pin_worker(request: Request) -> JSONResponse:
            session_id = request.path_params["session_id"]
            try:
                body = await request.json()
                worker_id = body.get("worker_id", "").strip()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
            if not worker_id:
                return JSONResponse({"error": "worker_id required"}, status_code=400)
            if not self.pin_session_worker(session_id, worker_id):
                return JSONResponse({"error": "Worker not found"}, status_code=404)
            return JSONResponse(
                {
                    "session_id": session_id,
                    "worker_id": worker_id,
                    "pinned": True,
                }
            )

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
            if not query:
                return JSONResponse({"error": "Missing ?q= parameter"}, status_code=400)
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
            store = self.sessions.store
            if not store:
                return JSONResponse({"results": []})
            results = store.search(query, limit=limit)
            return JSONResponse(
                {
                    "query": query,
                    "results": [
                        {
                            "conversation_id": r.conversation_id,
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
            """List all persisted conversations (not just active ones)."""
            try:
                limit = int(request.query_params.get("limit", "50"))
            except ValueError:
                return JSONResponse(
                    {"error": "limit must be an integer"}, status_code=400
                )
            limit = max(1, min(limit, 500))
            store = self.sessions.store
            if not store:
                return JSONResponse({"conversations": []})
            summaries = store.list_conversations(limit=limit)
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
            return JSONResponse(conv.to_dict())

        async def conversation_rename(request: Request) -> JSONResponse:
            """Rename a conversation."""
            conv_id = request.path_params["conv_id"]
            store = self.sessions.store
            if not store:
                return JSONResponse({"error": "No store"}, status_code=500)
            try:
                body = await request.json()
                title = body.get("title", "").strip()
            except Exception:
                return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
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
            self._session_workers.clear()
            return JSONResponse({"deleted": count})

        async def conversation_export(request: Request) -> HTMLResponse:
            """Export a conversation to markdown."""
            from starlette.responses import Response

            from towel.persistence.export import export_json, export_markdown, export_text

            conv_id = request.path_params["conv_id"]
            fmt = request.query_params.get("format", "markdown")
            # Reject unknown formats explicitly rather than silently
            # falling back to markdown — a client passing `format=evil`
            # got markdown back with no indication of the typo, which
            # made it hard to spot config errors.
            if fmt not in ("markdown", "json", "text"):
                return JSONResponse(
                    {"error": "format must be one of: markdown, json, text"},
                    status_code=400,
                )
            store = self.sessions.store
            if not store:
                return JSONResponse({"error": "No store"}, status_code=500)
            conv = store.load(conv_id)
            if not conv:
                return JSONResponse({"error": "Not found"}, status_code=404)

            if fmt == "json":
                content = export_json(conv)
                media_type = "application/json"
                ext = "json"
            elif fmt == "text":
                content = export_text(conv)
                media_type = "text/plain"
                ext = "txt"
            else:
                content = export_markdown(conv, include_metadata=True)
                media_type = "text/markdown"
                ext = "md"

            filename = f"towel-{conv_id[:16]}.{ext}"
            return Response(
                content,
                media_type=media_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        async def simple_ask(request: Request) -> JSONResponse:
            """POST /api/ask — simple one-shot question/answer endpoint.

            Body: {"message": "...", "session_id": "default", "system": null}
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

            message = body.get("message", "").strip()
            if not message:
                return JSONResponse({"error": "message is required"}, status_code=400)

            session_id = body.get("session_id") or body.get("session") or "api-default"
            system_override = body.get("system")

            session = self.sessions.get_or_create(session_id)
            session.conversation.channel = "api"
            session.conversation.add(Role.USER, message)

            # Temporary system prompt override
            old_identity = self.config.identity
            if system_override:
                self.config.identity = system_override
                self.agent.config = self.config

            try:
                # Route through cluster workers when available
                worker, intent = await self._route_by_role(message, session_id)
                if worker and intent == "chat":
                    response = await self._quick_remote_infer(
                        session_id, session, worker, max_tokens=256
                    )
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
                    if (response.metadata or {}).get("empty_text_fallback"):
                        alt = self._pick_alternate_chat_worker(exclude={worker.id})
                        if alt is not None:
                            log.info(
                                "worker %s returned empty text; retrying on %s",
                                worker.id, alt.id,
                            )
                            # Drop the diagnostic placeholder so the
                            # alt worker doesn't see it as its own
                            # prior assistant turn.
                            if session.conversation.messages and (
                                session.conversation.messages[-1].role == Role.ASSISTANT
                            ):
                                session.conversation.messages.pop()
                            retry_response = await self._quick_remote_infer(
                                session_id, session, alt, max_tokens=256
                            )
                            # Only adopt the retry if it actually
                            # produced text. If the alt worker ALSO
                            # returned empty, keep the original
                            # diagnostic (no point flapping).
                            if not (retry_response.metadata or {}).get(
                                "empty_text_fallback"
                            ):
                                retry_response.metadata = (
                                    retry_response.metadata or {}
                                ) | {
                                    "fallback_from_worker": worker.id,
                                    "fallback_reason": "empty_text",
                                }
                                response = retry_response
                elif worker:
                    response = await self._step_remote_inference(
                        session_id, session, worker
                    )
                else:
                    response = await self.agent.step(session.conversation)
                    session.conversation.messages.append(response)
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
                reported_tokens = meta.get("tokens", 0)
                if reported_tokens == 0 and response.content:
                    reported_tokens = count_tokens_fallback(response.content)
                body: dict[str, Any] = {
                    "response": response.content,
                    "session": session_id,
                    "tokens": reported_tokens,
                    "tps": round(meta.get("tps", 0), 1),
                    "worker": meta.get("remote_worker", "coordinator"),
                }
                if isinstance(meta.get("ttft_ms"), (int, float)):
                    body["ttft_ms"] = round(meta["ttft_ms"], 1)
                if isinstance(meta.get("total_ms"), (int, float)):
                    body["total_ms"] = round(meta["total_ms"], 1)
                if meta.get("empty_text_tool_call_fallback"):
                    body["fallback"] = "empty_text_tool_call"
                if meta.get("fallback_from_worker"):
                    body["fallback_from_worker"] = meta["fallback_from_worker"]
                    body["fallback_reason"] = meta.get("fallback_reason", "")
                return JSONResponse(body)
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            finally:
                if system_override:
                    self.config.identity = old_identity
                    self.agent.config = self.config

        async def api_sessions(request: Request) -> JSONResponse:
            """GET /api/sessions — list active and stored sessions with tags."""
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
            summaries = store.list_conversations(limit=limit)
            items = []
            for s in summaries:
                # Tags now ride on the summary itself — no second
                # file-read per session needed (was 50 extra disk
                # reads per /api/sessions call before).
                items.append(
                    {
                        "id": s.id,
                        "title": s.title,
                        "channel": s.channel,
                        "created_at": s.created_at,
                        "message_count": s.message_count,
                        "summary": s.summary,
                        "tags": list(s.tags),
                    }
                )
            return JSONResponse({"sessions": items})

        async def admin_restart(_request: Any) -> JSONResponse:
            """POST /admin/restart — gracefully re-exec this process."""
            import asyncio as _asyncio
            import os as _os
            import sys as _sys

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

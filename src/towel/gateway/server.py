"""Gateway server — the central nervous system of Towel.

Handles WebSocket connections from channels, nodes, and the web UI.
Routes messages to the agent runtime and streams responses back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from towel.gateway.handoff import HandoffManager, HandoffReason
from towel.gateway.sessions import SessionManager
from towel.gateway.workers import WorkerInfo, WorkerRegistry
from towel.memory.cluster import ClusterMemorySync
from towel.memory.store import MemoryStore
from towel.nodes.roles import NodeRole, assign_roles, best_node_for_role, classify_message_intent
from towel.nodes.tracker import NodeTracker
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore

log = logging.getLogger("towel.gateway")


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

    def __post_init__(self) -> None:
        self._session_pins = self.pin_store.load()
        self._worker_states = self.worker_state_store.load()
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

            self._mdns_advertiser = TowelServiceAdvertiser(port=gw.port)
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
        try:
            await http_server.serve()
        finally:
            reaper.cancel()
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
                        # Auto-assign roles based on hardware capabilities
                        roles = assign_roles(capabilities)
                        self._node_roles[conn_id] = roles
                        log.info(
                            "Worker %s assigned roles: %s",
                            conn_id,
                            ", ".join(str(r) for r in roles),
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
                # Trigger handoffs for sessions on this worker before unregistering
                sessions_to_handoff = self._handoff_manager.sessions_needing_handoff(
                    conn_id, self._session_workers
                )
                for sid in sessions_to_handoff:
                    self._handoff_manager.plan_handoff(
                        sid, conn_id, HandoffReason.WORKER_DISCONNECTED
                    )
                    # Clear affinity so _select_worker picks a new one
                    self._session_workers.pop(sid, None)
                    # Complete handoff — next request will land on a new worker
                    self._handoff_manager.complete_handoff(sid, success=True)

                self._workers.unregister(conn_id)
                self._node_tracker.unregister(conn_id)
                self._node_roles.pop(conn_id, None)
                self._context_sync.clear_worker(conn_id)
            # Cancel any running tasks for this connection
            for task in self._active_tasks.values():
                if not task.done():
                    task.cancel()

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
            }
            nodes.append(nd)
        return nodes

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
        from towel.agent.conversation import Conversation, Message

        classify_prompt = (
            "Classify this user message. Reply with exactly one word:\n"
            "- 'chat' if it's casual conversation, greetings, acknowledgements, small talk\n"
            "- 'tool' if it requires fetching URLs, web search, or running commands\n"
            "- 'task' if it's a question, instruction, coding, analysis, or creative request\n\n"
            f"Message: {message[:400]}\n\nClassification:"
        )
        conv = Conversation(messages=[Message(role=Role.USER, content=classify_prompt)])

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

    async def _route_by_role(
        self, message: str, session_id: str
    ) -> tuple[WorkerInfo | None, str]:
        """Route a message to the best worker based on roles and intent.

        Returns (worker, intent) where intent is 'chat', 'tool', or 'task'.
        Worker is None when the coordinator should handle it directly.

        Flow:
        1. Check session pin / affinity
        2. Quick heuristic classification (free, no LLM)
        3. If ambiguous, classify on cheapest CLASSIFIER node
        4. Route based on intent:
           - chat → cheapest CLASSIFIER node (lightweight infer, no tools)
           - tool → TOOL_WORKER node (full agent loop with tools)
           - task → best INFERENCE node (full agent loop)
        """
        # Respect explicit pins
        pinned_id = self._session_pins.get(session_id)
        if pinned_id:
            worker = self._workers.get(pinned_id)
            if worker and worker.enabled and not worker.busy:
                return worker, "task"

        # Session affinity — if a worker already has this context, prefer it
        affinity_id = self._session_workers.get(session_id)
        if affinity_id:
            affinity_worker = self._workers.get(affinity_id)
            if affinity_worker and affinity_worker.enabled and not affinity_worker.busy:
                node = self._node_tracker.get(affinity_id)
                if node and node.get_context_slot(session_id) is not None:
                    return affinity_worker, "task"

        # Step 1: Quick local heuristic (no inference cost)
        intent = classify_message_intent(message)

        # Step 2: If ambiguous, use the cheapest classifier worker
        if intent is None:
            classifier = self._worker_for_role(NodeRole.CLASSIFIER, session_id)
            if classifier:
                intent = await self._classify_on_worker(message, classifier)
            else:
                # No classifier workers — fall back to coordinator
                intent = "task"

        log.debug("Route: intent=%s message=%r", intent, message[:60])

        # Step 3: Route based on intent
        if intent == "chat":
            # Chat uses quick infer on the best inference node (fast t/s matters)
            worker = self._worker_for_role(NodeRole.INFERENCE, session_id)
            if worker:
                self._session_workers[session_id] = worker.id
                return worker, "chat"
            # No inference node — fall through to coordinator

        elif intent == "tool":
            # Tool requests go to tool-capable worker
            worker = self._worker_for_role(NodeRole.TOOL_WORKER, session_id)
            if worker:
                self._session_workers[session_id] = worker.id
                return worker, "tool"
            # No tool worker — try inference node, coordinator handles tools

        # intent == "task" or no suitable specialized worker found
        worker = self._worker_for_role(NodeRole.INFERENCE, session_id)
        if worker:
            self._session_workers[session_id] = worker.id
            return worker, "task"

        # Last resort — any available worker
        worker = self._worker_for_role(NodeRole.GENERAL, session_id)
        if worker:
            self._session_workers[session_id] = worker.id
            return worker, intent or "task"

        return None, intent or "task"  # No workers → coordinator handles it

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

        # Send as "infer" with capped tokens — worker uses generate_from_request
        messages = [
            {"role": m.role.value, "content": m.content}
            for m in session.conversation.messages
        ]
        # Determine the worker's inference mode from its capabilities
        modes = worker.capabilities.get("modes", [])
        mode = modes[0] if modes else "llama_chat"

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
                        "system": "",
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
                text = msg.get("result", {}).get("text", "")
                response = Message(
                    role=Role.ASSISTANT,
                    content=text,
                    metadata={"remote_worker": worker.id, "quick_infer": True},
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
        """Persist current worker operational state."""
        current = self.worker_state_store.load()
        current.update(self._workers.state_snapshot())
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

            # Try to pre-select a new worker
            new_worker = self._select_worker(sid, estimated_tokens=token_estimate)
            if new_worker:
                self._handoff_manager.assign_target(sid, new_worker.id)
                # Open context slot on the new node
                self._node_tracker.open_context_slot(new_worker.id, sid, token_estimate)
                self._handoff_manager.complete_handoff(sid, success=True)
            else:
                self._handoff_manager.complete_handoff(
                    sid, success=False, error="No suitable replacement worker available"
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

        await worker.ws.send(json.dumps(payload))

        accumulated_text = ""
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=120.0)
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
                        result = {"text": resp.get("content", ""), "metadata": resp.get("metadata", {})}
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

                response = Message(
                    role=Role.ASSISTANT,
                    content=text,
                    metadata=last_metadata | {"tokens": total_tokens},
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
                        "gateway_ws": f"ws://{self.config.gateway.host}:{self.config.gateway.port}",
                        "gateway_http": f"http://{self.config.gateway.host}:{self.config.gateway.port + 1}",
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
            for worker in self._workers.matching():
                wd = worker.to_dict()
                wd["roles"] = [str(r) for r in self._node_roles.get(worker.id, [])]
                workers_data.append(wd)
            return JSONResponse(
                {
                    "workers": workers_data,
                    "requirements": self._desired_worker_capabilities(),
                    "pins": dict(self._session_pins),
                }
            )

        async def cluster_nodes(_request: Any) -> JSONResponse:
            return JSONResponse(self._node_tracker.to_dict())

        async def cluster_handoffs(_request: Any) -> JSONResponse:
            return JSONResponse(
                {
                    "stats": self._handoff_manager.stats(),
                    "recent": self._handoff_manager.recent_handoffs(),
                }
            )

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

            if enabled is not None:
                self._workers.set_enabled(worker_id, bool(enabled))
            if draining is not None:
                self._workers.set_draining(worker_id, bool(draining))
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
            limit = int(request.query_params.get("limit", "20"))
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
            limit = int(request.query_params.get("limit", "50"))
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

        async def conversation_export(request: Request) -> HTMLResponse:
            """Export a conversation to markdown."""
            from starlette.responses import Response

            from towel.persistence.export import export_json, export_markdown, export_text

            conv_id = request.path_params["conv_id"]
            fmt = request.query_params.get("format", "markdown")
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

            Body: {"message": "...", "session": "default", "system": null}
            Response: {"response": "...", "session": "...", "tokens": N, "tps": N.N}

            Much simpler than /v1/chat/completions for quick integrations.
            """
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)

            message = body.get("message", "").strip()
            if not message:
                return JSONResponse({"error": "message is required"}, status_code=400)

            session_id = body.get("session", "api-default")
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
                elif worker:
                    response = await self._step_remote_inference(
                        session_id, session, worker
                    )
                else:
                    response = await self.agent.step(session.conversation)
                    session.conversation.messages.append(response)
                self.sessions.save(session_id)

                return JSONResponse(
                    {
                        "response": response.content,
                        "session": session_id,
                        "tokens": response.metadata.get("tokens", 0),
                        "tps": round(response.metadata.get("tps", 0), 1),
                        "worker": response.metadata.get("remote_worker", "coordinator"),
                    }
                )
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
            summaries = store.list_conversations(limit=50)
            items = []
            for s in summaries:
                item: dict[str, Any] = {
                    "id": s.id,
                    "title": s.title,
                    "channel": s.channel,
                    "created_at": s.created_at,
                    "message_count": s.message_count,
                    "summary": s.summary,
                }
                # Load tags
                try:
                    import json as _json

                    data = _json.loads(store._path_for(s.id).read_text(encoding="utf-8"))
                    item["tags"] = data.get("tags", [])
                except Exception:
                    item["tags"] = []
                items.append(item)
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
        from towel.gateway.openai_compat import build_openai_routes

        openai_routes = build_openai_routes(self.agent, self.config)

        from towel.agent.streaming_protocol import build_sse_routes

        sse_routes = build_sse_routes(self.agent, self.config)

        routes: list[Route | Mount] = [
            Route("/health", health),
            Route("/sessions", sessions_list),
            Route("/sessions/{session_id}/pin-worker", session_pin_worker, methods=["POST"]),
            Route("/sessions/{session_id}/pin-worker", session_unpin_worker, methods=["DELETE"]),
            Route("/workers", workers_list),
            Route("/workers/{worker_id}/state", worker_state_update, methods=["POST"]),
            Route("/cluster/nodes", cluster_nodes),
            Route("/cluster/handoffs", cluster_handoffs),
            Route("/conversations", conversations_list),
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
            Route("/", webchat),
        ]

        # Serve additional static assets if they exist (css, js, images)
        if web_dir.is_dir():
            routes.append(Mount("/static", StaticFiles(directory=str(web_dir)), name="static"))

        return Starlette(routes=routes)

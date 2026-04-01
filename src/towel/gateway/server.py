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

from towel.agent.conversation import Role
from towel.agent.events import AgentEvent
from towel.agent.runtime import AgentRuntime
from towel.agent.runtime import MAX_TOOL_ITERATIONS, format_tool_feedback, tool_result_is_error
from towel.agent.tool_parser import parse_tool_calls
from towel.config import TowelConfig
from towel.gateway.sessions import SessionManager
from towel.gateway.workers import WorkerInfo, WorkerRegistry
from towel.persistence.store import ConversationStore
from towel.persistence.session_pins import SessionPinStore
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
    _job_queues: dict[str, asyncio.Queue[dict[str, Any]]] = field(default_factory=dict)
    _session_workers: dict[str, str] = field(default_factory=dict)
    _session_pins: dict[str, str] = field(default_factory=dict)
    _session_jobs: dict[str, str] = field(default_factory=dict)
    _worker_states: dict[str, dict[str, bool]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._session_pins = self.pin_store.load()
        self._worker_states = self.worker_state_store.load()

    async def start(self) -> None:
        """Start the gateway (WebSocket + HTTP)."""
        gw = self.config.gateway

        # Start WebSocket server
        self._ws_server = await websockets.serve(
            self._handle_ws,
            gw.host,
            gw.port,
        )
        log.info(f"WebSocket listening on ws://{gw.host}:{gw.port}")

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

        await http_server.serve()

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
                    continue

                if msg_type == "heartbeat":
                    if conn_id and self._workers.get(conn_id):
                        self._workers.heartbeat(conn_id, msg.get("capabilities"))
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

                    worker = self._select_worker(session_id)
                    if stream:
                        # Run streaming in a task so cancel messages can be received
                        if worker:
                            task = asyncio.create_task(
                                self._stream_remote_inference(ws, session_id, session, worker)
                            )
                        else:
                            task = asyncio.create_task(self._stream_response(ws, session_id, session))
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
                        if worker:
                            response = await self._step_remote_inference(session_id, session, worker)
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
                self._workers.unregister(conn_id)
                for session_id, worker_id in list(self._session_workers.items()):
                    if worker_id == conn_id:
                        self._session_workers.pop(session_id, None)
            # Cancel any running tasks for this connection
            for task in self._active_tasks.values():
                if not task.done():
                    task.cancel()

    async def _stream_response(self, ws: ServerConnection, session_id: str, session: Any) -> None:
        """Stream agent response events to the WebSocket."""
        async for event in self.agent.step_streaming(session.conversation):
            await ws.send(json.dumps(event.to_ws_message(session_id)))

    def _select_worker(self, session_id: str) -> WorkerInfo | None:
        """Choose a worker for this session, preserving affinity when possible."""
        preferred_id = self._session_pins.get(session_id) or self._session_workers.get(session_id)
        worker = self._workers.acquire(
            preferred_id=preferred_id,
            requirements=self._desired_worker_capabilities(),
        )
        if worker:
            self._session_workers[session_id] = worker.id
        return worker

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

    def _desired_worker_capabilities(self) -> dict[str, Any]:
        """Describe the worker shape that best matches this controller runtime."""
        backend = "claude" if self.agent.__class__.__name__ == "ClaudeCodeRuntime" else "mlx"
        mode = "anthropic_messages" if backend == "claude" else "mlx_prompt"
        return {
            "backend": backend,
            "model": getattr(self.config.model, "name", ""),
            "mode": mode,
            "tools": False,
        }

    async def _cancel_remote_job(self, session_id: str) -> None:
        job_id = self._session_jobs.get(session_id)
        worker_id = self._session_workers.get(session_id)
        if not job_id or not worker_id:
            return
        worker = self._workers.get(worker_id)
        if not worker:
            return
        await worker.ws.send(json.dumps({"type": "cancel_job", "job_id": job_id, "session": session_id}))

    async def _remote_generate(
        self,
        session_id: str,
        conversation: Any,
        worker: WorkerInfo,
        *,
        stream: bool,
        client_ws: ServerConnection | None = None,
    ) -> dict[str, Any]:
        """Run one inference pass on a remote worker from a controller-built payload."""
        build_request = getattr(self.agent, "build_inference_request", None)
        if not callable(build_request):
            raise RuntimeError("Agent runtime does not support remote inference requests")

        request = build_request(conversation)
        job_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._job_queues[job_id] = queue
        self._session_jobs[session_id] = job_id
        self._workers.assign(worker.id, job_id, session_id)

        await worker.ws.send(
            json.dumps(
                {
                    "type": "infer",
                    "job_id": job_id,
                    "session": session_id,
                    "stream": stream,
                    "request": request,
                }
            )
        )

        try:
            while True:
                msg = await queue.get()
                msg_type = msg.get("type")
                if msg_type == "job_event":
                    event = msg.get("event", {})
                    if client_ws is not None:
                        await client_ws.send(json.dumps(event))
                elif msg_type == "job_done":
                    return msg.get("result", {})
                elif msg_type == "job_error":
                    raise RuntimeError(msg.get("message", "Remote worker failed"))
        finally:
            self._job_queues.pop(job_id, None)
            self._session_jobs.pop(session_id, None)
            self._workers.release(worker.id)

    async def _step_remote_inference(self, session_id: str, session: Any, worker: WorkerInfo) -> Any:
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
                await ws.send(json.dumps(AgentEvent.tool_call(tc.name, tc.arguments).to_ws_message(session_id)))
                try:
                    tool_result = await self.agent.skills.execute_tool(tc.name, tc.arguments)
                    result_str = tool_result if isinstance(tool_result, str) else str(tool_result)
                    is_error = tool_result_is_error(result_str)
                except Exception as exc:
                    result_str = f"Error executing {tc.name}: {exc}"
                    is_error = True
                    log.error(result_str)

                await ws.send(json.dumps(AgentEvent.tool_result(tc.name, result_str).to_ws_message(session_id)))
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
                    metadata={"tokens": total_tokens, "remote_worker": worker.id, "max_iterations": True},
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
            return JSONResponse(
                {
                    "status": "hoopy",
                    "version": "0.1.0",
                    "motto": "Don't Panic.",
                    "connections": len(self._connections),
                    "sessions": len(self.sessions),
                    "workers": self._workers.stats(),
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
            return JSONResponse(
                {
                    "workers": [worker.to_dict() for worker in self._workers.matching()],
                    "requirements": self._desired_worker_capabilities(),
                    "pins": dict(self._session_pins),
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
                response = await self.agent.step(session.conversation)
                self.sessions.save(session_id)

                return JSONResponse(
                    {
                        "response": response.content,
                        "session": session_id,
                        "tokens": response.metadata.get("tokens", 0),
                        "tps": round(response.metadata.get("tps", 0), 1),
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
            Route("/conversations", conversations_list),
            Route("/conversations/{conv_id}", conversation_detail, methods=["GET"]),
            Route("/conversations/{conv_id}", conversation_delete, methods=["DELETE"]),
            Route("/conversations/{conv_id}/rename", conversation_rename, methods=["POST"]),
            Route("/conversations/{conv_id}/export", conversation_export),
            Route("/search", search_conversations),
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

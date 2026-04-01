"""Gateway server — the central nervous system of Towel.

Handles WebSocket connections from channels, nodes, and the web UI.
Routes messages to the agent runtime and streams responses back.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig
from towel.gateway.sessions import SessionManager
from towel.persistence.store import ConversationStore

log = logging.getLogger("towel.gateway")


@dataclass
class GatewayServer:
    """WebSocket + HTTP gateway."""

    config: TowelConfig
    agent: AgentRuntime
    sessions: SessionManager = field(
        default_factory=lambda: SessionManager(store=ConversationStore())
    )
    _ws_server: Server | None = None
    _connections: dict[str, ServerConnection] = field(default_factory=dict)
    _active_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)

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
                    await ws.send(
                        json.dumps(
                            {
                                "type": "registered",
                                "id": conn_id,
                                "motto": "Don't Panic.",
                            }
                        )
                    )
                    continue

                if msg_type == "cancel":
                    session_id = msg.get("session", "default")
                    self.agent.cancel()
                    # Also cancel the running task if any
                    task = self._active_tasks.pop(session_id, None)
                    if task and not task.done():
                        task.cancel()
                    log.info(f"Cancelled generation for session {session_id}")
                    continue

                if msg_type == "message":
                    session_id = msg.get("session", "default")
                    session = self.sessions.get_or_create(session_id)
                    content = msg.get("content", "")
                    channel = msg.get("channel", "unknown")
                    stream = msg.get("stream", True)

                    session.conversation.add(Role.USER, content, channel=channel)

                    if stream:
                        # Run streaming in a task so cancel messages can be received
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
            # Cancel any running tasks for this connection
            for task in self._active_tasks.values():
                if not task.done():
                    task.cancel()

    async def _stream_response(self, ws: ServerConnection, session_id: str, session: Any) -> None:
        """Stream agent response events to the WebSocket."""
        async for event in self.agent.step_streaming(session.conversation):
            await ws.send(json.dumps(event.to_ws_message(session_id)))

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
                        }
                        for s in self.sessions.all()
                    ]
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

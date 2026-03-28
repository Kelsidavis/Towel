"""Webhook channel — receive messages via HTTP POST, reply with AI responses.

The simplest way to integrate Towel with external services.
Runs a lightweight HTTP server that accepts POST requests and
returns AI responses. Works with Slack, Discord bots, Zapier,
iOS Shortcuts, curl, or any HTTP client.

Usage:
    towel webhook                    # start on port 18750
    towel webhook --port 9000       # custom port
    towel webhook --token secret    # require auth token

API:
    POST /message
    {"text": "hello", "session": "optional-id"}
    → {"response": "...", "session": "...", "tokens": N}

    POST /message with Authorization: Bearer <token>
    (if --token is set)

    GET /health
    → {"status": "ok", "channel": "webhook"}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from towel.channels.base import Channel

log = logging.getLogger("towel.channels.webhook")


class WebhookChannel(Channel):
    """HTTP webhook channel — POST messages, get AI responses."""

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:18742",
        port: int = 18750,
        host: str = "127.0.0.1",
        token: str | None = None,
    ) -> None:
        super().__init__(gateway_url)
        self.port = port
        self.host = host
        self.token = token

    @property
    def name(self) -> str:
        return "webhook"

    async def listen(self) -> None:
        """Start the webhook HTTP server."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        import uvicorn

        async def health(_request: Request) -> JSONResponse:
            return JSONResponse({"status": "ok", "channel": "webhook"})

        async def message(request: Request) -> JSONResponse:
            # Auth check
            if self.token:
                auth = request.headers.get("authorization", "")
                if auth != f"Bearer {self.token}":
                    return JSONResponse({"error": "unauthorized"}, status_code=401)

            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON"}, status_code=400)

            text = body.get("text", "").strip()
            if not text:
                return JSONResponse({"error": "text is required"}, status_code=400)

            session = body.get("session", "webhook-default")

            try:
                resp = await self.send_to_gateway(text, session=session)
                return JSONResponse({
                    "response": resp.get("content", ""),
                    "session": session,
                    "tokens": resp.get("metadata", {}).get("tokens", 0),
                })
            except Exception as e:
                log.error(f"Webhook error: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)

        app = Starlette(routes=[
            Route("/health", health, methods=["GET"]),
            Route("/message", message, methods=["POST"]),
        ])

        log.info(f"Webhook channel listening on http://{self.host}:{self.port}")
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    async def send(self, content: str, **kwargs: Any) -> None:
        """Not used for webhook — responses are synchronous HTTP."""
        pass

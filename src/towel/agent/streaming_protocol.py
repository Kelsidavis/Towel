"""Server-Sent Events protocol — stream agent responses over HTTP.

Standard SSE format compatible with EventSource API, htmx, and
any HTTP client that supports streaming.

Usage from client:
    const es = new EventSource('/v1/stream?prompt=hello');
    es.onmessage = e => console.log(JSON.parse(e.data));
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route


def build_sse_routes(agent: Any, config: Any) -> list[Route]:
    """Build SSE streaming routes."""

    async def stream_endpoint(request: Request) -> StreamingResponse:
        """GET /v1/stream?prompt=...&session=... — SSE streaming."""
        prompt = request.query_params.get("prompt", "")
        session_id = request.query_params.get("session", "sse-default")

        if not prompt:
            return StreamingResponse(
                iter(['data: {"error": "prompt parameter required"}\n\n']),
                media_type="text/event-stream",
            )

        from towel.agent.conversation import Conversation, Role

        conv = Conversation(id=session_id, channel="sse")
        conv.add(Role.USER, prompt)

        async def generate() -> AsyncIterator[str]:
            from towel.agent.events import EventType

            yield f"data: {json.dumps({'type': 'start', 'session': session_id})}\n\n"

            full_text = ""
            async for event in agent.step_streaming(conv):
                if event.type == EventType.TOKEN:
                    chunk = event.data["content"]
                    full_text += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

                elif event.type == EventType.TOOL_CALL:
                    payload = {
                        'type': 'tool_call',
                        'tool': event.data['tool'],
                        'arguments': str(
                            event.data.get('arguments', {})
                        )[:200],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                elif event.type == EventType.TOOL_RESULT:
                    payload = {
                        'type': 'tool_result',
                        'tool': event.data['tool'],
                        'result': event.data['result'][:500],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                elif event.type == EventType.RESPONSE_COMPLETE:
                    meta = event.data.get("metadata", {})
                    payload = {
                        'type': 'done',
                        'content': full_text,
                        'tokens': meta.get('tokens', 0),
                        'tps': round(meta.get('tps', 0), 1),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                elif event.type == EventType.ERROR:
                    payload = {
                        'type': 'error',
                        'message': event.data.get(
                            'message', 'unknown'
                        ),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def stream_post(request: Request) -> StreamingResponse:
        """POST /v1/stream — SSE streaming with JSON body."""
        try:
            body = await request.json()
        except Exception:
            return StreamingResponse(
                iter(['data: {"error": "invalid JSON"}\n\n']),
                media_type="text/event-stream",
            )

        prompt = body.get("prompt", body.get("message", ""))
        session_id = body.get("session", "sse-default")

        if not prompt:
            return StreamingResponse(
                iter(['data: {"error": "prompt required"}\n\n']),
                media_type="text/event-stream",
            )

        from towel.agent.conversation import Conversation, Role

        conv = Conversation(id=session_id, channel="sse")
        conv.add(Role.USER, prompt)

        async def generate() -> AsyncIterator[str]:
            from towel.agent.events import EventType

            full_text = ""
            async for event in agent.step_streaming(conv):
                if event.type == EventType.TOKEN:
                    full_text += event.data["content"]
                    payload = {
                        'type': 'token',
                        'content': event.data['content'],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                elif event.type == EventType.RESPONSE_COMPLETE:
                    meta = event.data.get("metadata", {})
                    payload = {
                        'type': 'done',
                        'tokens': meta.get('tokens', 0),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return [
        Route("/v1/stream", stream_endpoint, methods=["GET"]),
        Route("/v1/stream", stream_post, methods=["POST"]),
    ]

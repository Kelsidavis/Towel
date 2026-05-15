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
                    meta = event.data.get("metadata", {}) or {}
                    # tps/tokens can arrive as explicit None when a
                    # worker emits a metadata block but never measured
                    # (e.g. empty-text fallback). `round(None, 1)`
                    # raises TypeError and would crash the SSE
                    # generator mid-stream. Coerce non-numeric to 0
                    # at the boundary (same fix /api/ask got in
                    # 8473883).
                    tps_raw = meta.get('tps')
                    tps_val = float(tps_raw) if isinstance(tps_raw, (int, float)) else 0.0
                    tokens_raw = meta.get('tokens')
                    tokens_val = (
                        int(tokens_raw) if isinstance(tokens_raw, (int, float)) else 0
                    )
                    payload = {
                        'type': 'done',
                        'content': full_text,
                        'tokens': tokens_val,
                        'tps': round(tps_val, 1),
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
                    meta = event.data.get("metadata", {}) or {}
                    # Same None-coerce as the other RESPONSE_COMPLETE
                    # branch in this file — workers occasionally emit
                    # `tokens: None` and a null in the SSE payload
                    # confuses clients that expect a number.
                    tokens_raw = meta.get('tokens')
                    tokens_val = (
                        int(tokens_raw) if isinstance(tokens_raw, (int, float)) else 0
                    )
                    payload = {
                        'type': 'done',
                        'tokens': tokens_val,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                elif event.type == EventType.ERROR:
                    # Parity with the GET stream handler — emit an
                    # error payload so the client sees the failure
                    # instead of a silently-empty stream.
                    payload = {
                        'type': 'error',
                        'message': event.data.get('message', 'unknown'),
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                elif event.type == EventType.CANCELLED:
                    payload = {
                        'type': 'cancelled',
                        'reason': event.data.get(
                            'metadata', {}
                        ).get('reason', 'user_cancelled'),
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

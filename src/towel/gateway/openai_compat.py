"""OpenAI-compatible API endpoint — /v1/chat/completions.

Makes Towel a drop-in replacement for any OpenAI client.
Supports both streaming (SSE) and non-streaming responses.

Usage with the openai Python SDK:
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:18743/v1", api_key="towel")
    resp = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
    )

Usage with curl:
    curl http://127.0.0.1:18743/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, TYPE_CHECKING

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from towel.agent.runtime import AgentRuntime
    from towel.config import TowelConfig


def build_openai_routes(agent: "AgentRuntime", config: "TowelConfig") -> list[Route]:
    """Build /v1/* routes for OpenAI API compatibility."""

    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": {"message": "Invalid JSON", "type": "invalid_request_error"}}, status_code=400)

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model_name = body.get("model", config.model.name)

        if not messages:
            return JSONResponse({"error": {"message": "messages is required", "type": "invalid_request_error"}}, status_code=400)

        # Build a temporary conversation from the messages
        from towel.agent.conversation import Conversation, Role

        conv = Conversation(channel="api")
        role_map = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}
        for msg in messages:
            role = role_map.get(msg.get("role", "user"), Role.USER)
            conv.add(role, msg.get("content", ""))

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        try:
            if stream:
                return StreamingResponse(
                    _stream_sse(agent, conv, request_id, created, model_name),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                response = await agent.step(conv)
                return JSONResponse(_format_completion(
                    request_id, created, model_name, response.content,
                    response.metadata.get("tokens", 0),
                ))
        except Exception as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "server_error"}},
                status_code=500,
            )

    async def list_models(request: Request) -> JSONResponse:
        """GET /v1/models — list available models."""
        models = [{"id": config.model.name, "object": "model", "owned_by": "towel"}]
        for name, profile in config.list_agents().items():
            models.append({"id": name, "object": "model", "owned_by": "towel"})
        return JSONResponse({"object": "list", "data": models})

    return [
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", list_models, methods=["GET"]),
    ]


async def _stream_sse(
    agent: "AgentRuntime",
    conv: "Conversation",
    request_id: str,
    created: int,
    model: str,
) -> Any:
    """Yield Server-Sent Events in OpenAI streaming format."""
    from towel.agent.events import EventType

    async for event in agent.step_streaming(conv):
        if event.type == EventType.TOKEN:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": event.data["content"]},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        elif event.type == EventType.RESPONSE_COMPLETE:
            # Final chunk with finish_reason
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        elif event.type == EventType.CANCELLED:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

    # Safety: always end stream
    yield "data: [DONE]\n\n"


def _format_completion(
    request_id: str,
    created: int,
    model: str,
    content: str,
    total_tokens: int,
) -> dict[str, Any]:
    """Format a non-streaming ChatCompletion response."""
    prompt_tokens = max(1, total_tokens // 4)  # rough estimate
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens,
            "total_tokens": prompt_tokens + total_tokens,
        },
    }

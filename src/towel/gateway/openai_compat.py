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
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from towel.agent.conversation import Conversation
    from towel.agent.runtime import AgentRuntime
    from towel.config import TowelConfig


def build_openai_routes(
    agent: AgentRuntime,
    config: TowelConfig,
    *,
    gateway: Any = None,
) -> list[Route]:
    """Build /v1/* routes for OpenAI API compatibility.

    ``gateway`` (the GatewayServer instance) is optional and threads
    the same worker-dispatch path /api/ask uses through to this
    endpoint. Without it, /v1/chat/completions falls back to running
    on the coordinator's local agent — which works for single-process
    setups but bypasses the worker fleet entirely. With it, chat-
    class queries route to the smallest qualified worker, the same
    as the rest of the system.
    """

    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status_code=400,
            )

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        model_name = body.get("model", config.model.name)

        if not messages:
            return JSONResponse(
                {"error": {"message": "messages is required", "type": "invalid_request_error"}},
                status_code=400,
            )

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
                # When a gateway is wired in and a worker is available
                # for this request, route the stream through the fleet
                # so SSE clients see the same model the rest of the
                # system uses. Falls back to the local agent's
                # step_streaming when no gateway is available, when
                # routing returns None (empty fleet), or when picking
                # a worker fails for any reason.
                generator = _stream_sse(agent, conv, request_id, created, model_name)
                if gateway is not None and messages:
                    last_user = next(
                        (m.get("content", "") for m in reversed(messages)
                         if m.get("role") == "user"),
                        "",
                    )
                    session_id = f"openai-{request_id}"
                    sess = gateway.sessions.get_or_create(session_id)
                    sess.conversation = conv
                    try:
                        worker, _intent = await gateway._route_by_role(
                            last_user, session_id,
                        )
                        if worker is not None:
                            generator = _stream_sse_remote(
                                gateway, session_id, sess, worker,
                                request_id, created, model_name,
                                fallback_agent=agent,
                                fallback_conv=conv,
                            )
                    except Exception as exc:
                        import logging
                        logging.getLogger("towel.openai_compat").debug(
                            "stream route failed, falling back: %s", exc,
                        )
                return StreamingResponse(
                    generator,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                # Non-streaming path: try to route through the worker
                # fleet first when a gateway was supplied. Falls back
                # to the local agent for empty-fleet setups or when the
                # router declines (returns None).
                response = None
                if gateway is not None and messages:
                    last_user = next(
                        (m.get("content", "") for m in reversed(messages)
                         if m.get("role") == "user"),
                        "",
                    )
                    session_id = f"openai-{request_id}"
                    sess = gateway.sessions.get_or_create(session_id)
                    sess.conversation = conv
                    try:
                        worker, intent = await gateway._route_by_role(
                            last_user, session_id,
                        )
                        if worker is not None:
                            if intent == "chat":
                                response = await gateway._quick_remote_infer(
                                    session_id, sess, worker, max_tokens=512,
                                )
                                # Same retry-on-empty path /api/ask uses
                                # (see commit 534e40f). When the small
                                # worker returns no text, try the next
                                # idle worker before surfacing the
                                # diagnostic placeholder.
                                if (response.metadata or {}).get(
                                    "empty_text_fallback"
                                ):
                                    alt = gateway._pick_alternate_chat_worker(
                                        exclude={worker.id},
                                    )
                                    if alt is not None:
                                        if sess.conversation.messages and (
                                            sess.conversation.messages[-1].role.value
                                            == "assistant"
                                        ):
                                            sess.conversation.messages.pop()
                                        retry = await gateway._quick_remote_infer(
                                            session_id, sess, alt, max_tokens=512,
                                        )
                                        if not (retry.metadata or {}).get(
                                            "empty_text_fallback"
                                        ):
                                            retry.metadata = (retry.metadata or {}) | {
                                                "fallback_from_worker": worker.id,
                                                "fallback_reason": "empty_text",
                                            }
                                            response = retry
                            else:
                                response = await gateway._step_remote_inference(
                                    session_id, sess, worker,
                                )
                    except Exception as exc:
                        # Worker route failed — fall through to local
                        # agent rather than 500 the request.
                        import logging
                        logging.getLogger("towel.openai_compat").debug(
                            "worker route failed, falling back: %s", exc,
                        )
                if response is None:
                    response = await agent.step(conv)
                meta = response.metadata or {}
                completion_tokens = meta.get("tokens", meta.get("output_tokens", 0))
                # Defensive: when the worker reports zero completion_tokens
                # but we have visible content, estimate from the response
                # text. Happens when an upstream llama-server build doesn't
                # populate usage, or when a worker is running pre-fix code
                # and silently drops the count for reasoning_content
                # substitutions.
                if completion_tokens == 0 and response.content:
                    from towel.agent.context import count_tokens_fallback
                    completion_tokens = count_tokens_fallback(response.content)
                # Prefer the worker's reported prompt_tokens; fall back
                # to estimating from the conversation we sent. Previous
                # code derived prompt_tokens from completion_tokens // 4
                # which gave nonsense (e.g. prompt=1 for a long input
                # that produced 0 tokens).
                prompt_tokens = meta.get("prompt_tokens")
                if prompt_tokens is None:
                    from towel.agent.context import count_tokens_fallback
                    prompt_tokens = sum(
                        count_tokens_fallback(msg.get("content", ""))
                        for msg in messages
                    )
                return JSONResponse(
                    _format_completion(
                        request_id,
                        created,
                        model_name,
                        response.content,
                        completion_tokens,
                        prompt_tokens=prompt_tokens,
                    )
                )
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


async def _stream_sse_remote(
    gateway: Any,
    session_id: str,
    session: Any,
    worker: Any,
    request_id: str,
    created: int,
    model: str,
    *,
    fallback_agent: Any = None,
    fallback_conv: Any = None,
) -> Any:
    """SSE generator that pipes tokens from a remote worker.

    Each token from gateway.iter_remote_tokens becomes an OpenAI
    chunk; the final chunk has finish_reason="stop". Errors mid-
    stream surface as a finish_reason="error" final chunk so the
    client doesn't hang waiting for [DONE].

    When ``fallback_agent`` is supplied and the remote worker errors
    BEFORE any token was streamed, we fall back to the local agent's
    streaming path so SSE clients still get a response. Once any
    token has been emitted we can't switch generators silently — at
    that point an error must surface as an error chunk.
    """
    yielded_any = False
    try:
        async for token in gateway.iter_remote_tokens(session_id, session, worker):
            yielded_any = True
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as exc:
        if not yielded_any and fallback_agent is not None and fallback_conv is not None:
            import logging
            logging.getLogger("towel.openai_compat").warning(
                "remote stream failed before any token (%s); "
                "falling back to local agent",
                exc,
            )
            async for chunk in _stream_sse(
                fallback_agent, fallback_conv, request_id, created, model,
            ):
                yield chunk
            return
        err_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "error"},
            ],
            "error": {"message": str(exc)},
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_sse(
    agent: AgentRuntime,
    conv: Conversation,
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": event.data["content"]},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        elif event.type == EventType.RESPONSE_COMPLETE:
            # Final chunk with finish_reason
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
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
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
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
    completion_tokens: int,
    *,
    prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """Format a non-streaming ChatCompletion response."""
    if prompt_tokens is None:
        # No prompt information supplied — fall back to a 1-token
        # placeholder rather than a number derived from completion
        # length, which is meaningless.
        prompt_tokens = 1
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

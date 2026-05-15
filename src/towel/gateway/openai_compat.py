"""OpenAI-compatible API endpoint — /v1/chat/completions.

Makes Towel a drop-in replacement for any OpenAI client.
Supports both streaming (SSE) and non-streaming responses.

Standard OpenAI params honored: model, messages, max_tokens (1..4096),
temperature (0..2), stream. Response objects include ``created``,
``system_fingerprint`` (per-process stable, derived from the towel
package version), and the usual ``usage`` block.

Towel-specific extensions (OpenAI's spec allows extra body fields,
so the official Python SDK passes these via ``extra_body=``):

    verify: bool   — opt-in to a second-worker review pass after
                     the primary answer lands. Mutually exclusive
                     with ``ensemble``. Requires ``stream=false``.
    ensemble: bool — opt-in to parallel fan-out across every idle
                     inference worker, with the coordinator
                     synthesizing the final answer. Same stream and
                     mutex constraints as verify.

Usage with the openai Python SDK:
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:18743/v1", api_key="towel")
    resp = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        extra_body={"verify": True},
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


# Stable Unix timestamp returned in every /v1/models `created`
# field — captured at module-import time so all models in the
# response (and across repeated calls within a process lifetime)
# share the same value. OpenAI's response uses a per-model
# train-cutoff constant; we have nothing meaningful per-model so
# pin to coordinator startup, which is at least deterministic and
# monotonically older than now() for any consumer who validates.
_OPENAI_MODELS_CREATED = int(time.time())


def _system_fingerprint() -> str:
    """Stable per-process system_fingerprint for ChatCompletion responses.

    Modern OpenAI clients read `system_fingerprint` to detect when
    the underlying inference system has changed — e.g. for cache
    invalidation. A response without the field is technically valid
    but breaks clients that rely on it (some LangChain caches, eval
    harnesses). Derive from the towel package version so the
    fingerprint flips on coordinator upgrades, matching OpenAI's
    behaviour of changing fingerprints on model revisions.
    """
    try:
        from towel import __version__ as _v
    except Exception:
        _v = "0"
    import hashlib as _hash
    return "fp_towel_" + _hash.sha256(_v.encode("utf-8")).hexdigest()[:10]


_SYSTEM_FINGERPRINT = _system_fingerprint()


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
        """POST /v1/chat/completions — OpenAI-compat chat endpoint.

        Routes the request through the worker fleet (or the local
        agent if no gateway is wired), formats the response in
        OpenAI's `chat.completion` shape (or SSE chunks when
        `stream=true`), and surfaces Towel-specific collaboration
        state under a `towel` namespaced field on non-streaming
        responses.

        See the module docstring for the full param schema,
        including towel extensions (`verify`, `ensemble`).
        """
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                status_code=400,
            )
        # Top-level body must be an object — an array / string / null
        # body crashed on `body.get(...)` and surfaced as plaintext
        # "Internal Server Error" HTTP 500, breaking OpenAI clients
        # that expect a structured 400.
        if not isinstance(body, dict):
            return JSONResponse(
                {"error": {"message": "body must be a JSON object", "type": "invalid_request_error"}},
                status_code=400,
            )

        messages = body.get("messages", [])
        stream = body.get("stream", False)
        # Strict bool check — `{"stream": "yes"}` would otherwise pass
        # as truthy and silently take the streaming path even though
        # the client likely intended the literal string. OpenAI's
        # contract uses a boolean; reject non-bool inputs.
        if not isinstance(stream, bool):
            return JSONResponse(
                {"error": {"message": "stream must be true or false", "type": "invalid_request_error"}},
                status_code=400,
            )
        model_name = body.get("model", config.model.name)
        # `model` is echoed back in the response and SSE chunks. A
        # non-string would render oddly in JSON output ("model":[1,2])
        # and confuse OpenAI clients that expect a string identifier.
        # The dispatcher routes by intent/task_type, not by model name,
        # so this field is cosmetic — but cosmetic should still be valid.
        if not isinstance(model_name, str):
            return JSONResponse(
                {"error": {"message": "model must be a string", "type": "invalid_request_error"}},
                status_code=400,
            )
        # Honor OpenAI-standard sampling params. max_tokens is clamped
        # to [1, 4096] so a hostile or accidental request can't burn
        # the worker's max generation budget; the previous behavior was
        # to always use 512 regardless of what the caller asked for —
        # which broke proper OpenAI clients (LangChain, llm-cli, etc.)
        # that expected the param to flow through.
        try:
            req_max_tokens = body.get("max_tokens", 512)
            if req_max_tokens is None:
                req_max_tokens = 512
            req_max_tokens = max(1, min(int(req_max_tokens), 4096))
        except (TypeError, ValueError):
            return JSONResponse(
                {"error": {"message": "max_tokens must be an integer", "type": "invalid_request_error"}},
                status_code=400,
            )
        try:
            req_temperature = body.get("temperature", 0.7)
            if req_temperature is None:
                req_temperature = 0.7
            req_temperature = float(req_temperature)
            # Clamp to OpenAI's documented [0, 2] range.
            req_temperature = max(0.0, min(req_temperature, 2.0))
        except (TypeError, ValueError):
            return JSONResponse(
                {"error": {"message": "temperature must be a number", "type": "invalid_request_error"}},
                status_code=400,
            )

        # OpenAI's spec allows extra body fields. Surface the two
        # collaboration modes here so non-towel clients (LangChain,
        # llm-cli, OpenAI Python SDK with extra_body=) can opt in
        # to multi-worker collaboration on the same endpoint they
        # already use. Mutually exclusive, same as /api/ask. Streaming
        # is intentionally not supported for these modes (the
        # synthesis step is inherently non-streaming), so they're
        # rejected for stream=true requests.
        verify_raw = body.get("verify", False)
        if not isinstance(verify_raw, bool):
            return JSONResponse(
                {"error": {"message": "verify must be true or false", "type": "invalid_request_error"}},
                status_code=400,
            )
        ensemble_raw = body.get("ensemble", False)
        if not isinstance(ensemble_raw, bool):
            return JSONResponse(
                {"error": {"message": "ensemble must be true or false", "type": "invalid_request_error"}},
                status_code=400,
            )
        if verify_raw and ensemble_raw:
            return JSONResponse(
                {"error": {"message": "ensemble and verify are mutually exclusive", "type": "invalid_request_error"}},
                status_code=400,
            )
        if stream and (verify_raw or ensemble_raw):
            return JSONResponse(
                {"error": {"message": "verify/ensemble require stream=false (synthesis can't be streamed)", "type": "invalid_request_error"}},
                status_code=400,
            )

        if not messages:
            return JSONResponse(
                {"error": {"message": "messages is required", "type": "invalid_request_error"}},
                status_code=400,
            )
        if not isinstance(messages, list) or not all(
            isinstance(m, dict) for m in messages
        ):
            return JSONResponse(
                {"error": {"message": "messages must be a list of objects", "type": "invalid_request_error"}},
                status_code=400,
            )
        # The OpenAI contract requires at least one message to carry
        # non-empty content. A request whose every message has empty
        # content silently sent the worker an unanswerable prompt and
        # then the caller waited the full chat-fast timeout (60s)
        # before getting `{"error": "worker ... did not respond ..."}`.
        # Fail loud at the coordinator instead.
        #
        # Also surface multimodal content (`content` as a list of
        # parts, OpenAI's vision/audio shape) with a specific error
        # — those clients deserve to know towel doesn't support
        # multimodal yet, not a generic "non-empty content" message.
        any_multimodal = any(
            isinstance(m.get("content"), list) for m in messages
        )
        if any_multimodal:
            return JSONResponse(
                {"error": {"message": "multimodal content (list parts) is not supported; pass a plain string", "type": "invalid_request_error"}},
                status_code=400,
            )
        has_content = any(
            isinstance(m.get("content"), str) and m.get("content", "").strip()
            for m in messages
        )
        if not has_content:
            return JSONResponse(
                {"error": {"message": "at least one message must have non-empty content", "type": "invalid_request_error"}},
                status_code=400,
            )
        # Require at least one USER turn. A system-only conversation
        # ("be terse" with no user prompt) has no question for the
        # model to answer; most workers hang or return empty for these
        # and the caller times out at 60s. OpenAI's API technically
        # accepts system-only requests but no real model produces
        # meaningful output — make it loud at the boundary so the
        # caller gets immediate feedback.
        has_user_content = any(
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m.get("content", "").strip()
            for m in messages
        )
        if not has_user_content:
            return JSONResponse(
                {"error": {"message": "messages must include at least one user turn with content", "type": "invalid_request_error"}},
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
                # Track the one-shot session id so we can clean up
                # affinity + the worker-side context slot after the
                # SSE stream completes. Same leak class fixed for the
                # non-streaming path below.
                openai_session_id: str | None = None
                if gateway is not None and messages:
                    last_user = next(
                        (m.get("content", "") for m in reversed(messages)
                         if m.get("role") == "user"),
                        "",
                    )
                    session_id = f"openai-{request_id}"
                    openai_session_id = session_id
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

                # Wrap the generator to clean up the one-shot session
                # state when the SSE stream finishes (success, client
                # disconnect, or mid-stream error). Without this, every
                # streaming call to /v1/chat/completions leaves a ghost
                # affinity entry + context slot behind.
                async def _cleanup_after_stream():
                    try:
                        async for chunk in generator:
                            yield chunk
                    finally:
                        if openai_session_id is not None and gateway is not None:
                            gateway.cleanup_ephemeral_session(openai_session_id)
                return StreamingResponse(
                    _cleanup_after_stream(),
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
                # OpenAI-compat creates a one-shot session_id per
                # request. Tracked here so the outer finally can clean
                # it up — without that, every /v1/chat/completions call
                # left a permanent ghost entry in _session_workers AND
                # a permanent context slot on the routed worker,
                # inflating context_pressure forever.
                openai_session_id: str | None = None
                if gateway is not None and messages:
                    last_user = next(
                        (m.get("content", "") for m in reversed(messages)
                         if m.get("role") == "user"),
                        "",
                    )
                    session_id = f"openai-{request_id}"
                    openai_session_id = session_id
                    sess = gateway.sessions.get_or_create(session_id)
                    sess.conversation = conv

                    # Ensemble: parallel fan-out + LLM-as-judge
                    # synthesis. Bypasses single-worker routing — same
                    # mechanism as /api/ask?ensemble=true (see commits
                    # c81f481, 7614e9f, b396319). Short-circuits when
                    # the fan-out produces a real answer; otherwise
                    # falls through to the normal single-worker route.
                    # Initialize the captured contributions + arb_mode
                    # at function scope (not just inside `if ensemble_raw`)
                    # so the response-building block downstream can
                    # surface them even on the fall-through path — parity
                    # with /api/ask's ensemble_contributions field.
                    _contribs: list[dict[str, Any]] = []
                    arb_mode = ""
                    if ensemble_raw:
                        arbitrated, _contribs, arb_mode = await gateway._ensemble_dispatch(
                            session_id, last_user, user_session=sess,
                        )
                        # Aggregate dispatch entry — parity with
                        # /api/ask's record_ensemble call (e00fb6d).
                        # Always record when the user opted in (drop the
                        # _contribs guard) so the "all workers busy →
                        # silent fall-through" case shows up in the
                        # dispatch log instead of looking like a normal
                        # single-worker dispatch.
                        if getattr(gateway, "_dispatcher", None) is not None:
                            try:
                                gateway._dispatcher.record_ensemble(
                                    session_id=session_id,
                                    contributions=_contribs,
                                    arbitration_mode=arb_mode,
                                )
                            except Exception:
                                pass
                        if arbitrated:
                            from towel.agent.conversation import Message
                            response = Message(
                                role=Role.ASSISTANT, content=arbitrated,
                                metadata={
                                    "ensemble": True,
                                    "ensemble_arbitration": arb_mode,
                                    "remote_worker": "ensemble",
                                },
                            )
                            sess.conversation.messages.append(response)

                    try:
                        # Skip the single-worker route if ensemble
                        # already produced an arbitrated answer.
                        if response is not None:
                            worker, intent = None, "task"
                        else:
                            worker, intent = await gateway._route_by_role(
                                last_user, session_id,
                            )
                        if worker is not None:
                            if intent == "chat":
                                # Wrap primary call so a timeout / worker
                                # error also gets a retry on the alternate
                                # — same as /api/ask in commit 92c5c1b.
                                try:
                                    response = await gateway._quick_remote_infer(
                                        session_id, sess, worker,
                                        max_tokens=req_max_tokens,
                                        temperature=req_temperature,
                                    )
                                    v1_primary_failed = False
                                    v1_primary_exc: Exception | None = None
                                except Exception as v1_exc:
                                    import logging as _v1_log
                                    _v1_log.getLogger("towel.openai_compat").info(
                                        "primary worker %s raised %s; will try alternate",
                                        worker.id, v1_exc,
                                    )
                                    response = None
                                    v1_primary_failed = True
                                    v1_primary_exc = v1_exc
                                # Same retry-on-empty path /api/ask uses
                                # (see commit 534e40f). When the small
                                # worker returns no text, try the next
                                # idle worker before surfacing the
                                # diagnostic placeholder.
                                v1_needs_retry = v1_primary_failed or (
                                    response is not None
                                    and (response.metadata or {}).get(
                                        "empty_text_fallback"
                                    )
                                )
                                if v1_needs_retry:
                                    alt = gateway._pick_alternate_chat_worker(
                                        exclude={worker.id},
                                    )
                                    if alt is not None:
                                        # Record the retry as its own
                                        # dispatch decision so
                                        # /dispatch/recent shows the
                                        # fallback path. Same as /api/ask.
                                        # Pass the real failure cause so
                                        # the dispatch notes don't claim
                                        # "empty response" for a timed-
                                        # out primary.
                                        if getattr(gateway, "_dispatcher", None) is not None:
                                            gateway._dispatcher.record_retry(
                                                session_id=session_id,
                                                retry_worker=alt,
                                                original_worker_id=worker.id,
                                                intent="chat",
                                                cause=(
                                                    f"primary_failed: {v1_primary_exc}"
                                                    if v1_primary_failed and v1_primary_exc is not None
                                                    else "empty_text"
                                                ),
                                            )
                                        # Pop the placeholder so the alt
                                        # worker doesn't see it as a prior
                                        # assistant turn. Only the empty-
                                        # text path has a placeholder to
                                        # pop — primary_failed never appended.
                                        popped: Any = None
                                        if (
                                            not v1_primary_failed
                                            and sess.conversation.messages
                                            and sess.conversation.messages[-1].role.value
                                            == "assistant"
                                        ):
                                            popped = sess.conversation.messages.pop()
                                        try:
                                            retry = await gateway._quick_remote_infer(
                                                session_id, sess, alt,
                                                max_tokens=req_max_tokens,
                                                temperature=req_temperature,
                                            )
                                        except Exception as retry_exc:
                                            import logging
                                            logging.getLogger(
                                                "towel.openai_compat"
                                            ).warning(
                                                "retry on %s failed (%s); keeping %s",
                                                alt.id, retry_exc,
                                                "primary exception" if v1_primary_failed
                                                else f"empty-text response from {worker.id}",
                                            )
                                            if popped is not None:
                                                sess.conversation.messages.append(popped)
                                            # If primary also failed and we
                                            # have no response, let the outer
                                            # except catch and fall through
                                            # to local agent.
                                            if v1_primary_failed:
                                                raise v1_primary_exc  # type: ignore[misc]
                                        else:
                                            alt_was_empty = (
                                                retry.metadata or {}
                                            ).get("empty_text_fallback")
                                            if (not alt_was_empty) or v1_primary_failed:
                                                retry.metadata = (retry.metadata or {}) | {
                                                    "fallback_from_worker": worker.id,
                                                    "fallback_reason": (
                                                        "primary_failed"
                                                        if v1_primary_failed
                                                        else "empty_text"
                                                    ),
                                                }
                                                response = retry
                                            else:
                                                # Dual-empty: primary AND alt
                                                # both returned the empty-
                                                # text fallback. Same
                                                # diagnostic /api/ask
                                                # surfaces — flag on the
                                                # metadata + warn at log
                                                # level so operators see
                                                # the fleet-wide tool-loop
                                                # rather than blaming a
                                                # single slow worker.
                                                import logging as _v1_log_warn
                                                _v1_log_warn.getLogger(
                                                    "towel.openai_compat"
                                                ).warning(
                                                    "Dual empty-text on session %s: "
                                                    "primary=%s alt=%s — both "
                                                    "workers tool-looped; review "
                                                    "system prompt or worker quality.",
                                                    session_id, worker.id, alt.id,
                                                )
                                                response.metadata = (
                                                    response.metadata or {}
                                                ) | {
                                                    "dual_empty_text": True,
                                                    "alt_worker": alt.id,
                                                }
                                    elif v1_primary_failed:
                                        # No alt; let the outer except
                                        # catch this and fall through to
                                        # local agent (existing behavior).
                                        assert v1_primary_exc is not None
                                        raise v1_primary_exc
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

                # Verify pass: opt-in second-worker review of the
                # primary's answer. Same mechanism as /api/ask's verify
                # flag — skipped on ensemble (the modes are mutually
                # exclusive, see the boundary check), on empty
                # placeholders, and when no alternate worker exists.
                if (
                    verify_raw
                    and gateway is not None
                    and response.content
                    and not (response.metadata or {}).get("empty_text_fallback")
                ):
                    # Primary worker id lives in the response metadata
                    # under remote_worker (set by _quick_remote_infer
                    # / _step_remote_inference).
                    primary_id = (response.metadata or {}).get("remote_worker")
                    if primary_id and primary_id != "ensemble":
                        last_user_text = next(
                            (m.get("content", "") for m in reversed(messages)
                             if m.get("role") == "user"),
                            "",
                        )
                        final, was_corrected, verifier_id = (
                            await gateway._verify_pass(
                                session_id, last_user_text,
                                response.content, primary_id,
                            )
                        )
                        # Aggregate dispatch entry — parity with
                        # /api/ask's record_verify call (e00fb6d).
                        # Always record when verify was opted in,
                        # including the no-alt skipped case
                        # (verifier_id=None) so operators see why a
                        # response lacks the verified_by marker.
                        if getattr(gateway, "_dispatcher", None) is not None:
                            try:
                                gateway._dispatcher.record_verify(
                                    session_id=session_id,
                                    verifier_id=verifier_id,
                                    primary_id=primary_id,
                                    was_corrected=was_corrected,
                                )
                            except Exception:
                                pass
                        if was_corrected and final != response.content:
                            response.content = final
                            response.metadata = (response.metadata or {}) | {
                                "verified_by": verifier_id,
                                "verifier_corrected": True,
                                "primary_worker": primary_id,
                            }
                        elif verifier_id is not None:
                            response.metadata = (response.metadata or {}) | {
                                "verified_by": verifier_id,
                                "verifier_corrected": False,
                                "primary_worker": primary_id,
                            }

                meta = response.metadata or {}
                # Workers occasionally emit `tokens: None` /
                # `output_tokens: None` after a job_error or
                # empty-text fallback. The downstream `prompt_tokens +
                # completion_tokens` in _format_completion raises
                # TypeError on None, turning a recoverable error into
                # HTTP 500 with no SSE complete frame. Coerce to int
                # at the boundary (same defensive shape /api/ask got
                # in 8473883 and SSE got in b553f7b).
                completion_raw = meta.get("tokens", meta.get("output_tokens", 0))
                completion_tokens = (
                    int(completion_raw)
                    if isinstance(completion_raw, (int, float))
                    else 0
                )
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
                prompt_raw = meta.get("prompt_tokens")
                prompt_tokens = (
                    int(prompt_raw)
                    if isinstance(prompt_raw, (int, float))
                    else None
                )
                if prompt_tokens is None:
                    from towel.agent.context import count_tokens_fallback
                    prompt_tokens = sum(
                        count_tokens_fallback(msg.get("content", ""))
                        for msg in messages
                    )
                try:
                    completion = _format_completion(
                        request_id,
                        created,
                        model_name,
                        response.content,
                        completion_tokens,
                        prompt_tokens=prompt_tokens,
                    )
                    # Vendor-namespaced metadata so OpenAI-strict
                    # clients ignore it and Towel-aware clients can
                    # see whether verify/ensemble actually ran. Sits
                    # under a single `towel` key rather than at the
                    # top level so we never collide with a future
                    # OpenAI field. Only emitted when there's
                    # something to report.
                    towel_meta: dict[str, Any] = {}
                    if meta.get("verified_by"):
                        towel_meta["verified_by"] = meta["verified_by"]
                        towel_meta["verifier_corrected"] = bool(
                            meta.get("verifier_corrected", False)
                        )
                        towel_meta["primary_worker"] = meta.get(
                            "primary_worker", meta.get("remote_worker", "")
                        )
                    elif verify_raw:
                        # verify=true was requested but no verifier
                        # ran. Mirror the /api/ask response shape so
                        # OpenAI-compat clients also see the skip
                        # signal in the towel-namespaced block.
                        towel_meta["verify_skipped"] = True
                        towel_meta["verify_skip_reason"] = (
                            "primary returned no usable text; nothing to verify"
                            if meta.get("empty_text_fallback")
                            or meta.get("dual_empty_text")
                            else "no alternate worker available"
                        )
                    if meta.get("ensemble"):
                        towel_meta["ensemble"] = True
                        if meta.get("ensemble_arbitration"):
                            towel_meta["ensemble_arbitration"] = (
                                meta["ensemble_arbitration"]
                            )
                    elif ensemble_raw and not meta.get("verified_by"):
                        # ensemble=true requested but fell through —
                        # the response metadata won't have `ensemble`
                        # unless arbitration produced a real answer.
                        # The captured _contribs lets us mirror
                        # /api/ask's three-bucket skip-reason +
                        # surface the per-worker contributions so
                        # OpenAI-compat clients diagnose the failure
                        # the same way HTTP callers do.
                        towel_meta["ensemble_skipped"] = True
                        if not _contribs:
                            towel_meta["ensemble_skip_reason"] = (
                                "no idle workers available"
                            )
                        else:
                            errors = [c.get("error") for c in _contribs]
                            if all(e == "empty_text" for e in errors):
                                towel_meta["ensemble_skip_reason"] = (
                                    "all workers tool-looped "
                                    "(returned empty text)"
                                )
                            elif all(e == "ensemble_timeout" for e in errors):
                                towel_meta["ensemble_skip_reason"] = (
                                    "all workers timed out before "
                                    "producing a response"
                                )
                            else:
                                towel_meta["ensemble_skip_reason"] = (
                                    "mixed failures across the "
                                    "fan-out — see ensemble_"
                                    "contributions for details"
                                )
                        towel_meta["ensemble_contributions"] = _contribs
                    if meta.get("fallback_from_worker"):
                        towel_meta["fallback_from_worker"] = (
                            meta["fallback_from_worker"]
                        )
                        towel_meta["fallback_reason"] = meta.get(
                            "fallback_reason", ""
                        )
                    if meta.get("dual_empty_text"):
                        towel_meta["dual_empty_text"] = True
                        if meta.get("alt_worker"):
                            towel_meta["alt_worker"] = meta["alt_worker"]
                    if towel_meta:
                        completion["towel"] = towel_meta
                    return JSONResponse(completion)
                finally:
                    # Drop the one-shot session's affinity + context
                    # slot now that the request is done; otherwise
                    # every /v1/chat/completions call accumulates
                    # permanent ghost state.
                    if openai_session_id is not None and gateway is not None:
                        gateway.cleanup_ephemeral_session(openai_session_id)
        except Exception as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "server_error"}},
                status_code=500,
            )

    async def list_models(request: Request) -> JSONResponse:
        """GET /v1/models — list available models."""
        # OpenAI's /v1/models response includes a `created` Unix
        # timestamp on every model entry. The official OpenAI Python
        # SDK and some downstream clients (LangChain, llm CLI) read
        # the field — older SDK versions actually raise a validation
        # error if it's missing. Stamp the coordinator's startup
        # time so the timestamp is stable per process (and consistent
        # across all models in the response, matching OpenAI's
        # behaviour where every "their" model shares its train-cutoff
        # constant).
        created = int(_OPENAI_MODELS_CREATED)
        # De-dup ids: if the primary model.name happens to match an
        # agent profile name, the response previously contained two
        # entries with the same id. Clients keyed by id would see
        # one entry shadow the other unpredictably.
        seen_ids: set[str] = set()
        models: list[dict[str, Any]] = []
        for mid in [config.model.name, *config.list_agents().keys()]:
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            models.append({
                "id": mid,
                "object": "model",
                "created": created,
                "owned_by": "towel",
            })
        # Sort by id for stable order. Clients caching by index were
        # at the mercy of config.list_agents()'s insertion order;
        # alphabetical order is deterministic and matches what
        # OpenAI's response is for clients that sort anyway.
        models.sort(key=lambda m: m["id"])
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
                "system_fingerprint": _SYSTEM_FINGERPRINT,
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
            "system_fingerprint": _SYSTEM_FINGERPRINT,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "error"},
            ],
            # OpenAI's error shape always carries a `type` alongside
            # `message` — clients that switch on the field were
            # crashing on `KeyError: 'type'`. "server_error" matches
            # the non-streaming 500 path's classification.
            "error": {"message": str(exc), "type": "server_error"},
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": _SYSTEM_FINGERPRINT,
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
    """Yield Server-Sent Events in OpenAI streaming format.

    A mid-iteration failure in agent.step_streaming (model crash,
    cancellation propagation) previously propagated up without
    sending [DONE] — SSE clients that wait for the terminator
    before flushing would hang. Wrap the iteration so any exception
    becomes a final `finish_reason="error"` chunk + [DONE],
    matching the error-frame shape _stream_sse_remote already uses.
    """
    from towel.agent.events import EventType

    try:
        async for event in agent.step_streaming(conv):
            if event.type == EventType.TOKEN:
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": _SYSTEM_FINGERPRINT,
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
                    "system_fingerprint": _SYSTEM_FINGERPRINT,
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
                    "system_fingerprint": _SYSTEM_FINGERPRINT,
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

            elif event.type == EventType.ERROR:
                # The agent's runtime emits AgentEvent.error(msg)
                # for graceful in-stream failures. Without this
                # branch the SSE generator dropped the event,
                # finished its loop, and emitted a bare [DONE] —
                # the client got a stream that ended cleanly with
                # no content and no error indication. Surface the
                # event message as a structured error chunk
                # matching the exception-path shape below.
                err_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "system_fingerprint": _SYSTEM_FINGERPRINT,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "error"},
                    ],
                    "error": {
                        "message": event.data.get("message", "agent error"),
                        "type": "server_error",
                    },
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
    except Exception as exc:
        err_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": _SYSTEM_FINGERPRINT,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "error"},
            ],
            "error": {"message": str(exc), "type": "server_error"},
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Safety: always end stream (loop completed without TERMINAL event)
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
    """Format a non-streaming ChatCompletion response.

    Emits the OpenAI 2024-spec shape: ``id``, ``object`` set to
    ``"chat.completion"``, ``created``, ``model``,
    ``system_fingerprint`` (per-process stable, derived from
    package version), ``choices`` (single message), and ``usage``.

    Towel-specific metadata (``verified_by``, ``ensemble``,
    ``verify_skipped``, ``dual_empty_text``, …) is attached by the
    chat_completions handler under a top-level ``towel`` key, NOT
    by this function. Keep this formatter's output strictly
    OpenAI-spec — the handler decides whether to extend.
    """
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
        # `system_fingerprint` is what OpenAI clients use for cache
        # invalidation — a stable per-process derivation from the
        # towel version flips it on coordinator upgrades, which is
        # the right signal for downstream caches.
        "system_fingerprint": _SYSTEM_FINGERPRINT,
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

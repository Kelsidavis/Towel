"""Tests for the OpenAI-compatible API endpoint."""

import json

import pytest
from starlette.testclient import TestClient

from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.store import ConversationStore


@pytest.fixture
def client(tmp_path):
    store = ConversationStore(store_dir=tmp_path)
    config = TowelConfig()
    agent = AgentRuntime(config)
    sessions = SessionManager(store=store)
    gw = GatewayServer(config=config, agent=agent, sessions=sessions)
    return TestClient(gw._build_http_app())


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        # Default model should be present
        names = [m["id"] for m in data["data"]]
        assert any("Llama" in n or "default" in n or "coder" in n for n in names)

    def test_models_include_agents(self, client):
        resp = client.get("/v1/models")
        data = resp.json()
        names = [m["id"] for m in data["data"]]
        assert "coder" in names
        assert "researcher" in names
        assert "writer" in names

    def test_model_format(self, client):
        resp = client.get("/v1/models")
        model = resp.json()["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert model["owned_by"] == "towel"

    def test_model_includes_created_timestamp(self, client):
        """OpenAI's /v1/models response includes a `created` Unix
        timestamp per model. Older versions of the official OpenAI
        Python SDK raised a validation error when this field was
        missing. Some downstream clients (LangChain, llm CLI) also
        read it. Without this, a Towel-backed OpenAI client that
        listed models couldn't tell what model registry version it
        was looking at."""
        import time

        resp = client.get("/v1/models")
        data = resp.json()
        for model in data["data"]:
            assert "created" in model, model
            assert isinstance(model["created"], int)
            # Plausible bounds: 2020-01-01 .. 1 hour in the future.
            # The coordinator-start constant should be well within
            # this window for any sane test run.
            assert 1577836800 <= model["created"] <= int(time.time()) + 3600

    def test_models_sorted_by_id(self, client):
        """Model entries returned in alphabetical order by id —
        clients caching by index need stable ordering, and the
        insertion-order semantics of config.list_agents() left them
        at the mercy of however the agents dict happened to be
        built. Alphabetical is deterministic and matches what most
        clients sort to anyway."""
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert ids == sorted(ids), ids

    def test_models_deduped_by_id(self, client):
        """If the primary model name collides with an agent profile
        name, the response previously emitted two entries with the
        same id — clients keyed by id saw one shadow the other
        unpredictably. With de-dup, every id appears at most once."""
        resp = client.get("/v1/models")
        ids = [m["id"] for m in resp.json()["data"]]
        assert len(ids) == len(set(ids)), ids

    def test_models_share_created_timestamp(self, client):
        """OpenAI returns the same `created` across all official
        models in a single response — it's a per-process constant,
        not "now". Match that shape so clients caching by created
        don't see thrash on every request."""
        resp = client.get("/v1/models")
        data = resp.json()
        created_values = {m["created"] for m in data["data"]}
        assert len(created_values) == 1, created_values


class TestChatCompletionsEndpoint:
    def test_missing_messages(self, client):
        resp = client.post("/v1/chat/completions", json={"model": "default"})
        assert resp.status_code == 400
        assert "messages" in resp.json()["error"]["message"]

    def test_invalid_json(self, client):
        resp = client.post(
            "/v1/chat/completions",
            content="not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_empty_messages(self, client):
        resp = client.post("/v1/chat/completions", json={"model": "default", "messages": []})
        assert resp.status_code == 400

    def test_rejects_all_empty_content(self, client):
        """A request whose every message has empty content used to
        sit through the full chat-fast 60s timeout and then surface
        a useless `worker ... did not respond`. Fail loud at the
        coordinator instead."""
        for body in (
            {"model": "x", "messages": [{"role": "user", "content": ""}]},
            {"model": "x", "messages": [{"role": "user"}]},
            {"model": "x", "messages": [{"role": "user", "content": "   "}]},
        ):
            resp = client.post("/v1/chat/completions", json=body)
            assert resp.status_code == 400, f"accepted {body}"
            assert "non-empty content" in resp.json()["error"]["message"]

    def test_rejects_non_dict_message_items(self, client):
        """A `messages` list of strings (e.g. someone forgot to wrap
        in a dict) would crash inside the role lookup with a vague
        AttributeError. Reject at the boundary."""
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": ["hi", "there"]},
        )
        assert resp.status_code == 400
        assert "list of objects" in resp.json()["error"]["message"]

    def test_rejects_non_integer_max_tokens(self, client):
        """OpenAI clients pass `max_tokens` as an integer. Garbage values
        should fail loud with a structured 400 rather than crashing
        deep in the dispatch path."""
        for bad in ("notanumber", [], {}):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "x",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": bad,
                },
            )
            assert resp.status_code == 400, f"accepted max_tokens={bad!r}"
            assert "max_tokens" in resp.json()["error"]["message"]

    def test_rejects_multimodal_content_with_clear_message(self, client):
        """OpenAI's vision/audio shape uses `content` as a list of
        parts. Towel doesn't support multimodal — but the previous
        generic "non-empty content" 400 made vision clients think
        their content was empty. The error should name multimodal."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "..."}},
                    ],
                }],
            },
        )
        assert resp.status_code == 400
        assert "multimodal" in resp.json()["error"]["message"]
        assert "plain string" in resp.json()["error"]["message"]

    def test_rejects_non_string_model(self, client):
        """`model` is cosmetic but echoes into the response and SSE
        chunks. Non-string would render badly in JSON output."""
        for bad in (42, [1, 2], {"x": 1}):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": bad,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 400, f"accepted model={bad!r}"
            assert "model" in resp.json()["error"]["message"]

    def test_rejects_non_bool_stream(self, client):
        """`{"stream": "yes"}` would otherwise pass as truthy and
        silently take the streaming path. OpenAI's contract uses a
        boolean; non-bool should fail at the boundary."""
        for bad in ("yes", "false", 1, 0, [], {"x": 1}):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "x",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": bad,
                },
            )
            assert resp.status_code == 400, f"accepted stream={bad!r}"
            assert "true or false" in resp.json()["error"]["message"]

    def test_rejects_system_only_conversation(self, client):
        """A system-only conversation has no user prompt — most models
        hang or return empty rather than producing meaningful output,
        and the caller times out at 60s. Reject at the boundary."""
        for body in (
            {"model": "x", "messages": [{"role": "system", "content": "be terse"}]},
            {
                "model": "x",
                "messages": [
                    {"role": "system", "content": "you are a helper"},
                    {"role": "assistant", "content": "ok"},
                ],
            },
        ):
            resp = client.post("/v1/chat/completions", json=body)
            assert resp.status_code == 400, f"accepted {body}"
            assert "user turn" in resp.json()["error"]["message"]

    def test_rejects_non_numeric_temperature(self, client):
        for bad in ("hot", [], {}):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "x",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": bad,
                },
            )
            assert resp.status_code == 400, f"accepted temperature={bad!r}"
            assert "temperature" in resp.json()["error"]["message"]

    def test_rejects_non_object_top_level_body(self, client):
        """Top-level array/string/number body would crash on
        `body.get(...)` and surface as plaintext "Internal Server
        Error" HTTP 500 — breaking OpenAI clients that expect a
        structured 400."""
        for raw in (b"[1,2,3]", b'"hi"', b"42", b"true", b"null"):
            resp = client.post(
                "/v1/chat/completions",
                content=raw,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400, f"accepted {raw!r}"
            assert "JSON object" in resp.json()["error"]["message"]

    def test_validates_messages_field(self):
        """Verify messages are required and validated."""
        # No messages field -> 400
        from pathlib import Path

        from starlette.testclient import TestClient

        store = ConversationStore(store_dir=Path("/tmp/towel-test-oai"))
        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        c = TestClient(gw._build_http_app())

        # Missing messages -> 400
        assert c.post("/v1/chat/completions", json={"model": "x"}).status_code == 400
        # Empty messages -> 400
        assert (
            c.post("/v1/chat/completions", json={"model": "x", "messages": []}).status_code == 400
        )
        # Invalid body -> 400
        assert (
            c.post(
                "/v1/chat/completions", content="bad", headers={"content-type": "application/json"}
            ).status_code
            == 400
        )


class TestResponseFormat:
    def test_completion_format(self):
        from towel.gateway.openai_compat import _format_completion

        result = _format_completion("id-1", 1234567890, "test-model", "Hello!", 10)
        assert result["id"] == "id-1"
        assert result["object"] == "chat.completion"
        assert result["model"] == "test-model"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "usage" in result
        assert result["usage"]["completion_tokens"] == 10

    def test_completion_has_required_fields(self):
        from towel.gateway.openai_compat import _format_completion

        result = _format_completion("id", 0, "m", "content", 5)
        # All fields required by the OpenAI spec
        assert "id" in result
        assert "object" in result
        assert "created" in result
        assert "model" in result
        assert "choices" in result
        assert "usage" in result
        choice = result["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

    def test_completion_includes_system_fingerprint(self):
        """`system_fingerprint` is what OpenAI clients use for cache
        invalidation — some LangChain caches and eval harnesses
        depend on it. Towel emits a per-process value derived from
        the package version so the fingerprint flips on coordinator
        upgrades (matching OpenAI's behaviour of changing it on
        model revisions) but stays stable within a process."""
        from towel.gateway.openai_compat import _format_completion

        result = _format_completion("id", 0, "m", "content", 5)
        assert "system_fingerprint" in result
        fp = result["system_fingerprint"]
        assert isinstance(fp, str)
        assert fp.startswith("fp_")
        # Second call within the same process must produce the same
        # value — caches keyed on this would thrash otherwise.
        result2 = _format_completion("id-2", 1, "m", "different", 3)
        assert result2["system_fingerprint"] == fp

    def test_format_completion_no_towel_field(self):
        """The `towel`-namespaced metadata is attached by the
        chat_completions handler, NOT by _format_completion itself.
        A direct call to _format_completion must produce the strict
        OpenAI shape — no `towel` field — so plain-vanilla
        ChatCompletion responses don't carry vendor noise when
        nothing collab-related happened."""
        from towel.gateway.openai_compat import _format_completion

        result = _format_completion("id", 0, "m", "content", 5)
        assert "towel" not in result

    def test_prompt_tokens_uses_supplied_value(self):
        """When the caller provides prompt_tokens (e.g. from the worker's
        usage data, or an estimate over the input messages), the formatter
        must report it rather than fabricating one from completion_tokens.

        The previous implementation did `prompt_tokens = completion // 4`,
        which gave 0 for empty responses no matter how long the input was —
        meaningless usage data for any OpenAI client tracking spend."""
        from towel.gateway.openai_compat import _format_completion

        result = _format_completion(
            "id", 0, "m", "content", 5, prompt_tokens=42,
        )
        assert result["usage"]["prompt_tokens"] == 42
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 47

    def test_prompt_tokens_default_is_independent_of_completion(self):
        """Without an explicit prompt_tokens the formatter falls back to 1,
        not to a value derived from completion length."""
        from towel.gateway.openai_compat import _format_completion

        result = _format_completion("id", 0, "m", "content", 9999)
        assert result["usage"]["prompt_tokens"] == 1


class TestNoneMetadataHandling:
    """Workers occasionally emit `tokens: None` / `output_tokens:
    None` / `prompt_tokens: None` after a job_error or empty-text
    fallback. The /v1/chat/completions non-streaming response
    builder must not crash on those — `prompt + completion` in
    _format_completion raises TypeError on None, turning a
    recoverable empty-text path into HTTP 500."""

    def test_non_stream_completion_handles_none_tokens(self, client):
        """A response with `tokens: None` in metadata must produce a
        well-formed completion response with the back-estimated count."""
        from towel.agent.conversation import Message, Role

        # Stub the agent's step to return a response with None token
        # counts in metadata — what a worker job_error or empty-text
        # path would produce.
        async def fake_step(conv):
            return Message(
                role=Role.ASSISTANT,
                content="hi there",
                metadata={"tokens": None, "prompt_tokens": None},
            )

        # The client fixture already wires a gateway + agent.
        # Find the agent and patch step.
        # Note: we can't easily patch the local agent from the client
        # fixture, so this test exercises the boundary via the
        # _format_completion code path directly.
        from towel.gateway.openai_compat import _format_completion
        # completion_tokens=0 (what coercion produces for None);
        # _format_completion must compute total = 1 + 0 = 1 with no
        # TypeError.
        result = _format_completion(
            "id", 0, "m", "hi there", 0, prompt_tokens=None,
        )
        assert result["usage"]["prompt_tokens"] == 1
        assert result["usage"]["completion_tokens"] == 0
        assert result["usage"]["total_tokens"] == 1


class TestSSEFormat:
    @pytest.mark.asyncio
    async def test_local_stream_error_event_emits_error_chunk(self):
        """When the agent emits AgentEvent.error(msg) — its graceful
        in-stream-failure signal — the SSE generator must surface
        it as a `finish_reason="error"` chunk instead of silently
        dropping the event. Previously the if/elif chain had no
        ERROR branch so the stream ended with a bare [DONE] and
        the client saw "successfully empty"."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Conversation, Role
        from towel.agent.events import AgentEvent
        from towel.gateway.openai_compat import _stream_sse

        agent = MagicMock()
        conv = Conversation()
        conv.add(Role.USER, "hi")

        async def erroring_stream(c):
            yield AgentEvent.token("partial-")
            yield AgentEvent.error("model wedged on prompt")
        agent.step_streaming = erroring_stream

        chunks = []
        async for chunk in _stream_sse(agent, conv, "rid", 0, "m"):
            chunks.append(chunk)

        joined = "".join(chunks)
        assert "partial-" in joined
        assert "finish_reason\": \"error\"" in joined
        # The agent's message text reaches the client via the error
        # frame's message field.
        err_frame = next(c for c in chunks if "finish_reason\": \"error\"" in c)
        payload = json.loads(err_frame.replace("data: ", ""))
        assert "model wedged on prompt" in payload["error"]["message"]
        assert payload["error"]["type"] == "server_error"
        # Stream still terminates with [DONE].
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_local_stream_exception_emits_error_chunk(self):
        """When agent.step_streaming raises mid-iteration, the SSE
        generator must still emit a final `finish_reason="error"`
        chunk + [DONE]. Previously the exception propagated up and
        clients waiting for [DONE] before flushing would hang."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Conversation, Role
        from towel.agent.events import AgentEvent
        from towel.gateway.openai_compat import _stream_sse

        agent = MagicMock()
        conv = Conversation()
        conv.add(Role.USER, "hi")

        async def crashing_stream(c):
            yield AgentEvent.token("partial-")
            raise RuntimeError("model wedged")
        agent.step_streaming = crashing_stream

        chunks = []
        async for chunk in _stream_sse(agent, conv, "rid", 0, "m"):
            chunks.append(chunk)

        joined = "".join(chunks)
        # The partial token came through.
        assert "partial-" in joined
        # And the failure produced a terminator pair.
        assert "finish_reason\": \"error\"" in joined
        assert chunks[-1] == "data: [DONE]\n\n"
        # Error frame carries the type field (parity with the
        # non-streaming 500 path and _stream_sse_remote's error
        # chunks).
        err_frame = next(c for c in chunks if "finish_reason\": \"error\"" in c)
        payload = json.loads(err_frame.replace("data: ", ""))
        assert payload["error"]["type"] == "server_error"

    @pytest.mark.asyncio
    async def test_sse_stream_format(self):
        """Test the SSE generator produces valid format."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Conversation, Role
        from towel.agent.events import AgentEvent
        from towel.gateway.openai_compat import _stream_sse

        # Mock agent that yields token events
        agent = MagicMock()
        conv = Conversation()
        conv.add(Role.USER, "hi")

        async def mock_stream(c):
            yield AgentEvent.token("Hello")
            yield AgentEvent.token(" world")
            yield AgentEvent.complete("Hello world", {"tokens": 5})

        agent.step_streaming = mock_stream

        chunks = []
        async for chunk in _stream_sse(agent, conv, "test-id", 123, "model"):
            chunks.append(chunk)

        # 5 chunks: role announce + 2 tokens + finish + [DONE].
        # The role chunk is part of OpenAI's documented streaming
        # protocol — clients keying on `delta.role` to start an
        # assistant turn were misbehaving without it.
        assert len(chunks) == 5
        assert chunks[0].startswith("data: ")
        # First chunk is the role announcement (no content yet).
        role_payload = json.loads(chunks[0].replace("data: ", ""))
        assert role_payload["choices"][0]["delta"] == {"role": "assistant"}
        # First content chunk is at index 1.
        assert '"Hello"' in chunks[1]
        assert '"Hello world"' not in chunks[1]  # streaming sends individual tokens
        assert chunks[-1] == "data: [DONE]\n\n"

        # Parse the finish chunk
        finish_data = json.loads(chunks[-2].replace("data: ", ""))
        assert finish_data["choices"][0]["finish_reason"] == "stop"

        # Every data chunk must include system_fingerprint —
        # parity with the non-streaming completion. OpenAI's
        # streaming format includes the field on each chunk; some
        # cache layers key on it per-chunk.
        from towel.gateway.openai_compat import _SYSTEM_FINGERPRINT
        for c in chunks[:-1]:  # all except trailing [DONE]
            payload = json.loads(c.replace("data: ", ""))
            assert payload.get("system_fingerprint") == _SYSTEM_FINGERPRINT, c


class TestRemoteStreamFallback:
    """When a remote worker errors at the start of a stream, the
    coordinator should fall back to local agent streaming so SSE
    clients still get a useful response. Observed live: workers
    running pre-fix code occasionally return 400 from their own
    llama-server on streaming requests, which would otherwise leave
    the SSE client with finish_reason=error and no content."""

    @pytest.mark.asyncio
    async def test_falls_back_to_local_when_remote_errors_before_first_token(self):
        from unittest.mock import MagicMock

        from towel.agent.conversation import Conversation, Role
        from towel.agent.events import AgentEvent
        from towel.gateway.openai_compat import _stream_sse_remote

        async def failing_remote(session_id, session, worker):
            raise RuntimeError("worker llama-server returned 400")
            yield  # pragma: no cover — make this an async generator

        gateway = MagicMock()
        gateway.iter_remote_tokens = failing_remote

        async def local_stream(c):
            yield AgentEvent.token("local-")
            yield AgentEvent.token("fallback")
            yield AgentEvent.complete("local-fallback", {"tokens": 2})

        agent = MagicMock()
        agent.step_streaming = local_stream

        conv = Conversation()
        conv.add(Role.USER, "hi")

        chunks = []
        async for chunk in _stream_sse_remote(
            gateway, "sid", MagicMock(), MagicMock(),
            "rid", 0, "m",
            fallback_agent=agent,
            fallback_conv=conv,
        ):
            chunks.append(chunk)

        # Tokens from local agent + finish + [DONE].
        joined = "".join(chunks)
        assert "local-" in joined
        assert "fallback" in joined
        # No finish_reason=error chunk should leak through when the
        # fallback runs successfully.
        assert "\"error\"" not in joined
        assert "[DONE]" in joined

    @pytest.mark.asyncio
    async def test_role_chunk_emitted_only_once_through_fallback(self):
        """When _stream_sse_remote fails before any token AND falls
        back to the local agent's _stream_sse, the client must see
        exactly ONE role-announcement chunk — not one from the
        remote path AND one from the local fallback. Two role
        declarations on the same response confuse SDKs that key on
        the role chunk to start a new assistant turn."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Conversation, Role
        from towel.agent.events import AgentEvent
        from towel.gateway.openai_compat import _stream_sse_remote

        async def failing_remote(session_id, session, worker):
            raise RuntimeError("worker llama-server returned 400")
            yield  # pragma: no cover

        gateway = MagicMock()
        gateway.iter_remote_tokens = failing_remote

        async def local_stream(c):
            yield AgentEvent.token("local-")
            yield AgentEvent.complete("local-", {"tokens": 1})

        agent = MagicMock()
        agent.step_streaming = local_stream

        conv = Conversation()
        conv.add(Role.USER, "hi")

        chunks = []
        async for chunk in _stream_sse_remote(
            gateway, "sid", MagicMock(), MagicMock(),
            "rid", 0, "m",
            fallback_agent=agent,
            fallback_conv=conv,
        ):
            chunks.append(chunk)

        role_chunks = [
            c for c in chunks
            if '"role": "assistant"' in c or '"role":"assistant"' in c
        ]
        assert len(role_chunks) == 1, (
            f"expected exactly 1 role chunk; got {len(role_chunks)}: {role_chunks}"
        )

    @pytest.mark.asyncio
    async def test_mid_stream_error_still_surfaces(self):
        """Once at least one token was emitted the fallback can't take
        over silently — we don't replay tokens. The client should see
        finish_reason=error so it knows something went wrong."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Conversation, Role
        from towel.gateway.openai_compat import _stream_sse_remote

        async def partial_remote(session_id, session, worker):
            yield "first-"
            yield "second-"
            raise RuntimeError("connection dropped")

        gateway = MagicMock()
        gateway.iter_remote_tokens = partial_remote

        conv = Conversation()
        conv.add(Role.USER, "hi")

        chunks = []
        async for chunk in _stream_sse_remote(
            gateway, "sid", MagicMock(), MagicMock(),
            "rid", 0, "m",
            fallback_agent=MagicMock(),
            fallback_conv=conv,
        ):
            chunks.append(chunk)

        joined = "".join(chunks)
        assert "first-" in joined
        assert "second-" in joined
        assert '"error"' in joined or "finish_reason\": \"error\"" in joined
        # The error frame must carry a `type` alongside `message` —
        # OpenAI clients switch on the field, so omitting it raised
        # `KeyError: 'type'` deep in the SDK.
        err_frame = next(c for c in chunks if "finish_reason\": \"error\"" in c)
        payload = json.loads(err_frame.replace("data: ", ""))
        assert payload["error"]["type"] == "server_error"


class TestCollaborationOnOpenAICompat:
    """The verify/ensemble collaboration primitives are also reachable
    via /v1/chat/completions so OpenAI-API clients (LangChain,
    llm-cli, OpenAI SDK with extra_body=) can opt into multi-worker
    flows without changing endpoint. Streaming intentionally rejects
    these modes — the synthesis step is non-streaming."""

    def test_verify_and_ensemble_mutually_exclusive(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "verify": True,
                "ensemble": True,
            },
        )
        assert resp.status_code == 400
        assert "mutually exclusive" in resp.json()["error"]["message"]

    def test_verify_must_be_bool(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "verify": "yes",
            },
        )
        assert resp.status_code == 400
        assert "verify" in resp.json()["error"]["message"]

    def test_ensemble_must_be_bool(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "ensemble": 1,
            },
        )
        assert resp.status_code == 400
        assert "ensemble" in resp.json()["error"]["message"]

    def test_ensemble_returns_arbitrated_answer(self, tmp_path):
        """End-to-end ensemble through /v1/chat/completions:
        non-streaming request with `ensemble: true` fans the prompt
        to every idle worker and returns the arbitrated answer in
        standard OpenAI shape. The collaboration is invisible to a
        spec-strict OpenAI client (just better quality), but works."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        gw._workers.register(
            "a", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gw._workers.register(
            "b", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            answers = {
                "a": "Paris is the capital of France.",
                "b": "Paris, capital of France, is also its largest city.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 8, "tps": 5.0},
            )

        gw._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "ensemble": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Standard OpenAI shape — collaboration is invisible at the
        # response shape level.
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "Paris" in data["choices"][0]["message"]["content"]
        # Towel-specific metadata rides under a `towel` key so
        # spec-strict OpenAI clients can ignore it but Towel-aware
        # ones see whether ensemble actually ran.
        assert data.get("towel", {}).get("ensemble") is True

    def test_fallback_from_worker_surfaces_in_towel_field(self, tmp_path):
        """When the primary worker returns empty text and the
        coordinator successfully retries on an alternate, the
        response carries `towel.fallback_from_worker` +
        `fallback_reason` so OpenAI-aware clients can see the
        retry happened. Without this, an answer that came from
        the alt looks like a normal primary response."""
        from unittest.mock import AsyncMock

        from towel.agent.conversation import Message, Role
        from towel.gateway.workers import WorkerInfo

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        primary = WorkerInfo(id="fb-primary", ws=AsyncMock(), capabilities={})
        alt = WorkerInfo(
            id="fb-alt", ws=AsyncMock(),
            capabilities={"total_vram_mb": 16000},
        )
        gw._workers._workers["fb-primary"] = primary
        gw._workers._workers["fb-alt"] = alt

        async def fake_route(_msg, _sid):
            return primary, "chat"
        gw._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            if worker.id == "fb-primary":
                return Message(
                    role=Role.ASSISTANT,
                    content="(empty text placeholder)",
                    metadata={
                        "remote_worker": worker.id,
                        "empty_text_fallback": True,
                    },
                )
            # Alt returns a real answer.
            return Message(
                role=Role.ASSISTANT,
                content="real answer from alt",
                metadata={"remote_worker": worker.id, "tokens": 4, "tps": 5.0},
            )
        gw._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # The successful retry's answer is the response content.
        assert "real answer from alt" in data["choices"][0]["message"]["content"]
        # Towel-namespaced fields surface the retry path.
        towel_meta = data.get("towel", {})
        assert towel_meta.get("fallback_from_worker") == "fb-primary"
        assert towel_meta.get("fallback_reason") == "empty_text"

    def test_dual_empty_text_surfaces_in_towel_field(self, tmp_path):
        """When both the primary and the retry worker return the
        empty-text fallback (every worker tool-loops on the prompt),
        the openai-compat response keeps the primary's placeholder
        but the `towel.dual_empty_text` field signals the fleet-wide
        condition. Clients can render "both workers tool-looped"
        instead of treating it like a one-worker miss."""
        from unittest.mock import AsyncMock

        from towel.agent.conversation import Message, Role
        from towel.gateway.workers import WorkerInfo

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        primary = WorkerInfo(id="dual-primary", ws=AsyncMock(), capabilities={})
        alt = WorkerInfo(
            id="dual-alt", ws=AsyncMock(),
            capabilities={"total_vram_mb": 16000},
        )
        gw._workers._workers["dual-primary"] = primary
        gw._workers._workers["dual-alt"] = alt

        async def fake_route(_msg, _sid):
            return primary, "chat"
        gw._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            msg = Message(
                role=Role.ASSISTANT,
                content="(empty text placeholder)",
                metadata={
                    "remote_worker": worker.id,
                    "empty_text_fallback": True,
                },
            )
            session.conversation.messages.append(msg)
            return msg
        gw._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        towel_meta = data.get("towel", {})
        assert towel_meta.get("dual_empty_text") is True
        assert towel_meta.get("alt_worker") == "dual-alt"

    def test_verify_skipped_surfaces_in_towel_field(self, tmp_path):
        """When verify=true is requested but only one worker is
        registered, the verify pass has no alternate to run on
        and falls through. The response must carry
        `towel.verify_skipped: true` + `verify_skip_reason` so
        OpenAI-aware clients see the same degraded-state signal
        they would on /api/ask."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        # Single worker → verify can't find an alternate.
        gw._workers.register(
            "solo", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"], "tools": False},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            msg = Message(
                role=Role.ASSISTANT, content="primary answer",
                metadata={"remote_worker": worker.id, "tokens": 2, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg
        gw._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return gw._workers.get("solo"), "chat"
        gw._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "verify": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary answer" in data["choices"][0]["message"]["content"]
        towel_meta = data.get("towel", {})
        assert towel_meta.get("verify_skipped") is True
        assert "no alternate worker" in towel_meta.get("verify_skip_reason", "")

    def test_tools_param_surfaces_tools_ignored_flag(self, tmp_path):
        """OpenAI's ``tools`` parameter is for client-supplied
        function schemas. Towel doesn't implement function-call
        passthrough yet — but rejecting with 400 would break
        clients (langchain, openai-python with structured output)
        that pass ``tools`` defensively even when not needed. Log
        a server warning AND surface ``towel.tools_ignored: true``
        so callers can detect the unsupported-feature path."""
        from towel.agent.conversation import Message, Role

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        async def fake_step(_conv, **_kwargs):
            return Message(role=Role.ASSISTANT, content="text answer")
        agent.step = fake_step  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return None, "chat"
        gw._route_by_role = fake_route  # type: ignore[method-assign]

        # tools param present → flag surfaces.
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "description": "Get current weather",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json().get("towel", {}).get("tools_ignored") is True

        # Empty tools list → no flag (client said "no tools").
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
            },
        )
        assert resp.status_code == 200
        assert "tools_ignored" not in resp.json().get("towel", {})

        # No tools key at all → no flag.
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert "tools_ignored" not in resp.json().get("towel", {})

    def test_ensemble_skipped_surfaces_in_towel_field(self, tmp_path):
        """When ensemble=true is requested but no idle inference
        workers exist, the request falls through to the local agent
        path. The response must carry `towel.ensemble_skipped: true`
        so OpenAI-aware clients can render the same degraded-state
        badge they would on /api/ask."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role
        from towel.agent.runtime import GenerationResult

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        # No workers registered → ensemble_dispatch returns 0
        # contributions → falls through. Local agent path needs
        # a stub since the real one isn't loaded.
        async def fake_step(_conv, **_kwargs):
            return Message(role=Role.ASSISTANT, content="local fallback answer")
        agent.step = fake_step  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return None, "chat"
        gw._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "hi"}],
                "ensemble": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # Local-agent fallback answered; ensemble didn't actually run.
        assert "local fallback" in data["choices"][0]["message"]["content"]
        # Towel-namespaced field surfaces the silent degradation.
        towel = data.get("towel", {})
        assert towel.get("ensemble_skipped") is True
        # No-candidates case carries the matching skip_reason.
        assert "no idle workers" in towel.get("ensemble_skip_reason", "")
        # Empty contributions list still surfaced — parity with
        # /api/ask, where the field is always present so clients
        # don't special-case the missing-field path.
        assert towel.get("ensemble_contributions") == []

    def test_ensemble_skipped_reasons_surface_for_each_failure_mode(
        self, tmp_path,
    ):
        """OpenAI-compat parity with /api/ask: when ensemble fan-out
        returns contributions but no usable answer, the towel-
        namespaced block must distinguish empty_text vs timeout vs
        mixed failures and surface the per-worker contributions list
        so OpenAI-aware clients diagnose without curl-ing the
        dispatch log."""
        from towel.agent.conversation import Message, Role

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        async def fake_step(_conv, **_kwargs):
            return Message(role=Role.ASSISTANT, content="local fallback answer")
        agent.step = fake_step  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return None, "chat"
        gw._route_by_role = fake_route  # type: ignore[method-assign]

        scenarios = [
            (
                [
                    {"worker_id": "x", "answer": "", "ms": 10.0, "error": "empty_text"},
                    {"worker_id": "y", "answer": "", "ms": 10.0, "error": "empty_text"},
                ],
                "tool-looped",
            ),
            (
                [
                    {"worker_id": "x", "answer": "", "ms": 90.0, "error": "ensemble_timeout"},
                    {"worker_id": "y", "answer": "", "ms": 90.0, "error": "ensemble_timeout"},
                ],
                "timed out",
            ),
            (
                [
                    {"worker_id": "x", "answer": "", "ms": 10.0, "error": "empty_text"},
                    {"worker_id": "y", "answer": "", "ms": 90.0, "error": "ensemble_timeout"},
                ],
                "mixed failures",
            ),
        ]

        for contributions, expected_phrase in scenarios:
            async def fake_ensemble(*_args, **_kwargs):
                return "", contributions, "none"
            gw._ensemble_dispatch = fake_ensemble  # type: ignore[method-assign]

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "hi"}],
                    "ensemble": True,
                },
            )
            assert resp.status_code == 200
            towel = resp.json().get("towel", {})
            assert towel.get("ensemble_skipped") is True, towel
            assert expected_phrase in towel.get("ensemble_skip_reason", ""), (
                f"expected {expected_phrase!r}; got "
                f"{towel.get('ensemble_skip_reason')!r}"
            )
            assert towel.get("ensemble_contributions") == contributions

    def test_verify_corrects_through_openai_compat(self, tmp_path):
        """End-to-end verify through /v1/chat/completions: the
        primary worker generates an answer, the alternate verifies,
        and a substantive correction replaces the primary's content
        in choices[0].message.content. Same contract /api/ask has,
        just reachable via OpenAI clients with extra_body=verify."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        gw._workers.register(
            "primary", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"], "tools": False},
        )
        gw._workers.register(
            "verifier", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"], "tools": False},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            # Verifier sessions are prefixed `_verify_`.
            if session_id.startswith("_verify_"):
                # Substantive correction: must be > 30 chars to not
                # be treated as a confirmation token.
                return Message(
                    role=Role.ASSISTANT,
                    content="The capital of Germany is Berlin.",
                    metadata={"remote_worker": worker.id, "tokens": 12, "tps": 5.0},
                )
            msg = Message(
                role=Role.ASSISTANT, content="The wrong answer is Paris.",
                metadata={"remote_worker": worker.id, "tokens": 7, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg

        gw._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return gw._workers.get("primary"), "chat"

        gw._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Capital of Germany?"}],
                "verify": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        content = data["choices"][0]["message"]["content"]
        # Verifier's correction wins, not primary's wrong answer.
        assert "Berlin" in content
        assert "Paris" not in content

    def test_streaming_rejects_collaboration_modes(self, client):
        """Streaming can't carry synthesis (the arbiter waits for all
        contributions, which is inherently non-streaming). Reject
        explicitly rather than silently degrade to non-collaboration."""
        for field in ("verify", "ensemble"):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    field: True,
                },
            )
            assert resp.status_code == 400, f"accepted stream+{field}"
            assert "stream=false" in resp.json()["error"]["message"]


class TestEphemeralSessionCleanup:
    """OpenAI-compat creates a one-shot session_id per request
    (`openai-<random>`). Without cleanup every call leaked both an
    affinity entry in _session_workers AND a context slot on the
    routed worker — inflating context_pressure forever for any
    coordinator that fronted /v1/chat/completions traffic."""

    def test_cleanup_ephemeral_session_helper(self, tmp_path):
        """The helper closes affinity + slot AND drops the in-memory
        Session. Safe to call on a session that doesn't exist."""
        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)

        # Set up: register a worker + node, then create the kind of
        # ephemeral state OpenAI-compat would leave behind.
        gw._workers.register(
            "node-a", object(),
            {
                "backend": "llama", "modes": ["llama_chat"],
                "context_window": 8192, "max_tokens": 4096,
                "total_vram_mb": 16000,
                "resources": {"hostname": "node-a", "ram_total_mb": 32000},
            },
        )
        gw._node_tracker.register(
            "node-a", gw._workers.get("node-a").capabilities,
        )
        gw._session_workers["openai-abc123"] = "node-a"
        gw._node_tracker.open_context_slot("node-a", "openai-abc123", 100)
        gw.sessions.get_or_create("openai-abc123")

        # Sanity pre-cleanup.
        assert "openai-abc123" in gw._session_workers
        assert gw._node_tracker.get("node-a").get_context_slot(
            "openai-abc123"
        ) is not None

        gw.cleanup_ephemeral_session("openai-abc123")

        # Everything's gone.
        assert "openai-abc123" not in gw._session_workers
        assert gw._node_tracker.get("node-a").get_context_slot(
            "openai-abc123"
        ) is None
        assert gw.sessions.get("openai-abc123") is None

    def test_cleanup_idempotent_on_unknown_session(self, tmp_path):
        """Cleaning a session that was never created (early-fail
        path before the gateway routed anywhere) must be a no-op,
        not a crash."""
        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)

        # No exception.
        gw.cleanup_ephemeral_session("never-existed")

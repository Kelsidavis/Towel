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

        assert len(chunks) == 4  # 2 tokens + finish + [DONE]
        assert chunks[0].startswith("data: ")
        assert '"Hello"' in chunks[0]
        assert '"Hello world"' not in chunks[0]  # streaming sends individual tokens
        assert chunks[-1] == "data: [DONE]\n\n"

        # Parse the finish chunk
        finish_data = json.loads(chunks[-2].replace("data: ", ""))
        assert finish_data["choices"][0]["finish_reason"] == "stop"


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

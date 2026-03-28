"""Tests for the OpenAI-compatible API endpoint."""

import json

import pytest
from starlette.testclient import TestClient

from towel.config import TowelConfig
from towel.agent.runtime import AgentRuntime
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
        assert c.post("/v1/chat/completions", json={"model": "x", "messages": []}).status_code == 400
        # Invalid body -> 400
        assert c.post("/v1/chat/completions", content="bad", headers={"content-type": "application/json"}).status_code == 400


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


class TestSSEFormat:
    @pytest.mark.asyncio
    async def test_sse_stream_format(self):
        """Test the SSE generator produces valid format."""
        from towel.gateway.openai_compat import _stream_sse
        from towel.agent.conversation import Conversation, Role
        from towel.agent.events import AgentEvent, EventType
        from unittest.mock import AsyncMock, MagicMock

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

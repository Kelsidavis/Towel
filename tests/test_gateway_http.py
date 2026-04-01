"""Tests for the gateway HTTP endpoints and web UI serving."""

import pytest
from starlette.testclient import TestClient

from towel.agent.conversation import Conversation, Role
from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.store import ConversationStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


@pytest.fixture
def gateway(store):
    config = TowelConfig()
    agent = AgentRuntime(config)
    sessions = SessionManager(store=store)
    return GatewayServer(config=config, agent=agent, sessions=sessions)


@pytest.fixture
def client(gateway):
    app = gateway._build_http_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "hoopy"
        assert data["motto"] == "Don't Panic."
        assert "version" in data
        assert "connections" in data
        assert "sessions" in data

    def test_health_shows_connection_count(self, client):
        data = client.get("/health").json()
        assert data["connections"] == 0
        assert data["sessions"] == 0


class TestSessionsEndpoint:
    def test_sessions_empty(self, client):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []

    def test_sessions_after_create(self, gateway, client):
        gateway.sessions.get_or_create("test-session")
        data = client.get("/sessions").json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["id"] == "test-session"


class TestWebUI:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "TOWEL" in resp.text
        assert "Don't Panic" in resp.text

    def test_index_has_websocket_js(self, client):
        resp = client.get("/")
        assert "WebSocket" in resp.text
        assert "ws://" in resp.text

    def test_index_has_streaming_handlers(self, client):
        resp = client.get("/")
        # Verify all event types are handled in the JS
        assert "'token'" in resp.text
        assert "'tool_call'" in resp.text
        assert "'tool_result'" in resp.text
        assert "'response_complete'" in resp.text
        assert "'error'" in resp.text

    def test_index_has_chat_input(self, client):
        resp = client.get("/")
        assert "user-input" in resp.text
        assert "send-btn" in resp.text

    def test_index_has_sidebar(self, client):
        resp = client.get("/")
        assert "sidebar" in resp.text
        assert "conv-list" in resp.text
        assert "new-chat-btn" in resp.text

    def test_index_has_localstorage(self, client):
        resp = client.get("/")
        assert "localStorage" in resp.text

    def test_index_has_themes(self, client):
        resp = client.get("/")
        assert "deep-space" in resp.text
        assert "frost" in resp.text
        assert "matrix" in resp.text
        assert "solarized" in resp.text
        assert "towel-theme" in resp.text
        assert "theme-btn" in resp.text

    def test_index_has_command_palette(self, client):
        resp = client.get("/")
        assert "cmd-palette" in resp.text
        assert "cmd-input" in resp.text
        assert "cmd-results" in resp.text
        assert "openPalette" in resp.text

    def test_index_has_toolbar(self, client):
        resp = client.get("/")
        assert "toolbar" in resp.text
        assert "tb-export" in resp.text
        assert "tb-delete" in resp.text

    def test_index_has_delete_button_on_conversations(self, client):
        resp = client.get("/")
        assert "conv-del" in resp.text
        assert "deleteConversation" in resp.text

    def test_index_has_markdown_renderer(self, client):
        resp = client.get("/")
        assert "renderMarkdown" in resp.text
        assert "md-content" in resp.text
        assert "towel-session" in resp.text


class TestConversationsAPI:
    def test_list_empty(self, client):
        resp = client.get("/conversations")
        assert resp.status_code == 200
        assert resp.json()["conversations"] == []

    def test_list_with_data(self, store, client):
        conv = Conversation(id="test-1", channel="cli")
        conv.add(Role.USER, "hello")
        store.save(conv)

        data = client.get("/conversations").json()
        assert len(data["conversations"]) == 1
        assert data["conversations"][0]["id"] == "test-1"
        assert data["conversations"][0]["message_count"] == 1

    def test_get_conversation(self, store, client):
        conv = Conversation(id="detail-1", channel="webchat")
        conv.add(Role.USER, "question")
        conv.add(Role.ASSISTANT, "answer")
        store.save(conv)

        resp = client.get("/conversations/detail-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "detail-1"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"

    def test_get_nonexistent(self, client):
        resp = client.get("/conversations/nope")
        assert resp.status_code == 404

    def test_delete_conversation(self, store, client):
        conv = Conversation(id="del-1")
        conv.add(Role.USER, "bye")
        store.save(conv)

        resp = client.request("DELETE", "/conversations/del-1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert not store.exists("del-1")

    def test_delete_nonexistent(self, client):
        resp = client.request("DELETE", "/conversations/nope")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is False

    def test_export_markdown(self, store, client):
        conv = Conversation(id="exp-1", channel="cli")
        conv.add(Role.USER, "hello")
        conv.add(Role.ASSISTANT, "hi there")
        store.save(conv)

        resp = client.get("/conversations/exp-1/export")
        assert resp.status_code == 200
        assert "text/markdown" in resp.headers["content-type"]
        assert "### You" in resp.text
        assert "### Towel" in resp.text
        assert "attachment" in resp.headers.get("content-disposition", "")

    def test_export_json(self, store, client):
        conv = Conversation(id="exp-2")
        conv.add(Role.USER, "test")
        store.save(conv)

        resp = client.get("/conversations/exp-2/export?format=json")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        data = resp.json()
        assert data["id"] == "exp-2"

    def test_export_text(self, store, client):
        conv = Conversation(id="exp-3")
        conv.add(Role.USER, "test")
        store.save(conv)

        resp = client.get("/conversations/exp-3/export?format=text")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "[you]" in resp.text

    def test_export_nonexistent(self, client):
        resp = client.get("/conversations/nope/export")
        assert resp.status_code == 404


class TestSimpleAskAPI:
    def test_ask_missing_message(self, client):
        resp = client.post("/api/ask", json={})
        assert resp.status_code == 400
        assert "message" in resp.json()["error"]

    def test_ask_empty_message(self, client):
        resp = client.post("/api/ask", json={"message": ""})
        assert resp.status_code == 400

    def test_ask_invalid_json(self, client):
        resp = client.post(
            "/api/ask", content=b"not json", headers={"content-type": "application/json"}
        )
        assert resp.status_code == 400

    def test_ask_creates_session(self, gateway, client):
        # The actual model call will fail (no model loaded), but we test the session creation
        _resp = client.post("/api/ask", json={"message": "hello", "session": "test-ask"})
        # Will be 500 (model not loaded) but session should exist
        session = gateway.sessions.get_or_create("test-ask")
        assert len(session.conversation) >= 1  # at least the user message


class TestApiSessions:
    def test_api_sessions_empty(self, client):
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []

    def test_api_sessions_with_tags(self, store, client):
        conv = Conversation(id="tagged-1", channel="api")
        conv.tags = ["work", "urgent"]
        conv.add(Role.USER, "hello")
        store.save(conv)

        resp = client.get("/api/sessions")
        data = resp.json()
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["tags"] == ["work", "urgent"]

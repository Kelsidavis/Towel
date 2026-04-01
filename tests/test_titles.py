"""Tests for conversation titles."""

from towel.agent.conversation import Conversation, Role
from towel.persistence.store import ConversationStore


class TestConversationTitle:
    def test_default_empty_title(self):
        conv = Conversation()
        assert conv.title == ""

    def test_display_title_uses_title(self):
        conv = Conversation(title="My Research")
        assert conv.display_title == "My Research"

    def test_display_title_falls_back_to_summary(self):
        conv = Conversation()
        conv.add(Role.USER, "How do I deploy to AWS?")
        assert conv.display_title == "How do I deploy to AWS?"

    def test_display_title_empty_no_messages(self):
        conv = Conversation()
        assert conv.display_title == "(empty)"

    def test_title_serialization_roundtrip(self):
        conv = Conversation(title="API Research")
        conv.add(Role.USER, "hello")
        d = conv.to_dict()
        assert d["title"] == "API Research"
        restored = Conversation.from_dict(d)
        assert restored.title == "API Research"

    def test_no_title_key_in_json_when_empty(self):
        conv = Conversation()
        d = conv.to_dict()
        assert "title" not in d

    def test_from_dict_without_title(self):
        """Backwards compat — old conversations without title field."""
        data = {
            "id": "old-conv",
            "channel": "cli",
            "created_at": "2026-01-01T00:00:00+00:00",
            "messages": [],
        }
        conv = Conversation.from_dict(data)
        assert conv.title == ""

    def test_summary_strips_file_refs(self):
        conv = Conversation()
        conv.add(Role.USER, "explain @src/main.py please")
        assert "@" not in conv.summary
        assert "explain" in conv.summary
        assert "please" in conv.summary


class TestStoreRename:
    def test_rename(self, tmp_path):
        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="rename-1")
        conv.add(Role.USER, "hello")
        store.save(conv)

        assert store.rename("rename-1", "My Chat")
        loaded = store.load("rename-1")
        assert loaded is not None
        assert loaded.title == "My Chat"

    def test_rename_nonexistent(self, tmp_path):
        store = ConversationStore(store_dir=tmp_path)
        assert not store.rename("nope", "Title")

    def test_list_shows_title(self, tmp_path):
        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="titled-1", title="API Research")
        conv.add(Role.USER, "something about APIs")
        store.save(conv)

        summaries = store.list_conversations()
        assert len(summaries) == 1
        assert summaries[0].title == "API Research"
        assert summaries[0].summary == "API Research"  # display_title uses title


class TestGatewayRename:
    def test_rename_endpoint(self, tmp_path):
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="gw-rename")
        conv.add(Role.USER, "test")
        store.save(conv)

        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        client = TestClient(gw._build_http_app())

        resp = client.post(
            "/conversations/gw-rename/rename",
            json={"title": "Renamed via API"},
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "Renamed via API"

        # Verify persisted
        loaded = store.load("gw-rename")
        assert loaded is not None
        assert loaded.title == "Renamed via API"

    def test_rename_nonexistent(self, tmp_path):
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        client = TestClient(gw._build_http_app())

        resp = client.post(
            "/conversations/nope/rename",
            json={"title": "Won't work"},
        )
        assert resp.status_code == 404

    def test_rename_empty_title_rejected(self, tmp_path):
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        client = TestClient(gw._build_http_app())

        resp = client.post(
            "/conversations/x/rename",
            json={"title": ""},
        )
        assert resp.status_code == 400

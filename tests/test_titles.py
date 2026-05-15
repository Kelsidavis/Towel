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

    def test_rename_updates_in_memory_session(self, tmp_path):
        """If a session is loaded in memory (mid-conversation) when
        a rename comes in, the in-memory conversation must also be
        updated. Otherwise its next save() — on the next /api/ask
        for that session — would clobber the rename with the stale
        title."""
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="mid-chat", title="old")
        conv.add(Role.USER, "hi")
        store.save(conv)

        config = TowelConfig()
        agent = AgentRuntime(config)
        sessions = SessionManager(store=store)
        gw = GatewayServer(config=config, agent=agent, sessions=sessions)
        client = TestClient(gw._build_http_app())

        # Force the session into memory.
        sessions.get_or_create("mid-chat")
        in_mem = sessions.get("mid-chat")
        assert in_mem is not None
        assert in_mem.conversation.title == "old"

        # Rename via API.
        resp = client.post(
            "/conversations/mid-chat/rename",
            json={"title": "renamed-now"},
        )
        assert resp.status_code == 200

        # In-memory conversation reflects the new title.
        assert in_mem.conversation.title == "renamed-now"
        # And a save from this session would persist "renamed-now",
        # not clobber it back to "old".
        sessions.save("mid-chat")
        loaded = store.load("mid-chat")
        assert loaded is not None
        assert loaded.title == "renamed-now"

    def test_rename_rejects_overlong_title(self, tmp_path):
        """Titles surface in the UI sidebar and dispatch logs;
        a 10k-char title destroys layouts and bloats every list
        response. Cap at 200."""
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="long-title")
        conv.add(Role.USER, "test")
        store.save(conv)
        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        client = TestClient(gw._build_http_app())

        resp = client.post(
            "/conversations/long-title/rename",
            json={"title": "z" * 10000},
        )
        assert resp.status_code == 400
        assert "200" in resp.json()["error"]

    def test_rename_rejects_non_dict_body(self, tmp_path):
        """An array/string body crashed `body.get(...)` and surfaced
        as generic "Invalid JSON body" — confusing because the JSON
        is fine, it's the shape that's wrong. Reject precisely."""
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="bad-body")
        conv.add(Role.USER, "test")
        store.save(conv)
        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        client = TestClient(gw._build_http_app())

        for raw in (b"[1,2]", b'"hi"', b"42"):
            resp = client.post(
                "/conversations/bad-body/rename",
                content=raw,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400, f"accepted {raw!r}"
            assert "JSON object" in resp.json()["error"]

    def test_rename_rejects_non_string_title(self, tmp_path):
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="nonstr-title")
        conv.add(Role.USER, "test")
        store.save(conv)
        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        client = TestClient(gw._build_http_app())

        for bad in (42, [1, 2], {"x": 1}, True):
            resp = client.post(
                "/conversations/nonstr-title/rename", json={"title": bad}
            )
            assert resp.status_code == 400, f"accepted {bad!r}"

    def test_rename_rejects_control_chars(self, tmp_path):
        """Multi-line titles break list-view rendering and log readability."""
        from starlette.testclient import TestClient

        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        conv = Conversation(id="multiline-title")
        conv.add(Role.USER, "test")
        store.save(conv)
        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent, sessions=SessionManager(store=store))
        client = TestClient(gw._build_http_app())

        for bad in ("line1\nline2", "tab\there", "null\x00"):
            resp = client.post(
                "/conversations/multiline-title/rename",
                json={"title": bad},
            )
            assert resp.status_code == 400, f"accepted bad title {bad!r}"
            assert "control" in resp.json()["error"].lower()


class TestAutoTitleHelper:
    """Auto-title helper must fire for both WS and HTTP entry points so
    api-channel conversations don't render as blank rows in the
    conversations list. Previously only the WS path titled — every
    /api/ask session shipped with title="" on disk."""

    def _make_gateway(self, tmp_path):
        from towel.agent.runtime import AgentRuntime
        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer
        from towel.gateway.sessions import SessionManager

        store = ConversationStore(store_dir=tmp_path)
        config = TowelConfig()
        agent = AgentRuntime(config)
        return GatewayServer(
            config=config, agent=agent, sessions=SessionManager(store=store),
        )

    def test_helper_sets_title_after_first_exchange(self, tmp_path):
        gw = self._make_gateway(tmp_path)
        session = gw.sessions.get_or_create("auto-title-sess")
        session.conversation.add(Role.USER, "Explain monad transformers please")
        session.conversation.add(Role.ASSISTANT, "Sure...")

        gw._maybe_set_auto_title(session)

        assert session.conversation.title  # non-empty
        # Generated from the user message, not the assistant reply.
        # `generate_title` strips stop-words so the result will pick
        # nouns like "monad" / "transformers".
        lower = session.conversation.title.lower()
        assert "monad" in lower or "transformers" in lower

    def test_helper_no_op_when_title_already_set(self, tmp_path):
        gw = self._make_gateway(tmp_path)
        session = gw.sessions.get_or_create("preserved-title-sess")
        session.conversation.title = "Hand-picked title"
        session.conversation.add(Role.USER, "Something boring")
        session.conversation.add(Role.ASSISTANT, "...")

        gw._maybe_set_auto_title(session)

        assert session.conversation.title == "Hand-picked title"

    def test_helper_no_op_before_full_exchange(self, tmp_path):
        """Before the assistant has replied (only one message), there's
        nothing to title from — must stay empty so the next save
        doesn't lock in a half-baked title."""
        gw = self._make_gateway(tmp_path)
        session = gw.sessions.get_or_create("partial-sess")
        session.conversation.add(Role.USER, "Just one user message")

        gw._maybe_set_auto_title(session)

        assert session.conversation.title == ""

    def test_helper_used_in_simple_ask_flow(self, tmp_path):
        """Integration: /api/ask path constructs a session with a user
        message + an assistant response (even an error one from a
        broken model) and saves. The helper runs just before save
        so the persisted shape carries a title."""
        from starlette.testclient import TestClient

        gw = self._make_gateway(tmp_path)
        client = TestClient(gw._build_http_app())
        # Drive a /api/ask request — the agent has no model loaded, so
        # the call will fail mid-step, but the session gets created
        # with the user message + (depending on path) an error reply.
        # Even if the helper sees only 1 message it must safely no-op
        # and not crash.
        client.post(
            "/api/ask",
            json={"message": "How does deque rotate work in python?",
                  "session_id": "auto-title-integration"},
        )
        session = gw.sessions.get_or_create("auto-title-integration")
        # The user message must be there; assistant response may or
        # may not have been appended depending on which exception path
        # fired. Verify the helper at least didn't crash.
        roles = [m.role for m in session.conversation.messages]
        assert Role.USER in roles

"""Tests for the gateway HTTP endpoints and web UI serving."""

import pytest
from starlette.testclient import TestClient

from towel.agent.conversation import Conversation, Role
from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


@pytest.fixture
def gateway(store):
    config = TowelConfig()
    agent = AgentRuntime(config)
    sessions = SessionManager(store=store)
    pin_store = SessionPinStore(path=store.store_dir / "session_pins.json")
    worker_state_store = WorkerStateStore(path=store.store_dir / "worker_state.json")
    return GatewayServer(
        config=config,
        agent=agent,
        sessions=sessions,
        pin_store=pin_store,
        worker_state_store=worker_state_store,
    )


@pytest.fixture
def client(gateway):
    app = gateway._build_http_app()
    return TestClient(app)


class TestAdminRestart:
    """`/admin/restart` re-execs the coordinator process; without a
    confirmation flag a stray curl or misclicked automation can drop
    all in-memory state (dispatch log, active sessions, in-flight
    worker assignments). The guard mirrors `DELETE /conversations`."""

    def test_restart_without_confirm_rejected(self, client):
        resp = client.post("/admin/restart")
        assert resp.status_code == 400
        assert "confirm=yes" in resp.json()["error"]

    def test_restart_with_wrong_confirm_rejected(self, client):
        resp = client.post("/admin/restart?confirm=please")
        assert resp.status_code == 400

    # NB: we don't test the `?confirm=yes` branch because it triggers
    # `os.execv` on the test process. The guard logic is straightforward
    # enough that a unit test of the negative case + the
    # web UI passing the flag (verified by grep in test_web_ui) is
    # sufficient.


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
        assert data["sessions"][0]["worker_id"] is None


class TestWorkersEndpoint:
    def test_workers_empty(self, client):
        resp = client.get("/workers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["workers"] == []
        # The requirements dict was renamed: scoring now uses preferred_*
        # hints (soft) rather than required_* (hard) so heterogeneous fleets
        # don't get rejected outright.
        assert data["requirements"]["preferred_backend"] == "mlx"
        assert data["requirements"]["preferred_mode"] == "mlx_prompt"

    def test_workers_list_connected_workers(self, gateway, client):
        gateway._workers.register(
            "desktop-1",
            object(),
            {
                "backend": "mlx",
                "model": "repo/model-a",
                "modes": ["mlx_prompt"],
                "tools": False,
            },
        )
        gateway._workers.assign("desktop-1", "job-1", "session-1")

        data = client.get("/workers").json()

        assert len(data["workers"]) == 1
        assert data["workers"][0]["id"] == "desktop-1"
        assert data["workers"][0]["busy"] is True
        assert data["workers"][0]["current_session_id"] == "session-1"
        assert data["workers"][0]["capabilities"]["backend"] == "mlx"
        assert data["workers"][0]["enabled"] is True
        assert data["workers"][0]["draining"] is False
        assert data["pins"] == {}


class TestWorkerStateEndpoint:
    def test_worker_state_update_sets_draining(self, gateway, client):
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )

        resp = client.post("/workers/desktop-1/state", json={"draining": True})

        assert resp.status_code == 200
        assert resp.json()["draining"] is True
        assert gateway._workers.get("desktop-1").draining is True

    def test_worker_state_update_sets_enabled(self, gateway, client):
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )

        resp = client.post("/workers/desktop-1/state", json={"enabled": False})

        assert resp.status_code == 200
        assert resp.json()["enabled"] is False
        assert gateway._workers.get("desktop-1").enabled is False

    def test_worker_state_update_rejects_unknown_worker(self, client):
        resp = client.post("/workers/missing/state", json={"enabled": False})

        assert resp.status_code == 404

    def test_worker_state_rejects_non_dict_body(self, gateway, client):
        """An array / string / number top-level body crashed on
        `body.get(...)` and surfaced as plaintext HTTP 500."""
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )
        for raw in (b"[1,2]", b'"hi"', b"42"):
            resp = client.post(
                "/workers/desktop-1/state",
                content=raw,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400, f"accepted {raw!r}"
            assert "JSON object" in resp.json()["error"]

    def test_worker_state_rejects_non_bool_values(self, gateway, client):
        """Previously the handler did `bool(value)` which made any
        non-empty string truthy: `{"draining": "yes"}` drained the
        worker, `{"draining": "false"}` *also* drained it (the string
        "false" is truthy in Python). This is an operator-facing
        endpoint — wrong inputs must fail loud."""
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )
        for bad in ("yes", "false", "1", 0, [], {"x": 1}):
            resp = client.post(
                "/workers/desktop-1/state", json={"draining": bad}
            )
            assert resp.status_code == 400, f"accepted bad draining={bad!r}"
            assert "true or false" in resp.json()["error"]
            # And the worker state must remain UNCHANGED.
            assert gateway._workers.get("desktop-1").draining is False

        for bad in ("yes", "false", 1, []):
            resp = client.post(
                "/workers/desktop-1/state", json={"enabled": bad}
            )
            assert resp.status_code == 400, f"accepted bad enabled={bad!r}"


class TestWorkerPinEndpoint:
    def test_pin_worker_sets_session_pin(self, gateway, client):
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )

        resp = client.post("/sessions/chat-1/pin-worker", json={"worker_id": "desktop-1"})

        assert resp.status_code == 200
        assert resp.json()["pinned"] is True
        assert gateway._session_pins["chat-1"] == "desktop-1"

    def test_pin_worker_rejects_unknown_worker(self, client):
        resp = client.post("/sessions/chat-1/pin-worker", json={"worker_id": "missing"})

        assert resp.status_code == 404

    def test_unpin_worker_clears_session_pin(self, gateway, client):
        gateway._session_pins["chat-1"] = "desktop-1"

        resp = client.request("DELETE", "/sessions/chat-1/pin-worker")

        assert resp.status_code == 200
        assert resp.json()["pinned"] is False
        assert "chat-1" not in gateway._session_pins


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
        assert "tb-fleet" in resp.text
        assert "tb-export" in resp.text
        assert "tb-delete" in resp.text

    def test_index_has_fleet_panel(self, client):
        resp = client.get("/")
        assert "fleet-overlay" in resp.text
        assert "fleet-workers-list" in resp.text
        assert "fleet-routes-list" in resp.text
        assert "Fleet Control" in resp.text

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

    def test_delete_clears_session_affinity(self, gateway, store, client):
        """Singular conversation_delete previously left a stale
        _session_workers entry behind. A delete-then-recreate of the
        same session_id (common in scripted /api/ask usage) would
        show the OLD worker_id in /sessions for the new session
        until the new session got dispatched."""
        conv = Conversation(id="affinity-leak")
        conv.add(Role.USER, "test")
        store.save(conv)
        gateway._session_workers["affinity-leak"] = "ghost-worker"

        resp = client.request("DELETE", "/conversations/affinity-leak")
        assert resp.status_code == 200
        # And the affinity dict must NOT carry the deleted session's entry.
        assert "affinity-leak" not in gateway._session_workers

    def test_delete_clears_session_worker_pin(self, gateway, store, client):
        """A pin set on a soon-deleted conversation would otherwise
        persist into the SessionPinStore on next save — a phantom pin
        that the dispatcher honors for the deleted session_id if it
        ever reappears."""
        conv = Conversation(id="pin-leak")
        conv.add(Role.USER, "test")
        store.save(conv)
        gateway._session_pins["pin-leak"] = "some-worker"
        gateway.pin_store.save(gateway._session_pins)

        resp = client.request("DELETE", "/conversations/pin-leak")
        assert resp.status_code == 200
        assert "pin-leak" not in gateway._session_pins
        # And the persisted pin store must have been re-saved without it.
        assert "pin-leak" not in gateway.pin_store.load()

    def test_delete_all_requires_confirmation(self, store, client):
        """DELETE /conversations is a "wipe everything" footgun. Without
        ?confirm=yes a stale curl in shell history or a misclicked UI
        button would silently destroy the entire archive. Require the
        explicit confirmation."""
        # Seed with a few entries so we can verify they SURVIVE the bad
        # call (an accidental wipe would leave us with zero).
        for i in range(3):
            conv = Conversation(id=f"survives-{i}")
            conv.add(Role.USER, "keep me")
            store.save(conv)

        resp = client.request("DELETE", "/conversations")
        assert resp.status_code == 400
        body = resp.json()
        assert "confirm=yes" in body["error"]
        assert body["would_delete"] == 3
        # Crucially: the conversations are still on disk.
        assert store.count == 3

    def test_delete_all_with_confirmation_wipes(self, store, client):
        for i in range(3):
            conv = Conversation(id=f"wipe-{i}")
            conv.add(Role.USER, "ok bye")
            store.save(conv)

        resp = client.request("DELETE", "/conversations?confirm=yes")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 3
        assert store.count == 0

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

    def test_export_unknown_format_rejected(self, store, client):
        """A client passing `format=evil` previously got markdown back
        with no indication of the typo. Better to fail loud — 400
        with the list of valid formats."""
        conv = Conversation(id="exp-fmt", channel="cli")
        conv.add(Role.USER, "hi")
        store.save(conv)

        resp = client.get("/conversations/exp-fmt/export?format=evil")
        assert resp.status_code == 400
        assert "markdown" in resp.json()["error"]
        assert "json" in resp.json()["error"]
        assert "text" in resp.json()["error"]


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

    def test_ask_non_object_body_rejected(self, client):
        """Array / string / null bodies crashed `body.get(...)` with
        an AttributeError that Starlette rendered as plaintext
        "Internal Server Error" HTTP 500 — not JSON, hard for API
        clients to handle uniformly. Reject at the boundary.

        Note `json=None` to the test client sends no body at all (so
        the JSON parse fails with "Invalid JSON" first); explicit
        non-objects use raw content."""
        # JSON array / string / number / boolean as the entire body.
        for raw in (b"[1,2,3]", b'"just a string"', b"42", b"true", b"null"):
            resp = client.post(
                "/api/ask",
                content=raw,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400, f"accepted {raw!r}"
            assert "JSON object" in resp.json()["error"]

    def test_ask_non_string_message_rejected(self, client):
        """A message field that isn't a string would crash on
        `.strip()` after `body.get("message", "")` succeeded."""
        for bad in (42, [1, 2], {"nested": "x"}, True, None):
            resp = client.post("/api/ask", json={"message": bad})
            assert resp.status_code == 400, f"accepted {bad!r}"

    def test_ask_creates_session(self, gateway, client):
        # The actual model call will fail (no model loaded), but we test the session creation
        _resp = client.post("/api/ask", json={"message": "hello", "session": "test-ask"})
        # Will be 500 (model not loaded) but session should exist
        session = gateway.sessions.get_or_create("test-ask")
        assert len(session.conversation) >= 1  # at least the user message

    def test_ask_accepts_session_id_key(self, gateway, client):
        """Clients reasonably pass ``session_id`` (the convention used
        everywhere else in towel — path params, internal APIs, the
        session list). Previously only ``session`` was honored, so
        ``session_id`` was silently dropped and every such request
        was merged into ``api-default``, sharing context with everyone."""
        _resp = client.post(
            "/api/ask",
            json={"message": "hello", "session_id": "test-ask-via-id-key"},
        )
        session = gateway.sessions.get_or_create("test-ask-via-id-key")
        assert len(session.conversation) >= 1
        # And api-default must NOT have received this message — the old
        # bug would route it there and contaminate the shared session.
        api_default = gateway.sessions.get_or_create("api-default")
        contents = [m.content for m in api_default.conversation.messages]
        assert "hello" not in contents

    def test_ask_rejects_overlong_session_id(self, client):
        """Session IDs flow into dispatch logs, filesystem paths, and
        URL params. A 1000-char session_id breaks every list view and
        log line — same length rule as memory keys (commit 1865e7d)."""
        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "z" * 1000},
        )
        assert resp.status_code == 400
        assert "256" in resp.json()["error"]

    def test_ask_rejects_control_chars_in_session_id(self, client):
        """Newlines in session_id break log readability and would
        appear in dispatch decision dumps as multi-line entries."""
        for bad in ("a\nb", "tab\there", "null\x00here"):
            resp = client.post(
                "/api/ask",
                json={"message": "hi", "session_id": bad},
            )
            assert resp.status_code == 400, f"accepted bad session_id {bad!r}"
            assert "control" in resp.json()["error"].lower()

    def test_ask_strips_session_id_whitespace(self, gateway, client):
        """`"  sid  "` and `"sid"` previously created two different
        in-memory sessions even though the on-disk filename sanitizer
        merged them to the same .json file. Loads from one key,
        saves to another — confusing for operators watching
        /api/sessions."""
        _resp = client.post(
            "/api/ask",
            json={"message": "hello", "session_id": "  spaced  "},
        )
        # In-memory session must be keyed by the stripped form.
        assert "spaced" in gateway.sessions._sessions
        assert "  spaced  " not in gateway.sessions._sessions

    def test_ask_all_whitespace_session_id_falls_back_to_default(self, gateway, client):
        _resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "   "},
        )
        # Whitespace-only session_id is ambiguous; we treat it as
        # "no session_id given" and route to api-default.
        assert "api-default" in gateway.sessions._sessions

    def test_ask_rejects_non_string_session_id(self, client):
        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": 12345},
        )
        assert resp.status_code == 400
        assert "string" in resp.json()["error"]

    def test_retry_failure_restores_original_placeholder(self, gateway, client):
        """When the empty-response retry path crashes (worker DC,
        timeout, anything), the session must still have a coherent
        assistant turn. The earlier implementation popped the original
        diagnostic placeholder before the retry call and never put it
        back on failure — so a crashed retry left the session with
        the user message and NO assistant reply, while the API caller
        got a 500."""
        import asyncio
        from unittest.mock import AsyncMock
        from towel.agent.conversation import Message, Role
        from towel.gateway.workers import WorkerInfo

        # Stub _route_by_role so the request flows into the chat path
        # without needing a real dispatcher decision.
        fake_worker = WorkerInfo(id="primary", ws=AsyncMock(), capabilities={})
        gateway._workers._workers["primary"] = fake_worker
        gateway._workers._workers["alt"] = WorkerInfo(
            id="alt", ws=AsyncMock(),
            capabilities={"total_vram_mb": 16000},
        )

        async def fake_route(message, session_id):
            return fake_worker, "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        # First call: original placeholder (empty_text_fallback=True).
        # Second call: raises to simulate a crashed retry.
        call_log = []

        async def fake_quick(session_id, session, worker, max_tokens=256):
            call_log.append(worker.id)
            if worker.id == "primary":
                placeholder = Message(
                    role=Role.ASSISTANT,
                    content="(The worker returned no text...)",
                    metadata={
                        "remote_worker": "primary",
                        "empty_text_fallback": True,
                    },
                )
                session.conversation.messages.append(placeholder)
                return placeholder
            else:
                # The retry path on the alt worker explodes.
                raise RuntimeError("simulated worker crash")

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "retry-restore"},
        )

        assert resp.status_code == 200
        # Both workers were attempted.
        assert call_log == ["primary", "alt"]
        # And crucially the session has a coherent assistant message
        # (the restored placeholder), not just the user turn.
        sess = gateway.sessions.get_or_create("retry-restore")
        roles = [m.role for m in sess.conversation.messages]
        assert roles[-1] == Role.ASSISTANT, (
            f"expected assistant placeholder restored, got roles={roles}"
        )
        # And the visible content is the original placeholder, not empty.
        assert sess.conversation.messages[-1].content.startswith(
            "(The worker returned no text"
        )


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

    def test_api_sessions_limit_param(self, store, client):
        for i in range(5):
            conv = Conversation(id=f"limit-{i}", channel="api")
            conv.add(Role.USER, f"msg {i}")
            store.save(conv)

        resp = client.get("/api/sessions?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["sessions"]) == 2

    def test_api_sessions_invalid_limit_rejected(self, client):
        resp = client.get("/api/sessions?limit=notanumber")
        assert resp.status_code == 400
        assert "limit" in resp.json()["error"]

    def test_api_sessions_limit_clamped(self, store, client):
        for i in range(3):
            conv = Conversation(id=f"clamp-{i}", channel="api")
            conv.add(Role.USER, f"msg {i}")
            store.save(conv)

        # limit=99999 must not crash or read 99999 files; the cap is
        # 500, but the result here is just "all three saved convos".
        resp = client.get("/api/sessions?limit=99999")
        assert resp.status_code == 200
        assert len(resp.json()["sessions"]) == 3


class TestAlternateChatWorker:
    """When the routed worker returns empty text on /api/ask, the
    coordinator picks the next-best worker to retry on. Picking
    must prefer the LARGEST idle worker (higher chance of producing
    real text), and must exclude busy / draining / disabled workers
    and the one we already tried."""

    def test_picks_largest_idle_non_excluded(self, gateway):
        gateway._workers.register(
            "small", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 4096},
        )
        gateway._workers.register(
            "big", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 16000},
        )
        gateway._workers.register(
            "medium", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 8192},
        )
        # Exclude the one we just tried — pick must NOT return it,
        # but should prefer "big" over "medium".
        alt = gateway._pick_alternate_chat_worker(exclude={"small"})
        assert alt is not None
        assert alt.id == "big"

    def test_returns_none_when_no_alternates(self, gateway):
        gateway._workers.register(
            "only", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 4096},
        )
        alt = gateway._pick_alternate_chat_worker(exclude={"only"})
        assert alt is None

    def test_prefers_idle_over_busy(self, gateway):
        """When both an idle and a busy alternate exist, pick idle —
        even if the busy one is bigger. The busy worker has a real
        job blocking the WebSocket queue."""
        gateway._workers.register(
            "small-idle", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 4096},
        )
        gateway._workers.register(
            "big-busy", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 16000},
        )
        gateway._workers.assign("big-busy", "job-x", "session-x")
        gateway._workers.register(
            "excluded", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 8000},
        )
        alt = gateway._pick_alternate_chat_worker(exclude={"excluded"})
        assert alt is not None
        assert alt.id == "small-idle"

    def test_falls_back_to_busy_when_no_idle(self, gateway):
        """When the only alternate is busy, pick it anyway — the
        WebSocket queue will serialize the request. Without this
        the retry-on-empty path silently turned into 'keep the
        diagnostic placeholder' whenever the only good worker was
        already handling another query."""
        gateway._workers.register(
            "small", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 4096},
        )
        gateway._workers.register(
            "big-busy", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 16000},
        )
        gateway._workers.assign("big-busy", "job-x", "session-x")
        alt = gateway._pick_alternate_chat_worker(exclude={"small"})
        assert alt is not None
        assert alt.id == "big-busy"

    def test_skips_draining_workers(self, gateway):
        gateway._workers.register(
            "small", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 4096},
        )
        gateway._workers.register(
            "draining", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 16000},
        )
        gateway._workers.set_draining("draining", True)
        alt = gateway._pick_alternate_chat_worker(exclude={"small"})
        assert alt is None

    def test_skips_stuck_busy_workers(self, gateway):
        """A worker that's been busy for 5+ minutes is wedged on a
        previous request. Queuing the retry behind it would inherit
        the wedge — turning a "slow but eventually correct" retry
        into a hung HTTP call. Prefer no retry over a stuck one."""
        from datetime import UTC, datetime, timedelta

        gateway._workers.register(
            "small", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 4096},
        )
        gateway._workers.register(
            "stuck-big", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 16000},
        )
        gateway._workers.assign("stuck-big", "stuck-job", "stuck-session")
        # Force busy_since to be 10 minutes ago.
        worker = gateway._workers.get("stuck-big")
        worker.busy_since = datetime.now(UTC) - timedelta(minutes=10)

        alt = gateway._pick_alternate_chat_worker(exclude={"small"})
        assert alt is None

    def test_keeps_recently_busy_worker_as_fallback(self, gateway):
        """A worker that just started a job (busy_since < 5 minutes)
        is normal — its queue will serve the retry shortly. Don't
        confuse it with stuck."""
        from datetime import UTC, datetime, timedelta

        gateway._workers.register(
            "small", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 4096},
        )
        gateway._workers.register(
            "busy-big", object(),
            {"backend": "llama", "modes": ["llama_chat"], "total_vram_mb": 16000},
        )
        gateway._workers.assign("busy-big", "job", "session")
        worker = gateway._workers.get("busy-big")
        worker.busy_since = datetime.now(UTC) - timedelta(seconds=30)

        alt = gateway._pick_alternate_chat_worker(exclude={"small"})
        assert alt is not None
        assert alt.id == "busy-big"

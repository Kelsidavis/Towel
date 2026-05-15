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


class TestClusterNodes:
    """`/cluster/nodes` must surface operator-set flags
    (enabled / draining / busy) on each node, so operators
    debugging "why isn't this worker being used" don't need to
    cross-reference /workers."""

    def test_cluster_nodes_includes_operator_state(self, gateway, client):
        gateway._workers.register(
            "node-a", object(),
            {
                "backend": "llama",
                "modes": ["llama_chat"],
                "total_vram_mb": 16000,
                "context_window": 8192,
                "max_tokens": 4096,
                "hostname": "node-a",
                "resources": {"hostname": "node-a", "ram_total_mb": 32000},
            },
        )
        # Also register the node in the tracker (the WS register
        # handler does this in prod; we exercise it here directly).
        gateway._node_tracker.register("node-a", gateway._workers.get("node-a").capabilities)
        gateway._workers.set_draining("node-a", True)

        resp = client.get("/cluster/nodes")
        assert resp.status_code == 200
        nodes = resp.json()["nodes"]
        assert "node-a" in nodes
        entry = nodes["node-a"]
        assert entry["enabled"] is True
        assert entry["draining"] is True
        assert entry["busy"] is False
        # Job/timing fields surface so operators debugging from this
        # endpoint don't have to bounce to /workers.
        assert entry["current_job_id"] is None
        assert entry["current_session_id"] is None
        assert entry["busy_since"] is None
        assert entry["busy_for_seconds"] is None

    def test_cluster_nodes_surfaces_busy_timing(self, gateway, client):
        """When a worker is actively assigned to a job, /cluster/nodes
        must echo busy_since + busy_for_seconds + current_job_id +
        current_session_id so an operator can spot a stuck job from
        the cluster panel without bouncing to /workers."""
        gateway._workers.register(
            "node-a", object(),
            {
                "backend": "llama",
                "modes": ["llama_chat"],
                "total_vram_mb": 16000,
                "context_window": 8192,
                "max_tokens": 4096,
                "hostname": "node-a",
                "resources": {"hostname": "node-a", "ram_total_mb": 32000},
            },
        )
        gateway._node_tracker.register("node-a", gateway._workers.get("node-a").capabilities)
        gateway._workers.assign("node-a", job_id="job-42", session_id="sess-7")

        resp = client.get("/cluster/nodes")
        assert resp.status_code == 200
        entry = resp.json()["nodes"]["node-a"]
        assert entry["busy"] is True
        assert entry["current_job_id"] == "job-42"
        assert entry["current_session_id"] == "sess-7"
        assert entry["busy_since"] is not None
        # busy_for_seconds is a float (newly-assigned, so near-zero
        # but never negative).
        assert isinstance(entry["busy_for_seconds"], (int, float))
        assert entry["busy_for_seconds"] >= 0

    def test_cluster_nodes_marks_unknown_workers(self, gateway, client):
        """If the node tracker has a stale entry the registry doesn't,
        the operator-state fields surface as null so the UI can flag
        the discrepancy explicitly rather than guessing."""
        # Register the node only in the tracker, not the registry.
        gateway._node_tracker.register(
            "ghost-node",
            {"backend": "llama", "context_window": 8192, "max_tokens": 4096},
        )
        resp = client.get("/cluster/nodes")
        entry = resp.json()["nodes"]["ghost-node"]
        assert entry["enabled"] is None
        assert entry["draining"] is None
        assert entry["busy"] is None
        # Job/timing fields are also None for stale entries — the UI
        # contract is "all operator-state fields are null when the
        # registry has no record."
        assert entry["current_job_id"] is None
        assert entry["current_session_id"] is None
        assert entry["busy_since"] is None
        assert entry["busy_for_seconds"] is None


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

    def test_worker_state_update_response_shape_is_worker_not_memory(
        self, gateway, client,
    ):
        """The response body used to flow through `_memory_entry_dict`,
        which set defaults for memory-entry fields (tags / source /
        scope / last_recalled_at). Operators saw `"tags": []` on a
        worker and wondered if memory entries were attached. Verify
        the response now uses WorkerInfo.to_dict shape only."""
        gateway._workers.register(
            "desktop-1", object(),
            {"backend": "mlx", "modes": ["mlx_prompt"]},
        )

        resp = client.post("/workers/desktop-1/state", json={"enabled": True})
        assert resp.status_code == 200
        data = resp.json()
        # Worker-shape fields must be present.
        assert "id" in data
        assert "capabilities" in data
        assert "enabled" in data
        assert "draining" in data
        # Memory-shape fields must NOT be there.
        for field in ("tags", "source", "scope", "last_recalled_at"):
            assert field not in data, f"unexpected memory field {field!r}"

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

    def test_pin_worker_rejects_non_dict_body(self, client):
        """A top-level array / string / null body previously crashed
        on body.get(...) and surfaced as a misleading "Invalid JSON
        body" (the JSON parsed fine — it was the shape that was
        wrong). Same boundary check applied to every other POST."""
        for raw in (b"null", b"[1,2,3]", b'"hi"'):
            resp = client.post(
                "/sessions/chat-1/pin-worker",
                content=raw,
                headers={"content-type": "application/json"},
            )
            assert resp.status_code == 400, f"accepted {raw!r}"
            assert "JSON object" in resp.json()["error"]

    def test_pin_worker_rejects_non_string_worker_id(self, client):
        """worker_id=42 previously hit .strip() with AttributeError
        and got the misleading "Invalid JSON body" message. Now a
        clear "must be a string" 400."""
        for bad in (42, [1, 2], {"x": 1}, True):
            resp = client.post(
                "/sessions/chat-1/pin-worker", json={"worker_id": bad},
            )
            assert resp.status_code == 400, f"accepted {bad!r}"
            assert "string" in resp.json()["error"].lower()

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
        # `tags` field present (empty list by default) matches the
        # shape of /api/sessions so API clients can use either.
        assert data["conversations"][0]["tags"] == []

    def test_list_carries_tags(self, store, client):
        """A conversation with tags exposes them on /conversations
        the same way /api/sessions does — same data, two paths."""
        conv = Conversation(id="tagged-x", channel="api")
        conv.tags = ["work", "urgent"]
        conv.add(Role.USER, "hi")
        store.save(conv)

        data = client.get("/conversations").json()
        assert len(data["conversations"]) == 1
        assert data["conversations"][0]["tags"] == ["work", "urgent"]

    def test_get_conversation_includes_title_and_tags(self, store, client):
        """Conversation.to_dict omits title/tags when empty, but the
        detail endpoint always emits them so API clients don't have
        to special-case detail vs list shapes."""
        conv = Conversation(id="shape-1", channel="cli")
        conv.add(Role.USER, "hi")
        # No title, no tags — defaults.
        store.save(conv)

        resp = client.get("/conversations/shape-1")
        assert resp.status_code == 200
        body = resp.json()
        # Both fields present even though the underlying conversation
        # had neither set.
        assert "title" in body
        assert body["title"] == ""
        assert "tags" in body
        assert body["tags"] == []

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

    def test_delete_closes_node_tracker_context_slot(
        self, gateway, store, client,
    ):
        """Deleting a conversation must also drop its slot from the
        NodeTracker. Without this, ghost slots accumulate as sessions
        are deleted, inflating `active_sessions` and
        `context_pressure` on /cluster/nodes — and the dispatcher's
        context-aware routing then avoids workers that are
        genuinely idle. Caught on the live coordinator where one
        worker showed pressure=1.0 from leftover probe slots."""
        # Set up: worker registered with a node, session pinned via
        # affinity, with an open context slot.
        gateway._workers.register(
            "node-a", object(),
            {
                "backend": "llama", "modes": ["llama_chat"],
                "context_window": 8192, "max_tokens": 4096,
                "total_vram_mb": 16000,
                "resources": {"hostname": "node-a", "ram_total_mb": 32000},
            },
        )
        gateway._node_tracker.register(
            "node-a", gateway._workers.get("node-a").capabilities,
        )
        conv = Conversation(id="slot-leak")
        conv.add(Role.USER, "test")
        store.save(conv)
        gateway._session_workers["slot-leak"] = "node-a"
        gateway._node_tracker.open_context_slot("node-a", "slot-leak", 100)

        # Sanity: slot is present pre-delete.
        node = gateway._node_tracker.get("node-a")
        assert node is not None
        assert node.get_context_slot("slot-leak") is not None

        resp = client.request("DELETE", "/conversations/slot-leak")
        assert resp.status_code == 200

        # Slot must be gone — no ghost slot left on the worker.
        node = gateway._node_tracker.get("node-a")
        assert node is not None
        assert node.get_context_slot("slot-leak") is None

    def test_delete_all_closes_all_context_slots(
        self, gateway, store, client,
    ):
        """Same fix on the bulk-delete path — clearing the conversation
        archive must also clear ghost slots, otherwise
        context_pressure stays inflated until workers disconnect."""
        gateway._workers.register(
            "node-x", object(),
            {
                "backend": "llama", "modes": ["llama_chat"],
                "context_window": 8192, "max_tokens": 4096,
                "total_vram_mb": 16000,
                "resources": {"hostname": "node-x", "ram_total_mb": 32000},
            },
        )
        gateway._node_tracker.register(
            "node-x", gateway._workers.get("node-x").capabilities,
        )
        for i in range(3):
            conv = Conversation(id=f"bulk-slot-{i}")
            conv.add(Role.USER, "test")
            store.save(conv)
            gateway._session_workers[f"bulk-slot-{i}"] = "node-x"
            gateway._node_tracker.open_context_slot(
                "node-x", f"bulk-slot-{i}", 100,
            )

        node = gateway._node_tracker.get("node-x")
        assert len(node.context_slots) == 3

        resp = client.request("DELETE", "/conversations?confirm=yes")
        assert resp.status_code == 200

        node = gateway._node_tracker.get("node-x")
        assert len(node.context_slots) == 0

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

    def test_export_filename_strips_quotes_and_other_unsafe_chars(
        self, store, client,
    ):
        """The /api/ask session_id validator only blocks control chars
        and ≥257-char ids — quotes, semicolons, and backslashes slip
        through. If the raw id is interpolated into the
        Content-Disposition filename="..." header, the inner quote
        truncates the filename and breaks the extension; worse, a
        sufficiently crafted id could inject extra header parameters.
        Sanitize the filename to alphanumerics + -_ at the boundary.
        """
        # Create a conversation whose id contains a quote — the store
        # path sanitizer strips it for disk, but the in-memory
        # `conv.id` keeps it.
        conv = Conversation(id='nasty"id;evil', channel="cli")
        conv.add(Role.USER, "hi")
        store.save(conv)

        resp = client.get(
            "/conversations/nasty%22id%3Bevil/export?format=json"
        )
        # The store sanitizer mapped both raw and URL-decoded
        # variants to the same disk path, so the load succeeds.
        assert resp.status_code == 200

        # Content-Disposition must not contain raw quotes or
        # semicolons inside the filename — the wrapping quotes are
        # the only ones allowed.
        cd = resp.headers["content-disposition"]
        # Header shape: attachment; filename="towel-<alnum>.json"
        assert cd.startswith("attachment; filename=\"")
        # Extract everything between the first and last quote, that's
        # the filename payload.
        first = cd.index('"')
        last = cd.rindex('"')
        payload = cd[first + 1 : last]
        # Quotes / semicolons / backslashes in the payload would
        # break the header.
        for forbidden in ('"', ";", "\\"):
            assert forbidden not in payload, (
                f"unsafe char {forbidden!r} in filename payload {payload!r}"
            )


class TestSearch:
    """`/search` walks the conversation archive matching `?q=` against
    every message. The gateway must keep nonsense queries out of the
    expensive regex scan."""

    def test_search_missing_q(self, client):
        resp = client.get("/search")
        assert resp.status_code == 400
        assert "?q=" in resp.json()["error"]

    def test_search_whitespace_only_q_rejected(self, store, client):
        """`?q=  ` previously slipped through to ConversationStore.search
        which compiled `re.escape("  ")` and matched essentially every
        message containing two adjacent spaces. Whitespace-only is
        functionally a "missing query"."""
        # Seed a conversation so a permissive regex would match it.
        conv = Conversation(id="match-all")
        conv.add(Role.USER, "hello  world")  # two spaces inside the text
        store.save(conv)

        # The test client's URL parser doesn't accept raw whitespace,
        # so encode each test case manually. The handler still sees
        # the decoded string after Starlette parses the query string.
        for encoded in ("%20%20", "%09", "%20%0a%20"):
            resp = client.get(f"/search?q={encoded}")
            assert resp.status_code == 400, f"accepted q={encoded!r}"
            assert "?q=" in resp.json()["error"]

    def test_search_bad_limit(self, client):
        resp = client.get("/search?q=hello&limit=junk")
        assert resp.status_code == 400
        assert "limit" in resp.json()["error"]

    def test_search_rejects_overlong_q(self, client):
        """A 2000-char `q` wastes CPU on a full-archive FTS scan,
        bloats the echoed response, and bloats every access-log
        line. Match the 256-char rule applied to session_id and
        memory keys."""
        resp = client.get("/search?q=" + "a" * 1000)
        assert resp.status_code == 400
        assert "256" in resp.json()["error"]

    def test_search_rejects_control_chars(self, client):
        """Null bytes / embedded newlines in `q` break log
        readability and surface in the echoed `query` JSON field."""
        # URL-encode control characters so the test client accepts
        # them; the handler validates after the URL parser decodes.
        for encoded in ("%00", "hello%0aworld", "tab%09here"):
            resp = client.get(f"/search?q={encoded}")
            assert resp.status_code == 400, f"accepted q={encoded!r}"
            assert "control" in resp.json()["error"].lower()


class TestDispatchExplain:
    """`/dispatch/explain` is a previewing endpoint — it shouldn't
    silently fall through bogus inputs (typo'd intent, negative
    tokens) because the whole point is to surface what the
    dispatcher would do."""

    def test_missing_session_id(self, client):
        resp = client.get("/dispatch/explain")
        assert resp.status_code == 400
        assert "session_id" in resp.json()["error"]

    def test_intent_must_be_known(self, client):
        """A typo like ?intent=tools previously fell into the task
        branch silently. The preview must surface the typo instead."""
        resp = client.get("/dispatch/explain?session_id=s&intent=tools")
        assert resp.status_code == 400
        assert "intent" in resp.json()["error"].lower()

    def test_estimated_tokens_must_be_non_negative(self, client):
        resp = client.get(
            "/dispatch/explain?session_id=s&estimated_tokens=-1",
        )
        assert resp.status_code == 400
        assert "≥" in resp.json()["error"] or "non-negative" in resp.json()["error"].lower() or "0" in resp.json()["error"]

    def test_estimated_tokens_must_be_sane_size(self, client):
        """A 17-digit token estimate would trip the dispatcher's
        context-window comparisons and mark quality_degraded for
        every fleet — preview should reject the absurd input."""
        resp = client.get(
            "/dispatch/explain?session_id=s&estimated_tokens=99999999999999",
        )
        assert resp.status_code == 400
        assert "10000000" in resp.json()["error"] or "≤" in resp.json()["error"]

    def test_estimated_tokens_not_an_int(self, client):
        resp = client.get("/dispatch/explain?session_id=s&estimated_tokens=abc")
        assert resp.status_code == 400
        assert "integer" in resp.json()["error"].lower()

    def test_task_type_overlong_rejected(self, client):
        """Same echo-bloat protection as /workers/{id}/tasks — a
        2000-char bogus value must not be inlined in the error."""
        resp = client.get(
            "/dispatch/explain?session_id=s&task_type=" + "a" * 200,
        )
        assert resp.status_code == 400
        assert "64" in resp.json()["error"]
        # Bad value must NOT be echoed back.
        assert "a" * 100 not in resp.json()["error"]

    def test_task_type_unknown_value(self, client):
        resp = client.get(
            "/dispatch/explain?session_id=s&task_type=not-a-task",
        )
        assert resp.status_code == 400
        assert "task_type" in resp.json()["error"]


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

    def test_ask_persists_user_message_on_inference_failure(
        self, gateway, store, client,
    ):
        """A brand-new /api/ask session that errors before any reply
        used to disappear: the user turn was added to the in-memory
        Conversation but the save() at the end of the try block was
        skipped when inference raised. The user opened
        /conversations later and saw no record of having asked the
        question. Now the save runs in a finally so the partial
        session is persisted."""
        # No model loaded → the inference call will raise and the
        # endpoint returns 500. The user message must STILL be on
        # disk afterwards.
        resp = client.post(
            "/api/ask",
            json={"message": "this should survive", "session_id": "errored-session"},
        )
        # Inference failed (no model / no workers), so we expect 500.
        # The exact status code matters less than what's on disk.
        assert resp.status_code in (200, 500)

        loaded = store.load("errored-session")
        assert loaded is not None, (
            "session was lost — the partial-save guard is missing"
        )
        # The user message must be in the persisted record.
        contents = [m.content for m in loaded.messages]
        assert "this should survive" in contents

    def test_ask_handles_none_tps_in_metadata(self, gateway, client):
        """A worker job_error / empty-text fallback can produce a
        response whose metadata has `tps: None` (no measurement was
        taken). `round(None, 1)` raises TypeError, which used to 500
        the whole /api/ask after an otherwise-recoverable empty-text
        path. Coerce non-numeric tps/tokens to 0 at the response
        boundary."""
        from towel.agent.conversation import Message, Role

        # Stub the local-agent path so we return a response whose
        # metadata carries the problematic shapes.
        async def fake_step(conv):
            return Message(
                role=Role.ASSISTANT,
                content="hello there",
                metadata={"tps": None, "tokens": None},
            )

        gateway.agent.step = fake_step  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "tps-none"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # tps coerced to a real number (0.0).
        assert body["tps"] == 0
        # tokens coerced to 0 and then back-estimated from content.
        assert isinstance(body["tokens"], int)
        assert body["tokens"] > 0

    def test_ask_ensemble_must_be_bool(self, client):
        for bad in ("yes", 1, "true", "false", [True]):
            resp = client.post(
                "/api/ask",
                json={"message": "hi", "session_id": "e-bad", "ensemble": bad},
            )
            assert resp.status_code == 400, f"accepted ensemble={bad!r}"

    def test_ask_ensemble_and_verify_mutually_exclusive(self, client):
        """Two different collaboration models — verify is sequential
        (draft → review), ensemble is parallel fan-out. Operators
        pick one; combining them is almost certainly a mistake."""
        resp = client.post(
            "/api/ask",
            json={
                "message": "hi", "session_id": "ev-conflict",
                "verify": True, "ensemble": True,
            },
        )
        assert resp.status_code == 400
        assert "mutually exclusive" in resp.json()["error"]

    def test_ask_ensemble_fans_out_and_picks_longest(self, gateway, client):
        """End-to-end ensemble: every idle worker gets the same prompt
        concurrently, coordinator picks the longest non-empty answer.
        Real multi-worker collaboration on a single request — every
        worker contributes input, coordinator arbitrates."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "small", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "large", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        # Each worker returns a different-length answer so we can
        # check arbitration picks the longest.
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            answers = {
                "small": "Short answer.",
                "large": "A more thorough, multi-sentence response that explains the topic in depth.",
            }
            return Message(
                role=Role.ASSISTANT,
                content=answers.get(worker.id, ""),
                metadata={"remote_worker": worker.id, "tokens": 10, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={
                "message": "deep reasoning question",
                "session_id": "ens-1",
                "ensemble": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # The longer answer (large worker) wins arbitration.
        assert "thorough" in body["response"]
        # Response metadata exposes the parallel run.
        assert body["worker"] == "ensemble"
        assert body["ensemble"] is True
        # Both workers contributed.
        ids = {c["worker_id"] for c in body["ensemble_contributions"]}
        assert ids == {"small", "large"}

    def test_ask_ensemble_synthesizes_via_local_agent(self, gateway, client):
        """When 2+ workers contribute, the coordinator's local agent
        synthesizes a final answer rather than picking a longest
        answer. This is the 'coordinator pieces it together' half of
        the collaboration model — workers each contribute an
        independent attempt, coordinator reconciles."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "a", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "b", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            answers = {
                "a": "Answer from A: Paris.",
                "b": "Answer from B: Paris, the capital of France.",
            }
            return Message(
                role=Role.ASSISTANT,
                content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 6, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # Stub the local agent's step so synthesis returns a known
        # reconciled answer — proves the synthesis path was reached
        # and used, rather than the longest-answer fallback.
        async def fake_step(conv):
            # The synthesis prompt should mention both workers' answers.
            user_msg = conv.messages[-1].content
            assert "Worker A answered:" in user_msg
            assert "Worker B answered:" in user_msg
            assert "Paris" in user_msg
            return Message(
                role=Role.ASSISTANT,
                content="Synthesized: The capital of France is Paris.",
                metadata={"tps": 10.0, "tokens": 8},
            )

        gateway.agent.step = fake_step  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={
                "message": "What is the capital of France?",
                "session_id": "ens-synth",
                "ensemble": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # The synthesized answer wins, not either individual worker's.
        assert "Synthesized" in body["response"]
        # Both workers' contributions are still reported.
        ids = {c["worker_id"] for c in body["ensemble_contributions"]}
        assert ids == {"a", "b"}

    def test_ask_ensemble_passes_session_history_to_each_worker(
        self, gateway, client,
    ):
        """Each fan-out worker must see the full conversation
        history, not just the latest turn — a follow-up like 'but
        make it shorter' loses its referent without context. The
        ephemeral session per worker clones the user's messages."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "a", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "b", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        # Pre-load the user's session with two prior turns.
        sess = gateway.sessions.get_or_create("ens-history")
        sess.conversation.add(Role.USER, "Describe a cat in one sentence.")
        sess.conversation.add(
            Role.ASSISTANT,
            "A small carnivorous mammal kept as a pet, known for its independence.",
        )

        seen_histories: dict[str, list[str]] = {}

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            # Record what each worker received as conversation context.
            seen_histories[worker.id] = [
                m.content for m in session.conversation.messages
            ]
            return Message(
                role=Role.ASSISTANT,
                content=f"shorter from {worker.id}",
                metadata={"remote_worker": worker.id, "tokens": 4, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={
                "message": "but shorter please",
                "session_id": "ens-history",
                "ensemble": True,
            },
        )
        assert resp.status_code == 200

        # Both workers saw the prior turns AND the new user message.
        for wid in ("a", "b"):
            hist = seen_histories.get(wid, [])
            assert any("cat" in m.lower() for m in hist), (
                f"worker {wid} missing prior context: {hist}"
            )
            assert any("shorter" in m.lower() for m in hist), (
                f"worker {wid} missing latest turn: {hist}"
            )

    def test_ask_ensemble_records_straggler_as_timeout(self, gateway, client):
        """A wedged worker can't extend the ensemble run beyond the
        slowest honest worker. The outer deadline cancels stragglers
        and records them as timeout contributions so the operator
        sees who lagged."""
        import asyncio

        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "fast", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "slow", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        # Shorten the deadline so the test doesn't take 90s.
        gateway.config.chat_fast_timeout = 0.2  # → ensemble bound ~0.3s

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            if worker.id == "slow":
                # Hang forever (will be cancelled by the deadline).
                await asyncio.sleep(60)
            return Message(
                role=Role.ASSISTANT,
                content="quick answer",
                metadata={"remote_worker": worker.id, "tokens": 3, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-slow", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Fast worker's answer wins (only one to contribute in time).
        assert "quick answer" in body["response"]
        # Straggler recorded with error="ensemble_timeout".
        contributions = body["ensemble_contributions"]
        slow_contrib = next(
            (c for c in contributions if c["worker_id"] == "slow"), None
        )
        assert slow_contrib is not None
        assert slow_contrib["error"] == "ensemble_timeout"

    def test_ask_ensemble_single_contribution_skips_synthesis(
        self, gateway, client,
    ):
        """When only one worker is idle (or only one returned text),
        there's nothing to arbitrate. Return that answer directly
        without burning local-agent compute on a single-answer
        'synthesis'."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "solo", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            return Message(
                role=Role.ASSISTANT,
                content="The only worker's answer.",
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # If synthesis fired this would be called — assert it wasn't.
        synth_called = []

        async def fake_step(conv):
            synth_called.append(True)
            return Message(role=Role.ASSISTANT, content="should not appear")

        gateway.agent.step = fake_step  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-solo", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "only worker" in body["response"]
        assert synth_called == [], "synthesis should not run for single contribution"

    def test_ask_ensemble_skips_empty_text_placeholder(self, gateway, client):
        """An empty-text fallback isn't a real contribution — the
        arbitrator must skip it and pick a real answer from another
        worker."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "small", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "large", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            if worker.id == "small":
                # Empty-text placeholder.
                return Message(
                    role=Role.ASSISTANT,
                    content="(placeholder)",
                    metadata={
                        "remote_worker": "small",
                        "empty_text_fallback": True,
                    },
                )
            return Message(
                role=Role.ASSISTANT,
                content="A real answer from the large worker.",
                metadata={"remote_worker": "large", "tokens": 8, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-2", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The placeholder must NOT have been picked.
        assert "placeholder" not in body["response"]
        assert "real answer" in body["response"]

    def test_ask_verify_must_be_bool(self, client):
        """Strict-bool guard on the opt-in `verify` flag — a truthy
        string like 'yes' would otherwise silently enable the
        two-worker pass."""
        for bad in ("yes", 1, "true", "false", [True]):
            resp = client.post(
                "/api/ask",
                json={"message": "hi", "session_id": "v-bad", "verify": bad},
            )
            assert resp.status_code == 400, f"accepted verify={bad!r}"

    def test_ask_verify_returns_corrected_answer(self, gateway, client):
        """End-to-end: when verify=true and a second worker is
        available, the verifier's correction replaces the primary's
        answer. This is the smallest piece of real multi-worker
        collaboration in the system — two workers acting on a single
        request, not just routing one request to one worker."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        # Register two workers so _pick_alternate_chat_worker finds one.
        gateway._workers.register(
            "primary", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"], "tools": False},
        )
        gateway._workers.register(
            "verifier", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"], "tools": False},
        )

        # Stub _quick_remote_infer to return different responses
        # depending on which worker is being called and which session.
        # The verifier session_id starts with "_verify_".
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            if session_id.startswith("_verify_"):
                # Verifier returns a corrected answer (not "VERIFIED").
                return Message(
                    role=Role.ASSISTANT,
                    content="Corrected: list.append adds one item; list.extend adds each item from an iterable.",
                    metadata={"remote_worker": worker.id, "tokens": 20, "tps": 5.0},
                )
            # Primary returns a wrong/incomplete answer.
            msg = Message(
                role=Role.ASSISTANT,
                content="They are the same.",
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # Force chat intent so the verify path runs.
        async def fake_route(_msg, _sid):
            return gateway._workers.get("primary"), "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "list.append vs extend?", "session_id": "v-corr", "verify": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The corrected answer wins, not the primary's wrong one.
        assert "Corrected" in body["response"]
        # Metadata exposes the verifier worker id and the correction flag.
        assert body.get("verified_by") == "verifier"
        assert body.get("verifier_corrected") is True
        assert body.get("primary_worker") == "primary"

    def test_ask_verify_accepts_lenient_verified_forms(self, gateway, client):
        """Models routinely add casing / punctuation / brief
        commentary around the literal VERIFIED token we ask for.
        Treat any short response containing VERIFIED as a
        confirmation; long substantive text means correction."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "primary", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "verifier", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        for verifier_output in (
            "VERIFIED",
            "verified",
            "Verified.",
            "VERIFIED!",
            "Yes, VERIFIED",
            "## VERIFIED",
        ):
            def make_fake_quick(out: str):
                async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
                    if session_id.startswith("_verify_"):
                        return Message(
                            role=Role.ASSISTANT, content=out,
                            metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
                        )
                    msg = Message(
                        role=Role.ASSISTANT, content="primary answer",
                        metadata={"remote_worker": worker.id, "tokens": 2, "tps": 5.0},
                    )
                    session.conversation.messages.append(msg)
                    return msg
                return fake_quick

            gateway._quick_remote_infer = make_fake_quick(verifier_output)  # type: ignore[method-assign]

            async def fake_route(_msg, _sid):
                return gateway._workers.get("primary"), "chat"

            gateway._route_by_role = fake_route  # type: ignore[method-assign]

            sid = f"v-lenient-{abs(hash(verifier_output))}"
            resp = client.post(
                "/api/ask",
                json={"message": "q", "session_id": sid, "verify": True},
            )
            assert resp.status_code == 200, f"failed for {verifier_output!r}"
            body = resp.json()
            assert body["response"] == "primary answer", (
                f"verifier output {verifier_output!r} did not confirm "
                f"primary; got response={body['response']!r}"
            )
            assert body.get("verifier_corrected") is False, (
                f"verifier output {verifier_output!r} treated as correction"
            )

    def test_ask_verify_long_response_treated_as_correction(
        self, gateway, client,
    ):
        """A long response from the verifier (even one that mentions
        VERIFIED) is a substantive correction, not a confirmation —
        we shouldn't drop the verifier's effort just because the
        word VERIFIED appears in passing."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "primary", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "verifier", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            if session_id.startswith("_verify_"):
                # Looks like a correction even though it mentions VERIFIED.
                return Message(
                    role=Role.ASSISTANT,
                    content=(
                        "The previous answer says Paris but the question "
                        "asked about Germany. The capital is Berlin. "
                        "(Not VERIFIED — corrected.)"
                    ),
                    metadata={"remote_worker": worker.id, "tokens": 30, "tps": 5.0},
                )
            msg = Message(
                role=Role.ASSISTANT, content="The capital is Paris.",
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return gateway._workers.get("primary"), "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "What is the capital of Germany?", "session_id": "v-long", "verify": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The substantive correction wins.
        assert "Berlin" in body["response"]
        assert body.get("verifier_corrected") is True

    def test_ask_verify_keeps_primary_when_verifier_confirms(
        self, gateway, client,
    ):
        """When the verifier returns exactly 'VERIFIED', the primary's
        answer is kept and the response metadata flags
        verifier_corrected=False so the caller can tell two-worker
        verification ran."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "primary", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"], "tools": False},
        )
        gateway._workers.register(
            "verifier", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"], "tools": False},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            if session_id.startswith("_verify_"):
                return Message(
                    role=Role.ASSISTANT,
                    content="VERIFIED",
                    metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
                )
            msg = Message(
                role=Role.ASSISTANT,
                content="A correct answer here.",
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return gateway._workers.get("primary"), "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "what?", "session_id": "v-ok", "verify": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "A correct answer here."
        assert body.get("verified_by") == "verifier"
        assert body.get("verifier_corrected") is False

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

    def test_ask_system_override_does_not_mutate_shared_config(self, gateway, client):
        """The `system` field per request must NOT mutate
        `self.config.identity` — that's shared mutable state and
        concurrent /api/ask calls with different overrides would
        race (req A's worker would see req B's identity). The
        identity_override flows as a per-call kwarg now."""
        from unittest.mock import AsyncMock
        from towel.agent.conversation import Message, Role
        from towel.gateway.workers import WorkerInfo

        baseline_identity = gateway.config.identity

        fake_worker = WorkerInfo(id="w1", ws=AsyncMock(), capabilities={})
        gateway._workers._workers["w1"] = fake_worker

        async def fake_route(message, session_id):
            return fake_worker, "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        captured: dict = {}

        async def fake_quick(
            session_id, session, worker, max_tokens=256, **kwargs,
        ):
            captured["identity_override"] = kwargs.get("identity_override")
            # Mid-call, the shared config.identity must still be the
            # baseline — if it had been mutated, a parallel request
            # would see the override.
            captured["mid_call_config_identity"] = gateway.config.identity
            return Message(
                role=Role.ASSISTANT,
                content="ok",
                metadata={"remote_worker": "w1"},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "x", "system": "be terse"},
        )
        assert resp.status_code == 200
        # Override was passed as a kwarg.
        assert captured["identity_override"] == "be terse"
        # Shared config was never touched.
        assert captured["mid_call_config_identity"] == baseline_identity
        assert gateway.config.identity == baseline_identity

    def test_ask_rejects_non_string_system(self, gateway, client):
        """`system` flows into `self.config.identity` for the
        request's lifetime. A non-string would either crash deeper
        in the dispatch path or corrupt the identity until the
        `finally` block restored it. Reject early."""
        for bad in (42, [1, 2], {"x": 1}, True):
            resp = client.post(
                "/api/ask",
                json={"message": "hi", "system": bad},
            )
            assert resp.status_code == 400, f"accepted system={bad!r}"
            assert "system" in resp.json()["error"]
        # Config identity must not have been touched.
        # (No assertion on the value — it was whatever config initialized with —
        # but it must still be a string.)
        assert isinstance(gateway.config.identity, str)

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

    def test_primary_failure_retries_on_alt(self, gateway, client):
        """When the primary worker crashes (timeout, connection lost),
        the alt worker should still get a try — same as the
        empty-response retry path. Operationally these are the same
        failure: 'primary didn't give us a useful answer.'"""
        from unittest.mock import AsyncMock
        from towel.agent.conversation import Message, Role
        from towel.gateway.workers import WorkerInfo

        fake_primary = WorkerInfo(id="primary-fail", ws=AsyncMock(), capabilities={})
        fake_alt = WorkerInfo(
            id="alt-good", ws=AsyncMock(),
            capabilities={"total_vram_mb": 16000},
        )
        gateway._workers._workers["primary-fail"] = fake_primary
        gateway._workers._workers["alt-good"] = fake_alt

        async def fake_route(message, session_id):
            return fake_primary, "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        call_log: list[str] = []

        async def fake_quick(session_id, session, worker, **kwargs):
            call_log.append(worker.id)
            if worker.id == "primary-fail":
                raise RuntimeError("worker primary-fail did not respond within 60s")
            msg = Message(
                role=Role.ASSISTANT,
                content="hi from alt",
                metadata={"remote_worker": "alt-good"},
            )
            session.conversation.messages.append(msg)
            return msg

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "primary-failure-retry"},
        )
        assert resp.status_code == 200
        # Both workers were attempted.
        assert call_log == ["primary-fail", "alt-good"]
        body = resp.json()
        assert body["response"] == "hi from alt"
        # fallback_from_worker should tag the primary.
        assert body["fallback_from_worker"] == "primary-fail"
        assert body["fallback_reason"] == "primary_failed"

    def test_error_response_carries_type_name_when_str_is_empty(self, gateway, client):
        """Several stdlib exceptions (asyncio.CancelledError,
        asyncio.TimeoutError, etc.) stringify to "". The handler used
        to return `{"error": ""}` HTTP 500 — unhelpful for any client
        trying to log or surface the failure. Now falls back to the
        type name when `str(exc)` is empty."""
        from unittest.mock import AsyncMock
        from towel.gateway.workers import WorkerInfo

        only_worker = WorkerInfo(id="w-empty-err", ws=AsyncMock(), capabilities={})
        gateway._workers._workers["w-empty-err"] = only_worker

        async def fake_route(message, session_id):
            return only_worker, "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_quick(session_id, session, worker, **kwargs):
            # `asyncio.TimeoutError` is the canonical empty-str
            # exception that motivated the helper. It's also what
            # bare `_quick_remote_infer` would convert via my earlier
            # fix, but we want to test the catch-all path here.
            raise TimeoutError()

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "empty-err"},
        )
        assert resp.status_code == 500
        # Critically the error field is non-empty. With str(TimeoutError())
        # being "", the helper falls back to the type name.
        body = resp.json()
        assert body["error"]
        assert body["error"] != ""
        # And it carries a useful identifier.
        assert "TimeoutError" in body["error"]

    def test_primary_failure_no_alt_re_raises(self, gateway, client):
        """If primary fails and there's no alt worker, the API
        caller should see the primary's exception bubble up as a
        500 — not silently masked."""
        from unittest.mock import AsyncMock
        from towel.gateway.workers import WorkerInfo

        only_worker = WorkerInfo(id="only-one", ws=AsyncMock(), capabilities={})
        gateway._workers._workers["only-one"] = only_worker

        async def fake_route(message, session_id):
            return only_worker, "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_quick(session_id, session, worker, **kwargs):
            raise RuntimeError("worker only-one did not respond within 60s")

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "single-worker-fail"},
        )
        assert resp.status_code == 500
        assert "only-one" in resp.json()["error"]
        assert "did not respond" in resp.json()["error"]

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

        async def fake_quick(
            session_id, session, worker, max_tokens=256, **kwargs,
        ):
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

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

    def test_sessions_channel_filter(self, gateway, client):
        """`?channel=` narrows live sessions to one channel — same
        semantics /api/sessions and /conversations expose, applied
        to the live in-memory set."""
        sess_a = gateway.sessions.get_or_create("api-sess")
        sess_a.conversation.channel = "api"
        sess_b = gateway.sessions.get_or_create("cli-sess")
        sess_b.conversation.channel = "cli"

        resp = client.get("/sessions?channel=cli")
        assert resp.status_code == 200
        ids = {s["id"] for s in resp.json()["sessions"]}
        assert ids == {"cli-sess"}

    def test_sessions_worker_filter(self, gateway, client):
        """`?worker=` narrows to sessions currently routed to a
        specific worker — unique to /sessions because the other
        list endpoints don't carry live routing state."""
        gateway.sessions.get_or_create("s1")
        gateway.sessions.get_or_create("s2")
        gateway.sessions.get_or_create("s3")
        gateway._session_workers["s1"] = "alpha"
        gateway._session_workers["s2"] = "beta"
        gateway._session_workers["s3"] = "alpha"

        resp = client.get("/sessions?worker=alpha")
        ids = {s["id"] for s in resp.json()["sessions"]}
        assert ids == {"s1", "s3"}

    def test_sessions_pinned_to_filter(self, gateway, client):
        """`?pinned_to=` narrows to sessions explicitly pinned to a
        given worker — fast answer to 'who's stuck on alpha?'"""
        gateway.sessions.get_or_create("p1")
        gateway.sessions.get_or_create("p2")
        gateway._session_pins["p1"] = "alpha"

        resp = client.get("/sessions?pinned_to=alpha")
        ids = {s["id"] for s in resp.json()["sessions"]}
        assert ids == {"p1"}

    def test_sessions_exposes_message_count_alias(self, gateway, client):
        """/sessions originally exposed the message count under
        `messages`, but /api/sessions and /conversations use
        `message_count` for the same datum. Clients hitting both
        endpoints had to special-case the field name. Expose
        `message_count` as an alias on /sessions while keeping
        `messages` for the existing web-UI / CLI callers."""
        from towel.agent.conversation import Role

        sess = gateway.sessions.get_or_create("alias-session")
        sess.conversation.add(Role.USER, "hi")
        sess.conversation.add(Role.ASSISTANT, "hello")

        data = client.get("/sessions").json()
        entry = data["sessions"][0]
        # Both names present, both reflect the same count.
        assert entry["messages"] == 2
        assert entry["message_count"] == 2


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

    def test_cluster_nodes_includes_roles_and_assigned_tasks(
        self, gateway, client,
    ):
        """`/cluster/nodes` previously omitted roles + assigned_tasks
        even though /workers carried them — operators debugging "why
        did this task not land on this worker" had to bounce between
        endpoints to compare hardware capabilities (cluster/nodes)
        against routing eligibility (/workers). Surface both here so
        the cluster panel can answer routing questions without a
        second roundtrip."""
        from unittest.mock import MagicMock

        from towel.nodes.roles import NodeRole, TaskType

        caps = {"backend": "llama", "modes": ["llama_chat"]}
        gateway._workers.register("rolled", MagicMock(), caps)
        # /cluster/nodes iterates the node_tracker, not the registry —
        # register there too so the worker actually appears.
        gateway._node_tracker.register("rolled", caps)
        gateway._node_roles["rolled"] = [NodeRole.INFERENCE, NodeRole.GENERAL]
        gateway._node_tasks["rolled"] = [TaskType.CHAT, TaskType.EXPLAIN]

        resp = client.get("/cluster/nodes")
        entry = resp.json()["nodes"]["rolled"]
        # The serialized forms are strings (the same way /workers
        # emits them) — JSON-safe and matching the UI's expectations.
        assert "roles" in entry
        assert "inference" in [r.lower() for r in entry["roles"]]
        assert "general" in [r.lower() for r in entry["roles"]]
        assert "assigned_tasks" in entry
        assert "chat" in [t.lower() for t in entry["assigned_tasks"]]
        assert "explain" in [t.lower() for t in entry["assigned_tasks"]]

    def test_cluster_nodes_roles_default_empty_for_unknown_workers(
        self, gateway, client,
    ):
        """Stale tracker entries (no live worker) should still get
        the new fields — both as empty lists, matching the same
        explicit-null contract the other operator-state fields use
        but for list-typed data."""
        gateway._node_tracker.register(
            "ghost-roleless",
            {"backend": "llama", "context_window": 8192, "max_tokens": 4096},
        )
        resp = client.get("/cluster/nodes")
        entry = resp.json()["nodes"]["ghost-roleless"]
        assert entry["roles"] == []
        assert entry["assigned_tasks"] == []

    def test_cluster_nodes_includes_quality_tier(self, gateway, client):
        """/cluster/nodes now surfaces the same low/medium/high
        quality_tier /workers exposes — operators previously had to
        eyeball total_vram_mb in resources and apply the bucketing
        rule themselves. Tier is derived from the same dispatcher
        signal so workers labelled `low` here won't surprise anyone
        when they're filtered out of a CODE_REVIEW dispatch."""
        from unittest.mock import MagicMock

        # High tier: 16GB VRAM crosses the high threshold.
        big_caps = {
            "backend": "llama",
            "modes": ["llama_chat"],
            "total_vram_mb": 16000,
            "context_window": 8192,
        }
        gateway._workers.register("big", MagicMock(), big_caps)
        gateway._node_tracker.register("big", big_caps)

        # Low tier: small VRAM, small context.
        small_caps = {
            "backend": "llama",
            "modes": ["llama_chat"],
            "total_vram_mb": 1024,
            "context_window": 2048,
        }
        gateway._workers.register("tiny", MagicMock(), small_caps)
        gateway._node_tracker.register("tiny", small_caps)

        resp = client.get("/cluster/nodes")
        nodes = resp.json()["nodes"]
        assert nodes["big"]["quality_tier"] == "high"
        assert nodes["tiny"]["quality_tier"] == "low"

    def test_cluster_nodes_quality_tier_synthesizes_for_stale_entries(
        self, gateway, client,
    ):
        """When the node tracker has an entry but the registry
        doesn't (worker disconnected mid-session), we still want a
        meaningful quality_tier. Synthesize it from the tracker's
        own resources fields — the field names there differ from
        WorkerInfo.capabilities (vram_total_mb vs total_vram_mb) so
        map at the boundary."""
        gateway._node_tracker.register(
            "stale-big",
            {
                "backend": "llama",
                "context_window": 8192,
                "max_tokens": 4096,
                "total_vram_mb": 24000,  # high tier
            },
        )
        resp = client.get("/cluster/nodes")
        entry = resp.json()["nodes"]["stale-big"]
        # Without the synthesis the field would have been "unknown"
        # or "low" — the tracker had the data, just under a different
        # field name.
        assert entry["quality_tier"] == "high"


class TestClusterHandoffs:
    """/cluster/handoffs returns the same stats + history shape but
    now respects ?limit= and ?only_failed=1 — the earlier handler
    ignored the URL entirely so an operator triaging a recent
    disconnect storm couldn't see past the most recent 20 records."""

    def _seed_handoffs(self, gateway, n_success: int, n_failed: int) -> None:
        from datetime import UTC, datetime

        from towel.gateway.handoff import HandoffReason, HandoffRecord

        for i in range(n_success):
            r = HandoffRecord(
                session_id=f"ok-{i}",
                from_worker_id="a",
                to_worker_id="b",
                reason=HandoffReason.WORKER_DRAINING,
                started_at=datetime.now(UTC),
            )
            r.complete(success=True)
            gateway._handoff_manager._history.append(r)
        for i in range(n_failed):
            r = HandoffRecord(
                session_id=f"fail-{i}",
                from_worker_id="a",
                to_worker_id="c",
                reason=HandoffReason.WORKER_DISCONNECTED,
                started_at=datetime.now(UTC),
            )
            r.complete(success=False, error="simulated")
            gateway._handoff_manager._history.append(r)

    def test_limit_respected(self, gateway, client):
        """`?limit=` caps the recent records returned — earlier
        signature always returned 20 regardless of the URL."""
        self._seed_handoffs(gateway, n_success=15, n_failed=0)
        resp = client.get("/cluster/handoffs?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["recent"]) == 5
        # Stats still reflect the full history, not the limited slice.
        assert body["stats"]["total"] == 15

    def test_limit_rejects_garbage(self, client):
        resp = client.get("/cluster/handoffs?limit=junk")
        assert resp.status_code == 400

    def test_only_failed_filter(self, gateway, client):
        """`?only_failed=1` narrows to handoffs that didn't succeed —
        operators triaging "what's going wrong" can skip past the
        successful migrations without grepping client-side."""
        self._seed_handoffs(gateway, n_success=3, n_failed=2)
        resp = client.get("/cluster/handoffs?only_failed=1")
        assert resp.status_code == 200
        recent = resp.json()["recent"]
        assert len(recent) == 2
        assert all(not r["success"] for r in recent)

    def test_reason_filter(self, gateway, client):
        """`?reason=` narrows to a single HandoffReason — the seeded
        records cover worker_draining (success) and
        worker_disconnected (failure), filter picks each one out."""
        self._seed_handoffs(gateway, n_success=2, n_failed=3)
        resp = client.get("/cluster/handoffs?reason=worker_draining")
        recent = resp.json()["recent"]
        assert len(recent) == 2
        assert all(r["reason"] == "worker_draining" for r in recent)

        resp = client.get("/cluster/handoffs?reason=worker_disconnected")
        recent = resp.json()["recent"]
        assert len(recent) == 3
        assert all(r["reason"] == "worker_disconnected" for r in recent)

    def test_from_worker_and_to_worker_filters(self, gateway, client):
        """`?from_worker=X` answers "what did worker X shed?" and
        `?to_worker=X` answers "what did X inherit?". The seeded
        records all originate on worker "a"; success cases land on
        "b", failures on "c". Parity with dispatch_recent's
        worker/previous_worker filter pair so operators don't have
        to learn a different filter vocabulary per endpoint."""
        self._seed_handoffs(gateway, n_success=2, n_failed=3)
        # All 5 came from "a".
        resp = client.get("/cluster/handoffs?from_worker=a")
        recent = resp.json()["recent"]
        assert len(recent) == 5
        assert all(r["from_worker_id"] == "a" for r in recent)
        # 2 success landed on "b".
        resp = client.get("/cluster/handoffs?to_worker=b")
        recent = resp.json()["recent"]
        assert len(recent) == 2
        assert all(r["to_worker_id"] == "b" for r in recent)
        # Unknown worker → empty (no false positives).
        resp = client.get("/cluster/handoffs?from_worker=ghost")
        assert resp.json()["recent"] == []

    def test_from_to_worker_length_cap(self, client):
        """Same 256-char cap as /dispatch/recent's worker filters —
        bogus large values shouldn't bloat the request line."""
        resp = client.get(
            "/cluster/handoffs?from_worker=" + "x" * 257
        )
        assert resp.status_code == 400

    def test_pending_handoffs_carry_elapsed_ms(self, gateway, client):
        """In-progress handoff records expose `elapsed_ms` (now -
        started_at) so operators triaging "what's been stuck the
        longest?" can sort by it directly. Completed handoffs
        already carry `duration_ms`; this is the pending equivalent."""
        import time
        from datetime import UTC, datetime

        from towel.gateway.handoff import HandoffReason, HandoffRecord

        # Started ~50ms ago — elapsed should reflect at least that.
        active = HandoffRecord(
            session_id="elapsed-sess",
            from_worker_id="a",
            to_worker_id="b",
            reason=HandoffReason.WORKER_DRAINING,
            started_at=datetime.now(UTC),
        )
        gateway._handoff_manager._pending["elapsed-sess"] = active
        time.sleep(0.05)

        resp = client.get("/cluster/handoffs")
        body = resp.json()
        assert len(body["pending"]) == 1
        entry = body["pending"][0]
        assert "elapsed_ms" in entry
        # At least 40ms since started (allow scheduler slack).
        assert entry["elapsed_ms"] >= 40

    def test_pending_handoffs_in_response(self, gateway, client):
        """In-progress handoffs surface alongside the completed
        history — operators triaging "what's stuck right now?" need
        more than the bare count. The pending list carries the same
        record shape as the recent list (minus completed_at fields)."""
        from datetime import UTC, datetime

        from towel.gateway.handoff import HandoffReason, HandoffRecord

        # Seed one pending (started, never completed).
        active = HandoffRecord(
            session_id="stuck-sess",
            from_worker_id="a",
            to_worker_id="b",
            reason=HandoffReason.WORKER_DRAINING,
            started_at=datetime.now(UTC),
        )
        gateway._handoff_manager._pending["stuck-sess"] = active

        resp = client.get("/cluster/handoffs")
        assert resp.status_code == 200
        body = resp.json()
        assert "pending" in body
        assert len(body["pending"]) == 1
        entry = body["pending"][0]
        assert entry["session_id"] == "stuck-sess"
        assert entry["reason"] == "worker_draining"
        # Stats still report the bare count for legacy callers.
        assert body["stats"]["pending"] == 1

    def test_reason_filter_rejects_unknown(self, client):
        """A typo like `?reason=draining` (missing the worker_
        prefix) returns 400 with the list of valid reasons —
        better than silent empty results that look like "the
        coordinator is idle"."""
        resp = client.get("/cluster/handoffs?reason=draining")
        assert resp.status_code == 400
        # All five enum values listed for fixing the typo.
        for r in (
            "worker_draining", "worker_disconnected", "worker_overloaded",
            "manual_rebalance", "capacity_exceeded",
        ):
            assert r in resp.json()["error"]


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

    def test_pin_worker_warns_when_pinning_to_disabled(self, gateway, client):
        """Pinning to a disabled worker still takes effect (the pin
        will fire when the worker is re-enabled), but the operator
        deserves a heads-up — otherwise the next request will go to
        a different worker and they'll wonder why their pin didn't
        work."""
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )
        gateway._workers.set_enabled("desktop-1", False)

        resp = client.post(
            "/sessions/chat-1/pin-worker", json={"worker_id": "desktop-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pinned"] is True
        assert "warning" in body
        assert "disabled" in body["warning"]

    def test_pin_worker_warns_when_pinning_to_draining(self, gateway, client):
        """Symmetric to the disabled case — pinning to a draining
        worker should warn so operators don't see silent pin_missed
        routing on the next request without context."""
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )
        gateway._workers.set_draining("desktop-1", True)

        resp = client.post(
            "/sessions/chat-1/pin-worker", json={"worker_id": "desktop-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pinned"] is True
        assert "warning" in body
        assert "draining" in body["warning"]

    def test_pin_worker_no_warning_for_routable_worker(self, gateway, client):
        """A pin to a healthy worker should NOT include a warning —
        otherwise tests/automation would either ignore the field or
        be tripped up by it."""
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )
        resp = client.post(
            "/sessions/chat-1/pin-worker", json={"worker_id": "desktop-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pinned"] is True
        assert "warning" not in body


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

    def test_list_channel_filter(self, store, client):
        """`?channel=` narrows results to conversations created on a
        specific channel. Operators looking through a mixed archive
        previously had to client-filter — /api/sessions and
        /conversations both returned everything regardless of source."""
        for sid, ch in (("cli-1", "cli"), ("api-1", "api"), ("api-2", "api")):
            conv = Conversation(id=sid, channel=ch)
            conv.add(Role.USER, "hi")
            store.save(conv)

        resp = client.get("/conversations?channel=api")
        assert resp.status_code == 200
        ids = {c["id"] for c in resp.json()["conversations"]}
        assert ids == {"api-1", "api-2"}

    def test_list_tag_filter(self, store, client):
        """`?tag=` narrows results to conversations carrying a given
        tag — same data path /api/sessions exposes, now filterable."""
        for sid, tags in (
            ("plain", []),
            ("work-1", ["work"]),
            ("work-urgent", ["work", "urgent"]),
        ):
            conv = Conversation(id=sid, channel="api")
            conv.tags = tags
            conv.add(Role.USER, "hi")
            store.save(conv)

        resp = client.get("/conversations?tag=work")
        ids = {c["id"] for c in resp.json()["conversations"]}
        assert ids == {"work-1", "work-urgent"}

        resp = client.get("/conversations?tag=urgent")
        ids = {c["id"] for c in resp.json()["conversations"]}
        assert ids == {"work-urgent"}

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

    def test_get_conversation_includes_routing_state(
        self, gateway, store, client,
    ):
        """Conversation detail mirrors /api/sessions / /sessions and
        surfaces routing state (current worker + pin) alongside the
        persisted conversation. Without this, a UI opening a single
        conversation had to issue a second call to one of the list
        endpoints just to render the pin badge — and the answer was
        sometimes missing entirely because /sessions only carries
        live in-memory entries."""
        from unittest.mock import MagicMock

        conv = Conversation(id="routing-detail", channel="api")
        conv.add(Role.USER, "hi")
        store.save(conv)

        gateway._workers.register(
            "alpha", MagicMock(), {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._session_pins["routing-detail"] = "alpha"

        resp = client.get("/conversations/routing-detail")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pinned_worker_id"] == "alpha"
        # Not currently routed (no live affinity entry).
        assert body["worker_id"] is None

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

    def test_delete_all_clears_session_pins(
        self, gateway, store, client, tmp_path,
    ):
        """Delete-all clears the in-memory session affinity dict +
        node tracker context slots, but previously left
        `_session_pins` untouched. Pins for now-deleted conversations
        then re-persisted on the next /sessions/<id>/pin-worker save,
        showing up as ghost entries in the on-disk pin file. Operators
        who used delete-all to reset state and then re-pinned a new
        session found stale entries reappearing.

        Parity with `test_delete_clears_session_worker_pin` for the
        single-conversation delete path."""
        from unittest.mock import MagicMock

        gateway._workers.register(
            "alpha", MagicMock(), {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "beta", MagicMock(), {"backend": "llama", "modes": ["llama_chat"]},
        )

        # Two conversations + two pins.
        for sid, worker in (("conv-1", "alpha"), ("conv-2", "beta")):
            conv = Conversation(id=sid)
            conv.add(Role.USER, "hi")
            store.save(conv)
            gateway._session_pins[sid] = worker
        assert len(gateway._session_pins) == 2

        resp = client.request("DELETE", "/conversations?confirm=yes")
        assert resp.status_code == 200
        # All in-memory pins gone after delete-all.
        assert gateway._session_pins == {}
        # Persisted state is also empty so the next save can't
        # resurrect the stale entries.
        assert gateway.pin_store.load() == {}

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

    def test_export_json_pretty_default(self, store, client):
        """JSON exports default to pretty-printed (indent=2) so a
        browser/curl operator sees readable output. Existing
        behaviour — guarded by test so a flag flip doesn't break
        operators who eyeballed `format=json` responses."""
        conv = Conversation(id="exp-pretty", channel="cli")
        conv.add(Role.USER, "hi")
        store.save(conv)

        resp = client.get("/conversations/exp-pretty/export?format=json")
        assert resp.status_code == 200
        # Indented JSON has newlines between top-level fields.
        assert "\n" in resp.text

    def test_export_json_compact_opt_in(self, store, client):
        """`?pretty=0` opts into compact JSON for piping into jq or
        any byte-conscious tooling. Earlier the gateway always
        returned pretty regardless — operators wanting compact had
        to either pipe through jq -c or run their own Python."""
        conv = Conversation(id="exp-compact", channel="cli")
        conv.add(Role.USER, "hi")
        store.save(conv)

        resp = client.get(
            "/conversations/exp-compact/export?format=json&pretty=0"
        )
        assert resp.status_code == 200
        # Compact JSON: no indenting newlines.
        assert "\n" not in resp.text
        # Still valid JSON.
        import json as _json
        data = _json.loads(resp.text)
        assert data["id"] == "exp-compact"

    def test_export_html(self, store, client):
        """export_html already shipped in the persistence layer (and
        is tested in test_export.py for the rendering itself) but
        the gateway route only exposed markdown/json/text. Operators
        wanting to share a conversation as a single openable file
        had to either run a python repl or drop down to the library
        directly; now `format=html` returns the same dark-themed
        page the export tests already validate."""
        conv = Conversation(id="exp-4", channel="api")
        conv.add(Role.USER, "what is towel?")
        conv.add(Role.ASSISTANT, "an agent runtime — don't panic")
        store.save(conv)

        resp = client.get("/conversations/exp-4/export?format=html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<!DOCTYPE html>" in resp.text
        assert "an agent runtime" in resp.text
        # Filename uses the .html extension so saving in a browser
        # produces a sensible name.
        assert ".html" in resp.headers.get("content-disposition", "")

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
        # All four supported formats listed in the error so the
        # caller can fix the typo without checking docs.
        for f in ("markdown", "json", "text", "html"):
            assert f in resp.json()["error"]

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

    def test_export_filename_uses_conversation_title_when_set(
        self, store, client,
    ):
        """If the conversation has a title, the export filename
        should use it instead of the session_id. "towel-How-to-
        deploy.md" is far more useful than "towel-openai-chatcmp.md"
        for a saved file the operator wants to find later."""
        conv = Conversation(id="openai-chatcmpl-abc123def", channel="api")
        conv.title = "How to deploy"
        conv.add(Role.USER, "hi")
        store.save(conv)

        resp = client.get(
            "/conversations/openai-chatcmpl-abc123def/export?format=markdown"
        )
        assert resp.status_code == 200
        cd = resp.headers["content-disposition"]
        # Filename uses the title (spaces → hyphens, alphanumerics +
        # -_ preserved). The session_id doesn't appear because the
        # title takes precedence.
        assert "How-to-deploy" in cd
        assert "openai-chatcmpl" not in cd

    def test_export_filename_falls_back_to_conv_id_when_no_title(
        self, store, client,
    ):
        """Without a title, the filename stem is the sanitized
        session_id — keeps the previous behaviour for untitled
        conversations."""
        conv = Conversation(id="raw-id-here", channel="api")
        # No title set.
        conv.add(Role.USER, "hi")
        store.save(conv)

        resp = client.get("/conversations/raw-id-here/export?format=markdown")
        cd = resp.headers["content-disposition"]
        assert "raw-id-here" in cd


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

    def test_search_role_filter(self, store, client):
        """`?role=user` narrows results to messages from a single
        role. Without this, an operator searching for "hello" got
        every hit including the assistant's friendly greetings —
        which is rarely what they wanted."""
        conv = Conversation(id="role-search")
        conv.add(Role.USER, "hello towel")
        conv.add(Role.ASSISTANT, "hello back")
        store.save(conv)

        # No filter: both hits.
        resp = client.get("/search?q=hello")
        assert resp.status_code == 200
        results = resp.json()["results"]
        if results:
            matches = results[0]["matches"]
            roles = {m["role"] for m in matches}
            assert {"user", "assistant"}.issubset(roles)

        # User-only filter: just the user line.
        resp = client.get("/search?q=hello&role=user")
        assert resp.status_code == 200
        roles = {
            m["role"] for r in resp.json()["results"] for m in r["matches"]
        }
        assert roles == {"user"}, roles

    def test_search_role_filter_rejects_unknown(self, client):
        """A typo like ?role=usr should fail fast with 400 rather
        than silently match nothing — the empty-result state is
        indistinguishable from the user genuinely finding zero hits."""
        resp = client.get("/search?q=hi&role=usr")
        assert resp.status_code == 400
        assert "role" in resp.json()["error"].lower()

    def test_search_regex_mode(self, store, client):
        """`?regex=1` lets operators use a real regex when substring
        match isn't expressive enough — e.g. "deploy(ed|ing)" to
        catch tense variations. Without it they had to file a
        separate hit per spelling."""
        conv = Conversation(id="regex-search")
        conv.add(Role.USER, "deployed yesterday and deploying again now")
        store.save(conv)

        resp = client.get("/search?q=deploy%28ed%7Cing%29&regex=1")
        assert resp.status_code == 200
        assert len(resp.json()["results"]) >= 1

    def test_search_regex_invalid_pattern_rejected(self, store, client):
        """A malformed regex previously silently returned [] from
        the store — operators couldn't tell apart "no results" from
        "your pattern is broken". Validate at the gateway."""
        # Unclosed group.
        resp = client.get("/search?q=%28unclosed&regex=1")
        assert resp.status_code == 400
        assert "regex" in resp.json()["error"].lower()

    def test_search_surfaces_conversation_title(self, store, client):
        """Search results previously omitted `title` — UIs fell back
        to the conversation_id (e.g. "openai-chatcmpl-abc123") in the
        results panel, which is meaningless for browsing. Title now
        rides alongside conversation_id so the operator-readable name
        appears in search hits."""
        conv = Conversation(id="search-titled")
        conv.title = "How to deploy"
        conv.add(Role.USER, "tell me about ingress controllers")
        store.save(conv)

        resp = client.get("/search?q=ingress")
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1, results
        hit = next(r for r in results if r["conversation_id"] == "search-titled")
        assert hit["title"] == "How to deploy"

    def test_search_title_empty_for_untitled_conversations(self, store, client):
        """Untitled conversations get an empty-string title rather
        than the field being absent — keeps the response shape
        uniform so client code doesn't need an `if "title" in hit`
        special case."""
        conv = Conversation(id="search-untitled")
        # No conv.title set.
        conv.add(Role.USER, "find this needle")
        store.save(conv)

        resp = client.get("/search?q=needle")
        assert resp.status_code == 200
        hit = next(
            r for r in resp.json()["results"]
            if r["conversation_id"] == "search-untitled"
        )
        assert "title" in hit
        assert hit["title"] == ""


class TestDispatchRecentEphemeralFilter:
    """/dispatch/recent hides ephemeral collaboration sessions by
    default — each ensemble run records one decision per fan-out
    worker, which would otherwise dominate the view and confuse
    operators looking for actual user sessions. Opt-in via
    `?include_ephemeral=1`."""

    def _record_decision(self, gateway, session_id: str):
        from towel.gateway.dispatcher import DispatchDecision
        d = DispatchDecision(
            worker=None, intent="task", reason="test",
            session_id=session_id,
        )
        gateway._dispatcher._history.append(d)

    def test_ensemble_aggregate_dispatch_has_timing(self, gateway, client):
        """The aggregate `ensemble` dispatch entry stamps total_ms so
        an operator can see latency-at-a-glance without drilling
        into the per-worker decisions. total_ms = slowest-contributor
        + synthesis_ms (parallel fan-out is bound by slowest)."""
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
            return Message(
                role=Role.ASSISTANT, content=f"answer from {worker.id}",
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-time-disp", "ensemble": True},
        )
        assert resp.status_code == 200

        agg = client.get("/dispatch/recent?session=ens-time-disp").json()
        ensemble_entries = [d for d in agg["decisions"] if d.get("reason") == "ensemble"]
        assert ensemble_entries
        entry = ensemble_entries[0]
        # total_ms reflects slowest worker (synthesis didn't run on
        # this test path because the stubbed agent.step/generate
        # isn't loaded). Whatever the actual value, it should be a
        # non-negative number.
        assert "total_ms" in entry
        assert entry["total_ms"] >= 0

    def test_ensemble_records_aggregate_dispatch_entry(self, gateway, client):
        """When ensemble runs, an aggregate dispatch entry is
        recorded under the USER session_id (not the ephemeral
        per-worker ids) so operators see a single 'ensemble ran for
        session X' entry alongside other dispatch events."""
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
            return Message(
                role=Role.ASSISTANT, content=f"answer from {worker.id}",
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-dispatch", "ensemble": True},
        )
        assert resp.status_code == 200

        # Find the ensemble entry under the user's session_id.
        agg = client.get("/dispatch/recent?session=ens-dispatch").json()
        decisions = agg["decisions"]
        ensemble_entries = [d for d in decisions if d.get("reason") == "ensemble"]
        assert len(ensemble_entries) == 1, decisions
        entry = ensemble_entries[0]
        assert "2/2 answered" in entry["notes"]

    def test_ensemble_skip_with_no_idle_workers_records_dispatch(
        self, gateway, client,
    ):
        """When ensemble is requested but no workers are idle, the
        request silently fell through to single-worker dispatch — an
        operator looking at /dispatch/recent had no way to see that
        ensemble was attempted and skipped. Now the skip is logged
        as a distinct dispatch entry with notes='ensemble: skipped'
        so the operator can see why the response looks single-worker
        despite the ensemble flag.

        Reproducer for the live-coordinator behavior observed in
        2026-05: two requests with `ensemble=true` returned single-
        worker responses with no trace of ensemble anywhere in the
        log."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        # No workers registered → ensemble has 0 candidates.
        # We still need a worker for the fallthrough single-worker
        # path so the request completes, but it must be filtered out
        # of ensemble candidates. The easiest way to get that is to
        # mark it busy via the workers' busy() lifecycle.
        gateway._workers.register(
            "a", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        # Mark the only worker busy via the public lifecycle setter.
        # _ensemble_dispatch skips busy workers (fan-out wants real
        # concurrency, not queue thrash), so this leaves zero
        # candidates and exercises the skip path.
        gateway._workers.assign("a", "job-1", "other-session")

        # Single-worker fallback uses _route_by_role; mock it to
        # return None so the request completes via the local agent
        # path rather than waiting for the busy worker.
        async def fake_route(_msg, _sid):
            return None, "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_step(_conv, **_kwargs):
            return Message(role=Role.ASSISTANT, content="local fallback")
        gateway.agent.step = fake_step  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-skip", "ensemble": True},
        )
        assert resp.status_code == 200

        # Aggregate ensemble entry must exist with notes='skipped'
        # so the operator sees the no-candidates case in the log.
        agg = client.get("/dispatch/recent?session=ens-skip").json()
        ensemble_entries = [
            d for d in agg["decisions"] if d.get("reason") == "ensemble"
        ]
        assert len(ensemble_entries) == 1, agg["decisions"]
        entry = ensemble_entries[0]
        assert "skipped" in entry["notes"], entry["notes"]
        assert "no idle workers" in entry["notes"], entry["notes"]
        # candidates_considered=0 distinguishes "no candidates" from
        # "candidates ran but failed" in the operator UI.
        assert entry["candidates_considered"] == 0
        # The response body itself must surface the skip too —
        # symmetric to verify_skipped. Clients that don't read the
        # dispatch log still get a signal that their `ensemble=true`
        # silently degraded.
        body = resp.json()
        assert body.get("ensemble_skipped") is True
        assert "no idle workers" in body.get("ensemble_skip_reason", "")

    def test_ws_stream_response_wraps_exceptions(self):
        """`_stream_response` previously had no try/except — a model
        crash mid-iteration tore down the WS connection (the
        exception propagated past the caller's CancelledError-only
        handler). Now it catches Exception, emits an
        AgentEvent.error frame, and lets the connection survive
        for subsequent messages."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._stream_response)
        # Body wraps the async-for iteration in try/except.
        assert "try:" in src
        assert "AgentEvent.error" in src
        # asyncio.CancelledError is re-raised so the caller's
        # dedicated cancellation event still fires for that path.
        assert "asyncio.CancelledError" in src
        assert "raise" in src

    def test_preempt_idle_task_yields_after_cancel(self):
        """Live observation: preempting an active idle generation
        (especially the long-running PROACTIVE_HELP path) and
        immediately dispatching a chat on the same worker produced
        empty-text responses ≈100% of the time. The coordinator
        sent ``cancel_job``, released the worker, and fired the new
        infer all in the same event-loop tick — the worker saw
        infer before its cancel handler ran.

        ``await asyncio.sleep(0)`` after the release lets the
        cancel_job WS frame land ahead of the next dispatch's
        write. Pinning the yield via source inspection so a future
        refactor can't silently re-introduce the same-tick race."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._preempt_idle_task)
        # The yield must come AFTER the worker.release call —
        # otherwise the worker stays busy on the coordinator side
        # while the cancel is still in flight.
        release_pos = src.index("self._workers.release(worker.id)")
        sleep_pos = src.index("await asyncio.sleep(0)")
        assert release_pos < sleep_pos, (
            "asyncio.sleep(0) must come after _workers.release "
            "so the cancel_job frame flushes before the next "
            "dispatch's assign"
        )

    def test_ws_unknown_register_role_warns(self):
        """A typo'd `role` in a WS register message (e.g. "workre")
        previously got channel treatment silently — the worker
        never appeared in /workers and the operator had no
        breadcrumb. The handler now logs at WARNING for any role
        outside the known set."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        assert "unrecognized role" in src
        # Both valid roles named in the warning so the operator's
        # log entry is self-correcting.
        assert "'worker', 'channel'" in src

    def test_ws_unknown_msg_type_logged(self):
        """Unknown WS message types previously fell through silently
        — a client sending `{"type": "messsage"}` (typo) got no
        response and no log entry, leaving the operator with no
        clue why the message disappeared. Confirm the known-types
        allowlist + unknown-type debug log exist."""
        import inspect

        from towel.gateway import server as server_mod
        from towel.gateway.server import GatewayServer

        # Module-level allowlist contains every type the handler
        # dispatches on.
        known = server_mod._WS_KNOWN_TYPES
        for t in (
            "register", "heartbeat", "memory_sync",
            "job_event", "job_done", "job_error",
            "cancel", "message",
        ):
            assert t in known, f"missing {t!r} from allowlist"

        # The handler checks against it and logs unknowns.
        src = inspect.getsource(GatewayServer._handle_ws)
        assert "_WS_KNOWN_TYPES" in src
        assert "unknown type" in src

    def test_ws_accepts_max_tokens_and_temperature_overrides(self):
        """WS clients were pinned to defaults (max_tokens=256,
        temperature=0.7) because the handler called
        _quick_remote_infer with hardcoded values, ignoring whatever
        the WS message specified. /api/ask has accepted these knobs
        since the OpenAI-parity fix; WS clients wanting deterministic
        output (temperature=0) or longer responses (max_tokens=2048)
        had no way to express that.

        Source-inspection test — confirms the parsing + the call site
        passes the parsed values through. A full WS round-trip would
        need a websocket harness."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # Parsing exists for both knobs, with the WS-namespaced
        # variable names the call site uses.
        assert "ws_max_tokens" in src
        assert "ws_temperature" in src
        # Clamp ranges mirror /api/ask: max_tokens [1, 4096],
        # temperature [0, 2].
        assert "min(int(_mt_raw), 4096)" in src
        assert "0.0 <= _temp_val <= 2.0" in src
        # The call site uses the parsed values, not the old hardcoded
        # max_tokens=256. (Bad inputs fall back to defaults — WS has
        # no clean 400 path the way HTTP does.)
        assert "max_tokens=ws_max_tokens" in src
        assert "temperature=ws_temperature" in src

    def test_ws_stream_with_collab_logs_warning(self):
        """WS doesn't have an HTTP-style 400 path, so the openai-
        compat-equivalent 'verify/ensemble with stream=true is
        rejected' error becomes a server-side log warning instead.
        Operators see the degradation in the coordinator log instead
        of wondering why a WS client's `ensemble=true` did nothing.

        Source-inspection test — confirms the warning string exists
        in the WS handler. A full WS round-trip would need a
        websocket harness."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # The warning fires when stream=true is set alongside
        # ensemble_flag or verify_flag.
        assert (
            "stream and (ensemble_flag or verify_flag)" in src
        ), "WS handler must log when collab flags are silently ignored on streaming"
        # The log message points operators at the fix (stream=false).
        assert "stream=false" in src

    def test_ws_ensemble_skipped_surfaced_in_metadata(self):
        """Parity with /api/ask's ensemble_skipped field: when
        ensemble=true was set on a WS message but _ensemble_dispatch
        returned no arbitrated answer (no candidates / all empty),
        the WS response metadata now carries `ensemble_skipped` +
        `ensemble_skip_reason` so client UIs render the same "ensemble
        attempted but couldn't run" badge."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        assert "ensemble_skipped" in src
        # Both reason variants are present.
        assert "no idle workers available" in src
        assert "all workers tool-looped" in src

    def test_ws_verify_skipped_surfaced_in_metadata(self):
        """Parity with /api/ask's verify_skipped response field: when
        verify=true was set on a WS message but the verify pass
        couldn't run (no alternate worker, or primary returned an
        empty placeholder), the WS response metadata now carries
        `verify_skipped` + `verify_skip_reason` so client UIs can
        render the same "verify attempted but couldn't run" badge."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # Both skip-reason variants present in the WS handler.
        assert '"verify_skipped": True' in src
        assert "no alternate worker available" in src
        assert "nothing to verify" in src

    def test_ws_ensemble_response_includes_contributions_for_parity(self):
        """The WS ensemble response previously sent only `ensemble`,
        `ensemble_arbitration`, and `remote_worker` in the metadata —
        but /api/ask has carried `ensemble_contributions` since the
        ensemble feature shipped. WS clients had no way to render
        "Workers: A, B, C" the way HTTP-side UIs do.

        Source-inspection test: the WS message handler must thread
        `_contribs` into the response metadata under the
        `ensemble_contributions` key. (A full WS round-trip would
        need a websocket harness; this is the lighter equivalent.)"""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # The relevant metadata block builds in the ensemble short-
        # circuit (search around the `if arbitrated:` branch).
        assert '"ensemble_contributions": _contribs' in src, (
            "WS ensemble path must include ensemble_contributions in "
            "the response metadata (parity with /api/ask)."
        )

    def test_verify_skipped_when_no_alternate_records_dispatch(
        self, gateway, client,
    ):
        """When verify=true but only one worker is registered, the
        verifier pass falls through without finding an alternate.
        Previously the dispatch log showed nothing — operators saw a
        verify=true request that didn't get verified and had no
        record explaining why. Now a "verify: skipped" aggregate
        entry surfaces with the skip reason."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        # Only ONE worker — _verify_pass will return verifier_id=None.
        gateway._workers.register(
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
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return gateway._workers.get("solo"), "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={
                "message": "q", "session_id": "ver-skip", "verify": True,
            },
        )
        assert resp.status_code == 200

        agg = client.get("/dispatch/recent?session=ver-skip").json()
        verify_entries = [
            d for d in agg["decisions"] if d.get("reason") == "verify"
        ]
        assert len(verify_entries) == 1, agg["decisions"]
        entry = verify_entries[0]
        assert "skipped" in entry["notes"], entry["notes"]
        assert "no alternate worker" in entry["notes"], entry["notes"]
        # The response itself shouldn't claim it was verified, AND
        # should surface verify_skipped so clients that don't read
        # the dispatch log know why their verify=true request did
        # nothing.
        body = resp.json()
        assert "verified_by" not in body
        assert body.get("verify_skipped") is True
        assert "no alternate worker" in body.get("verify_skip_reason", "")

    def test_ensemble_skip_reason_distinguishes_empty_text_from_timeouts(
        self, gateway, client,
    ):
        """The old ensemble_skip_reason had a single "all workers
        tool-looped" bucket for every non-empty contributions case —
        misleading when the actual failure was timeouts or mixed.
        Operators triaging a slow worker that never returned saw the
        same message as a flaky worker emitting tool calls, sending
        them down the wrong path.

        Three buckets now: all empty_text, all timeouts, or mixed."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "a", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_route(_msg, _sid):
            return None, "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_step(_conv, **_kwargs):
            return Message(role=Role.ASSISTANT, content="local fallback")
        gateway.agent.step = fake_step  # type: ignore[method-assign]

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
            gateway._ensemble_dispatch = fake_ensemble  # type: ignore[method-assign]

            resp = client.post(
                "/api/ask",
                json={
                    "message": "q",
                    "session_id": f"ens-skip-{expected_phrase.replace(' ', '_')}",
                    "ensemble": True,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body.get("ensemble_skipped") is True, body
            assert expected_phrase in body.get("ensemble_skip_reason", ""), (
                f"expected {expected_phrase!r} in skip_reason for "
                f"contributions={contributions}; got {body.get('ensemble_skip_reason')!r}"
            )
            # ensemble_contributions must be surfaced in the body
            # too — the success path always carries them; the skip
            # path used to drop the list, so callers couldn't see
            # which workers actually tried.
            assert "ensemble_contributions" in body, body
            assert len(body["ensemble_contributions"]) == len(contributions)

    def test_ensemble_all_workers_failed_notes_have_no_dangling_colon(
        self, gateway,
    ):
        """When every fan-out worker errors out, contributions is
        non-empty but `contributing` is empty. The earlier formatter
        produced "ensemble: none (0/2 answered: )" with a trailing
        colon and empty list — operators kept asking whether the
        notes were truncated. Now the empty trailing list is dropped."""
        decision = gateway._dispatcher.record_ensemble(
            session_id="all-failed",
            contributions=[
                {"worker_id": "a", "answer": "", "ms": 100.0, "error": "boom"},
                {"worker_id": "b", "answer": "", "ms": 100.0, "error": "boom"},
            ],
            arbitration_mode="none",
        )
        # The notes should report "0/2 answered" without a trailing
        # ":" or empty worker list.
        assert decision.notes == "ensemble: none (0/2 answered)"

    def test_verify_records_aggregate_dispatch_entry(self, gateway, client):
        """Symmetric to the ensemble case — verify pass records an
        aggregate 'verify' entry under the user's session_id."""
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
                role=Role.ASSISTANT, content="primary",
                metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return gateway._workers.get("primary"), "chat"

        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ver-dispatch", "verify": True},
        )
        assert resp.status_code == 200

        agg = client.get("/dispatch/recent?session=ver-dispatch").json()
        verify_entries = [d for d in agg["decisions"] if d.get("reason") == "verify"]
        assert len(verify_entries) == 1
        entry = verify_entries[0]
        assert entry["previous_worker_id"] == "primary"
        assert "confirmed" in entry["notes"]

    def test_ephemeral_sessions_hidden_by_default(self, gateway, client):
        # Mix user-facing and ephemeral session ids.
        self._record_decision(gateway, "user-session-1")
        self._record_decision(gateway, "_ens_user-session-1_worker-a_abc")
        self._record_decision(gateway, "_verify_user-session-1_xyz")
        self._record_decision(gateway, "_synth_ensemble")

        resp = client.get("/dispatch/recent")
        assert resp.status_code == 200
        ids = {d["session_id"] for d in resp.json()["decisions"]}
        # Only the user-facing one survives the default filter.
        assert "user-session-1" in ids
        assert all(not s.startswith(("_ens_", "_verify_", "_synth_")) for s in ids)

    def test_include_ephemeral_opt_in_surfaces_all(self, gateway, client):
        self._record_decision(gateway, "user-session-2")
        self._record_decision(gateway, "_ens_user-session-2_worker-a_abc")
        self._record_decision(gateway, "_verify_user-session-2_xyz")

        resp = client.get("/dispatch/recent?include_ephemeral=1")
        ids = {d["session_id"] for d in resp.json()["decisions"]}
        # All three present.
        assert "user-session-2" in ids
        assert "_ens_user-session-2_worker-a_abc" in ids
        assert "_verify_user-session-2_xyz" in ids


class TestDispatchExplain:
    """`/dispatch/explain` is a previewing endpoint — it shouldn't
    silently fall through bogus inputs (typo'd intent, negative
    tokens) because the whole point is to surface what the
    dispatcher would do."""

    def test_missing_session_id(self, client):
        resp = client.get("/dispatch/explain")
        assert resp.status_code == 400
        assert "session_id" in resp.json()["error"]

    def test_session_id_too_long_rejected(self, client):
        """Same 256-char cap /api/ask applies — without this, an
        absurd session_id flowed into the dispatcher decision and
        got echoed verbatim in the response and any access log."""
        resp = client.get("/dispatch/explain?session_id=" + "a" * 300)
        assert resp.status_code == 400
        assert "256" in resp.json()["error"]

    def test_session_id_with_control_chars_rejected(self, client):
        """Newlines / NUL / tab in session_id break log readability
        and would surface in the echoed JSON. Same rule as the rest
        of the session-id-accepting endpoints."""
        for encoded in ("%00", "a%0ab", "tab%09here"):
            resp = client.get(f"/dispatch/explain?session_id={encoded}")
            assert resp.status_code == 400, f"accepted {encoded!r}"
            assert "control" in resp.json()["error"].lower()

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

    def test_ask_max_tokens_invalid_rejected(self, client):
        """`max_tokens` must be a positive integer (≤ 4096). Garbage
        values fail loud at the boundary rather than crashing deep
        in the dispatch path. Same [1, 4096] clamp /v1/chat/completions
        uses so behaviour is consistent."""
        for bad in ("abc", [], {}, 0, -1):
            resp = client.post(
                "/api/ask",
                json={"message": "hi", "max_tokens": bad},
            )
            assert resp.status_code == 400, f"accepted {bad!r}"

    def test_ask_max_tokens_flows_to_quick_infer(self, gateway, client):
        """A valid `max_tokens` reaches `_quick_remote_infer` so the
        worker actually generates up to the requested length. Without
        this plumbing, /api/ask was silently hard-capped at 256 and
        clients had no way to raise the ceiling for longer answers."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "w", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_route(_msg, _sid):
            return gateway._workers.get("w"), "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        captured: dict = {}

        async def fake_quick(
            session_id, session, worker, max_tokens=256, **kwargs,
        ):
            captured["max_tokens"] = max_tokens
            msg = Message(
                role=Role.ASSISTANT, content="answer",
                metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "max-tok-ask", "max_tokens": 2048},
        )
        assert resp.status_code == 200
        assert captured["max_tokens"] == 2048

    def test_ask_temperature_invalid_rejected(self, client):
        """`temperature` must be a number in [0, 2]. Same range
        /v1/chat/completions documents. Non-numeric or out-of-range
        values fail loud at the boundary."""
        # Booleans aren't included because Python's float(True) = 1.0
        # silently coerces — matches /v1/chat/completions' lenient
        # parser. The range check still catches the truly bad cases.
        for bad in ("abc", [], {}, -0.1, 2.1):
            resp = client.post(
                "/api/ask",
                json={"message": "hi", "temperature": bad},
            )
            assert resp.status_code == 400, f"accepted {bad!r}"

    def test_ask_temperature_flows_to_quick_infer(self, gateway, client):
        """A valid `temperature` reaches `_quick_remote_infer` so the
        worker actually uses the requested sampling. Without this,
        /api/ask was hard-pinned at 0.7 — clients couldn't request
        deterministic (0.0) or more creative (1.5+) outputs."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "w", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_route(_msg, _sid):
            return gateway._workers.get("w"), "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        captured: dict = {}

        async def fake_quick(
            session_id, session, worker,
            max_tokens=256, temperature=0.7, **kwargs,
        ):
            captured["temperature"] = temperature
            msg = Message(
                role=Role.ASSISTANT, content="answer",
                metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # Deterministic generation.
        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "temp-0", "temperature": 0},
        )
        assert resp.status_code == 200
        assert captured["temperature"] == 0.0

        # Higher creativity.
        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "temp-1p5", "temperature": 1.5},
        )
        assert resp.status_code == 200
        assert captured["temperature"] == 1.5

    def test_ask_max_tokens_clamped_to_4096(self, gateway, client):
        """Requesting an absurd value should clamp to the same upper
        bound /v1/chat/completions uses rather than rejecting with
        400 — matches the OpenAI-compat path's lenient behaviour
        for over-large but otherwise-valid integers."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        gateway._workers.register(
            "w", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_route(_msg, _sid):
            return gateway._workers.get("w"), "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        captured: dict = {}

        async def fake_quick(
            session_id, session, worker, max_tokens=256, **kwargs,
        ):
            captured["max_tokens"] = max_tokens
            msg = Message(
                role=Role.ASSISTANT, content="answer",
                metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "max-tok-clamp", "max_tokens": 100000},
        )
        assert resp.status_code == 200
        assert captured["max_tokens"] == 4096

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

        # Stub the local agent's generate so synthesis returns a
        # known reconciled answer — proves the synthesis path was
        # reached and used, rather than the longest-answer fallback.
        # Synthesis uses generate() (not step()) so a tool-call from
        # the synthesizer can't trigger side effects.
        from towel.agent.runtime import GenerationResult

        async def fake_generate(conv, **kwargs):
            # The synthesis prompt should mention both workers' answers.
            user_msg = conv.messages[-1].content
            assert "Worker A answered:" in user_msg
            assert "Worker B answered:" in user_msg
            assert "Paris" in user_msg
            return GenerationResult(
                text="Synthesized: The capital of France is Paris.",
                tokens_per_second=10.0,
                total_tokens=8,
            )

        gateway.agent.generate = fake_generate  # type: ignore[method-assign]

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

    def test_ask_ensemble_synthesis_timeout_falls_back(
        self, gateway, client,
    ):
        """A wedged local-agent synthesis can't extend the ensemble
        run forever. With a tight timeout, hanging synthesis
        triggers the longest-fallback deterministic pick — and the
        response surfaces arbitration mode 'longest_fallback' so
        the operator can see the synthesis hop bailed."""
        import asyncio
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

        # Shorten the bound so the test doesn't take 90s.
        gateway.config.chat_fast_timeout = 0.1  # synthesis bound ~0.15s

        # Divergent answers → synthesis would normally fire.
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            answers = {
                "a": "Trees photosynthesize using sunlight.",
                "b": "Cars run on gasoline as fuel.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # Synthesis hangs forever — outer timeout must cancel it.
        async def fake_generate(conv, **kwargs):
            await asyncio.sleep(60)
            from towel.agent.runtime import GenerationResult
            return GenerationResult(text="never returns")

        gateway.agent.generate = fake_generate  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-synth-timeout", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Synthesis bailed → longest_fallback arbitration mode.
        assert body.get("ensemble_arbitration") == "longest_fallback"
        # One of the worker answers won (longest wins on tie).
        assert any(
            phrase in body["response"]
            for phrase in ("Trees photosynthesize", "Cars run on gasoline")
        )
        # Each contribution carries synthesis_timeout=True so the
        # operator can distinguish "synthesis bailed" from "synthesis
        # produced empty" — both surface as longest_fallback but only
        # the former has the timeout flag.
        for c in body["ensemble_contributions"]:
            if c.get("error") != "ensemble_timeout":
                assert c.get("synthesis_timeout") is True

    def test_ask_ensemble_concurrent_requests_dont_collide(
        self, gateway, client,
    ):
        """Ephemeral per-worker session ids must be unique per call.
        Two concurrent ensemble requests on the same user-facing
        session_id would otherwise share `_ens_{session_id}_{worker}`
        keys and interleave their conversations, corrupting the
        prompts each worker received. The fix is a uuid run_id
        suffix on every ephemeral session id."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._ensemble_dispatch)
        # The fix adds a run_id-based suffix to the sess_id.
        assert "run_id" in src
        assert "{run_id}" in src
        # Same fix landed in _verify_pass.
        src_v = inspect.getsource(GatewayServer._verify_pass)
        assert "verify_sess_id" in src_v
        assert "uuid" in src_v

    def test_ask_ensemble_shuffles_contributions_in_synthesis_prompt(
        self, gateway, client, monkeypatch,
    ):
        """LLM-as-judge has measurable primacy/recency bias. Workers
        arrive in completion order (fastest first), so a fast-but-
        shallow worker would consistently get the privileged 'Worker
        A' slot. Random ordering removes that bias from arbitration.
        Test by stubbing random.shuffle to reverse the list and
        confirming Worker A's answer is the LAST contribution from
        the un-shuffled candidate list."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role
        from towel.agent.runtime import GenerationResult

        gateway._workers.register(
            "fast", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "slow", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        # Divergent so synthesis fires.
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            answers = {
                "fast": "Quick fast-worker answer.",
                "slow": "Slow but thorough slow-worker answer.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # Capture the synthesis prompt so we can check ordering.
        captured_prompts: list[str] = []

        async def fake_generate(conv, **kwargs):
            captured_prompts.append(conv.messages[-1].content)
            return GenerationResult(text="Synthesized.")

        gateway.agent.generate = fake_generate  # type: ignore[method-assign]

        # Force a deterministic shuffle that REVERSES the list.
        import random
        original_shuffle = random.shuffle

        def reverse_shuffle(lst):
            lst.reverse()

        monkeypatch.setattr(random, "shuffle", reverse_shuffle)

        try:
            resp = client.post(
                "/api/ask",
                json={"message": "q", "session_id": "ens-shuf", "ensemble": True},
            )
        finally:
            monkeypatch.setattr(random, "shuffle", original_shuffle)

        assert resp.status_code == 200
        assert captured_prompts, "synthesis didn't run"
        prompt = captured_prompts[0]
        # With the reverse-shuffle, Worker A should be whichever
        # contribution was LAST in arrival order. Both contributions
        # appear in the prompt; we just verify the shuffle ran by
        # confirming the prompt contains both worker answers.
        assert "Quick fast-worker answer" in prompt
        assert "Slow but thorough slow-worker answer" in prompt
        # Find positions of the two answer strings.
        fast_pos = prompt.find("Quick fast-worker answer")
        slow_pos = prompt.find("Slow but thorough slow-worker answer")
        # In normal completion order, "fast" arrived first → would be
        # Worker A → appears earlier. With reverse-shuffle, "slow"
        # should appear first.
        assert slow_pos < fast_pos, (
            f"shuffle didn't fire: fast={fast_pos} slow={slow_pos}"
        )

    def test_ask_ensemble_synthesis_timing_in_contributions(
        self, gateway, client,
    ):
        """When synthesis runs, every contribution gets a
        `synthesis_ms` field so the caller can see how long the
        local-agent arbitration took alongside the per-worker
        timings. Operators investigating slow ensemble runs need
        this to attribute latency."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role
        from towel.agent.runtime import GenerationResult

        gateway._workers.register(
            "a", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "b", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        # Divergent answers → synthesis path fires.
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            answers = {
                "a": "Trees photosynthesize using sunlight.",
                "b": "Cars use gasoline as fuel.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # Stub generate so synthesis returns a known answer quickly.
        async def fake_generate(conv, **kwargs):
            return GenerationResult(
                text="Synthesized answer.",
                tokens_per_second=10.0, total_tokens=4,
            )

        gateway.agent.generate = fake_generate  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-time", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("ensemble_arbitration") == "synthesis"
        # Every contribution carries the synthesis_ms timing.
        for c in body["ensemble_contributions"]:
            assert "synthesis_ms" in c
            assert isinstance(c["synthesis_ms"], (int, float))
            assert c["synthesis_ms"] >= 0

    def test_verify_pass_uses_low_temperature(self, gateway, client):
        """Symmetric to the ensemble-synthesis temperature override:
        the verifier should be decisive (return VERIFIED or a
        corrected answer), not creative. Low temperature reduces
        rerun-to-rerun drift."""
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

        captured: dict = {}
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            # Stash the verifier-call's temperature for assertion.
            if session_id.startswith("_verify_"):
                captured["verifier_temp"] = kwargs.get("temperature")
                return Message(
                    role=Role.ASSISTANT, content="VERIFIED",
                    metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
                )
            msg = Message(
                role=Role.ASSISTANT, content="primary answer",
                metadata={"remote_worker": worker.id, "tokens": 2, "tps": 5.0},
            )
            session.conversation.messages.append(msg)
            return msg
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        async def fake_route(_msg, _sid):
            return gateway._workers.get("primary"), "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ver-temp", "verify": True},
        )
        assert resp.status_code == 200
        # Verifier ran with the deterministic-ish temperature.
        assert captured["verifier_temp"] == 0.2

    def test_ensemble_synthesis_uses_low_temperature(self, gateway, client):
        """Arbitration should be deterministic-ish, not creative —
        the goal is "pick the best answer and copy it", not
        "rewrite both in the arbiter's voice." The synthesizer
        is invoked with a low temperature (0.2) override so the
        arbitration doesn't drift across identical inputs."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role
        from towel.agent.runtime import GenerationResult

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
                "a": "Apple is a fruit.",
                "b": "Cars use gasoline.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        captured: dict = {}
        async def fake_generate(conv, **kwargs):
            captured["temperature"] = kwargs.get("temperature")
            return GenerationResult(text="Synthesized.", total_tokens=2)
        gateway.agent.generate = fake_generate  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-temp", "ensemble": True},
        )
        assert resp.status_code == 200
        # Synthesis ran with the deterministic-ish temperature.
        assert captured["temperature"] == 0.2

    def test_ensemble_synthesis_handles_none_text_gracefully(
        self, gateway, client,
    ):
        """A buggy local agent could return GenerationResult(text=None)
        even though the field is typed `str`. Before the defensive
        coerce, parse_tool_calls(None) crashed and the outer except
        caught it ~90s later (full synth_timeout). Now the coerce
        treats None as empty text → falls through to
        longest_fallback immediately."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role
        from towel.agent.runtime import GenerationResult

        gateway._workers.register(
            "a", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "b", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            # Divergent answers to force the synthesis path.
            answers = {
                "a": "Apple is a fruit.",
                "b": "Cars use gasoline as fuel.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 5, "tps": 5.0},
            )
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        # Synthesizer returns a result with None text (simulates a
        # buggy backend).
        async def fake_generate_none(conv):
            return GenerationResult(text=None, tokens_per_second=0.0, total_tokens=0)  # type: ignore[arg-type]
        gateway.agent.generate = fake_generate_none  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-none", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Synthesis returned no usable text → longest_fallback path.
        # The response is one of the two real answers (whichever was
        # longer, in this case b).
        assert body["ensemble_arbitration"] == "longest_fallback"
        assert body["response"] in (
            "Apple is a fruit.",
            "Cars use gasoline as fuel.",
        )

    def test_ask_ensemble_arbitration_mode_surfaces_in_response(
        self, gateway, client,
    ):
        """The response body should expose WHICH arbitration path
        fired (synthesis / consensus / single / longest_fallback /
        none) so callers can tell whether their answer came from a
        real merge or a deterministic fallback. Operators
        investigating quality regressions need this signal."""
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

        # Near-identical answers → consensus path.
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            answers = {
                "a": "Paris is the capital of France and largest city.",
                "b": "Paris is the capital of France and its largest city.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 10, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-arbmode", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Consensus path fired (no synthesis).
        assert body.get("ensemble_arbitration") == "consensus"

    def test_ask_ensemble_single_worker_arbitration_mode(self, gateway, client):
        """When only one inference worker is available, ensemble
        fans out to that single worker and returns its answer with
        arbitration_mode="single". Clients can distinguish this
        from a real multi-worker merge — arbitration="single" means
        the fleet didn't have a quorum, only one worker contributed.
        Important signal: operators benchmarking expect to see this
        when one worker is busy and ensemble degraded to single."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role

        # Only one inference worker registered.
        gateway._workers.register(
            "solo", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            return Message(
                role=Role.ASSISTANT, content="The answer is 42.",
                metadata={"remote_worker": worker.id, "tokens": 4, "tps": 5.0},
            )
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-single", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Single-worker arbitration. The answer is the worker's
        # raw output (no synthesis, no consensus pick).
        assert body.get("ensemble_arbitration") == "single"
        assert body["response"] == "The answer is 42."
        # Contributions list has exactly one entry.
        assert len(body["ensemble_contributions"]) == 1

    def test_ask_ensemble_trivial_agreement_skips_synthesis(
        self, gateway, client,
    ):
        """For very short answers ("42", "Berlin."), the Jaccard
        consensus check sees empty token sets (after the ≥3-char
        filter) and won't fire. The trivial-agreement short-circuit
        catches that case: if every answer is the same string
        (case-folded, stripped), there's nothing to arbitrate."""
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

        # Identical short answers — Jaccard would see empty sets
        # after the 3-char filter and not declare consensus.
        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            return Message(
                role=Role.ASSISTANT, content="42",
                metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        synth_called = []
        async def fake_step(conv):
            synth_called.append(True)
            return Message(role=Role.ASSISTANT, content="should not appear")

        gateway.agent.step = fake_step  # type: ignore[method-assign]

        async def fake_gen(conv, **kwargs):
            synth_called.append(True)
            from towel.agent.runtime import GenerationResult
            return GenerationResult(text="should not appear")

        gateway.agent.generate = fake_gen  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "answer?", "session_id": "ens-trivial", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Synthesis must NOT have run.
        assert synth_called == [], "synthesis ran on trivial-agreement"
        assert body["response"] == "42"
        assert body["ensemble_arbitration"] == "consensus"

    def test_ask_ensemble_trivial_agreement_strips_trailing_punctuation(
        self, gateway, client,
    ):
        """Trailing-punctuation differences ("42." vs "42", "Berlin!"
        vs "Berlin") should still hit the trivial-agreement short-
        circuit. Without this, Jaccard sees empty token sets (3-char
        filter dropped both) and the run wasted ~30s on synthesis to
        reconcile cosmetically-different but factually-identical
        answers."""
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
            # One worker punctuates, the other doesn't. Same factual
            # answer — synthesis must be skipped.
            answers = {"a": "42.", "b": "42"}
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 1, "tps": 5.0},
            )
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        synth_called = []
        async def fake_gen(conv, **kwargs):
            synth_called.append(True)
            from towel.agent.runtime import GenerationResult
            return GenerationResult(text="should not appear")
        gateway.agent.generate = fake_gen  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={
                "message": "q", "session_id": "ens-trivial-punct",
                "ensemble": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert synth_called == [], (
            "synthesis ran on punctuation-only diff"
        )
        assert body["ensemble_arbitration"] == "consensus"
        # The returned answer is one of the workers' raw answers
        # (whichever was first in the real_answers list). Both are
        # acceptable since they're factually identical.
        assert body["response"] in {"42.", "42"}

    def test_ask_ensemble_consensus_skips_synthesis(self, gateway, client):
        """When workers basically agree (Jaccard ≥ 0.7 on
        lowercased word tokens), skip the ~30s local-agent synthesis
        call and return the longest answer directly. Synthesis adds
        no value when the workers already converged."""
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
            # Nearly-identical answers — high Jaccard overlap.
            answers = {
                "a": "The capital of France is Paris and Paris is famous for the Eiffel Tower.",
                "b": "The capital of France is Paris; Paris is also famous for the Eiffel Tower.",
            }
            return Message(
                role=Role.ASSISTANT, content=answers[worker.id],
                metadata={"remote_worker": worker.id, "tokens": 14, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        synth_called = []
        async def fake_step(conv):
            synth_called.append(True)
            return Message(role=Role.ASSISTANT, content="should not appear")

        gateway.agent.step = fake_step  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-cons", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Synthesis must NOT have run — workers agreed.
        assert synth_called == [], "synthesis ran on near-consensus answers"
        # The answer comes from one of the workers, not the synthesizer.
        assert "Paris" in body["response"]

    def test_ask_ensemble_skips_classifier_only_workers(
        self, gateway, client,
    ):
        """A classifier-only worker isn't sized for substantive
        answers — including it in ensemble wastes its compute and
        pollutes the arbiter with low-effort responses. Workers with
        no INFERENCE/GENERAL role must be filtered from the fan-out
        pool. Workers with no role info at all stay eligible (covers
        test fixtures and freshly-registered workers)."""
        from unittest.mock import MagicMock

        from towel.agent.conversation import Message, Role
        from towel.nodes.roles import NodeRole

        gateway._workers.register(
            "inference-worker", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "classifier-only", MagicMock(),
            {"backend": "llama", "modes": ["llama_chat"]},
        )
        # Assign roles explicitly.
        gateway._node_roles["inference-worker"] = [NodeRole.INFERENCE]
        gateway._node_roles["classifier-only"] = [NodeRole.CLASSIFIER]

        seen_workers: list[str] = []

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            seen_workers.append(worker.id)
            return Message(
                role=Role.ASSISTANT, content=f"from {worker.id}",
                metadata={"remote_worker": worker.id, "tokens": 3, "tps": 5.0},
            )

        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "q", "session_id": "ens-roles", "ensemble": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Only the inference worker contributed.
        assert "inference-worker" in seen_workers
        assert "classifier-only" not in seen_workers
        # And the response carries that one contribution.
        ids = {c["worker_id"] for c in body["ensemble_contributions"]}
        assert ids == {"inference-worker"}

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

    def test_ask_verify_strict_match_catches_just_the_token(
        self, gateway, client,
    ):
        """The strict-confirmation layer (strip non-word, compare
        to "VERIFIED") catches every cosmetic variant of just the
        token regardless of length: whitespace, markdown bullets,
        underscores, mixed case. Different from the lenient fallback
        which gates on length."""
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

        # All these normalise to "VERIFIED" exactly. The strict
        # layer should accept each one as a confirmation without
        # falling back to the length-gated lenient path.
        for verifier_output in (
            "  VERIFIED  ",            # surrounded by whitespace
            "**VERIFIED**",            # markdown bold
            "***\nVERIFIED\n***",      # markdown thematic break
            "verified.",               # lowercase with punctuation
            "Verified",                # title case
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

            sid = f"v-strict-{abs(hash(verifier_output))}"
            resp = client.post(
                "/api/ask",
                json={"message": "q", "session_id": sid, "verify": True},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["response"] == "primary answer", (
                f"strict-match cosmetic variant {verifier_output!r} "
                f"did not confirm primary; got {body['response']!r}"
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


    def test_verify_skipped_reason_when_primary_empty(self, gateway, client):
        """When verify=true is set but the primary itself returned
        an empty-text placeholder, the verify pass is skipped
        because there's nothing to verify — distinct from the
        "no alternate worker" reason. Surface the actual cause."""
        from unittest.mock import AsyncMock

        from towel.agent.conversation import Message, Role
        from towel.gateway.workers import WorkerInfo

        primary = WorkerInfo(id="empty-primary", ws=AsyncMock(), capabilities={})
        alt = WorkerInfo(
            id="empty-alt", ws=AsyncMock(),
            capabilities={"total_vram_mb": 16000},
        )
        gateway._workers._workers["empty-primary"] = primary
        gateway._workers._workers["empty-alt"] = alt

        async def fake_route(message, session_id):
            return primary, "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            msg = Message(
                role=Role.ASSISTANT,
                content="(placeholder)",
                metadata={
                    "remote_worker": worker.id,
                    "empty_text_fallback": True,
                },
            )
            session.conversation.messages.append(msg)
            return msg
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "ver-empty", "verify": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Verify was opted in but the primary had nothing to verify.
        assert body.get("verify_skipped") is True
        assert "nothing to verify" in body.get("verify_skip_reason", "")

    def test_dual_empty_text_surfaces_in_api_ask_response(self, gateway, client):
        """When BOTH the primary and the retry alt return empty text
        (the fleet-wide tool-loop case), the metadata flag
        `dual_empty_text` was set but never surfaced into the
        /api/ask response body. Clients saw the same diagnostic
        placeholder they'd get from a single-worker empty response
        and had no way to tell the system actually tried twice.

        Reproducer for the live-coordinator behavior observed in
        2026-05: a "Hi" prompt produced an empty-text placeholder
        after both Gemma and SparklesMint tool-looped; the dispatch
        log showed two attempts but the response body looked single-
        worker."""
        from unittest.mock import AsyncMock

        from towel.agent.conversation import Message, Role
        from towel.gateway.workers import WorkerInfo

        primary = WorkerInfo(id="dual-primary", ws=AsyncMock(), capabilities={})
        alt = WorkerInfo(
            id="dual-alt", ws=AsyncMock(),
            capabilities={"total_vram_mb": 16000},
        )
        gateway._workers._workers["dual-primary"] = primary
        gateway._workers._workers["dual-alt"] = alt

        async def fake_route(message, session_id):
            return primary, "chat"
        gateway._route_by_role = fake_route  # type: ignore[method-assign]

        async def fake_quick(session_id, session, worker, max_tokens=256, **kwargs):
            # Both workers return the empty-text fallback shape.
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
        gateway._quick_remote_infer = fake_quick  # type: ignore[method-assign]

        resp = client.post(
            "/api/ask",
            json={"message": "hi", "session_id": "dual-empty"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The body must signal that the fleet hit a dual-empty case
        # so clients can render "Try rephrasing — both workers tool-
        # looped" instead of the generic placeholder.
        assert body.get("dual_empty_text") is True, body
        assert body.get("alt_worker") == "dual-alt", body


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

    def test_api_sessions_channel_filter(self, store, client):
        """Same `?channel=` semantics /conversations exposes, mirrored
        here so an operator can switch between the two list endpoints
        without losing the filter UX."""
        for sid, ch in (("api-a", "api"), ("cli-a", "cli"), ("api-b", "api")):
            conv = Conversation(id=sid, channel=ch)
            conv.add(Role.USER, "hi")
            store.save(conv)

        resp = client.get("/api/sessions?channel=cli")
        assert resp.status_code == 200
        ids = {s["id"] for s in resp.json()["sessions"]}
        assert ids == {"cli-a"}

    def test_api_sessions_tag_filter(self, store, client):
        """Same `?tag=` semantics /conversations exposes."""
        for sid, tags in (("a", ["x"]), ("b", ["y"]), ("c", ["x", "y"])):
            conv = Conversation(id=sid, channel="api")
            conv.tags = tags
            conv.add(Role.USER, "hi")
            store.save(conv)

        resp = client.get("/api/sessions?tag=y")
        ids = {s["id"] for s in resp.json()["sessions"]}
        assert ids == {"b", "c"}

    def test_api_sessions_includes_worker_routing_state(
        self, gateway, store, client,
    ):
        """/api/sessions previously omitted worker_id and
        pinned_worker_id — operators triaging "where is this session
        pinned?" had to cross-reference with /sessions which only
        carries live in-memory entries. Now both fields ride
        alongside title/summary/tags on every entry. Sessions with
        no routing state get None for both, keeping the response
        shape uniform."""
        from unittest.mock import MagicMock

        # Two persisted sessions; one is pinned, one routed via
        # affinity, one has neither.
        for sid in ("pinned-conv", "routed-conv", "plain-conv"):
            conv = Conversation(id=sid, channel="api")
            conv.add(Role.USER, "hi")
            store.save(conv)

        gateway._workers.register(
            "alpha", MagicMock(), {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._workers.register(
            "beta", MagicMock(), {"backend": "llama", "modes": ["llama_chat"]},
        )
        gateway._session_pins["pinned-conv"] = "alpha"
        gateway._session_workers["routed-conv"] = "beta"

        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        by_id = {s["id"]: s for s in resp.json()["sessions"]}

        assert by_id["pinned-conv"]["pinned_worker_id"] == "alpha"
        assert by_id["pinned-conv"]["worker_id"] is None
        assert by_id["routed-conv"]["worker_id"] == "beta"
        assert by_id["routed-conv"]["pinned_worker_id"] is None
        assert by_id["plain-conv"]["worker_id"] is None
        assert by_id["plain-conv"]["pinned_worker_id"] is None


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

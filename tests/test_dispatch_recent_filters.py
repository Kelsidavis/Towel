"""Filters on /dispatch/recent so operators can ask narrow questions of
the decision log without dumping every entry to the console.

The endpoint pre-filters before the limit so tight limits don't hide
matches buried earlier in the ring buffer.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from towel.config import TowelConfig
from towel.gateway.dispatcher import DispatchDecision
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore


class _FakeAgent:
    pass


@pytest.fixture
def gateway(tmp_path):
    store = ConversationStore(store_dir=tmp_path)
    return GatewayServer(
        config=TowelConfig(),
        agent=_FakeAgent(),
        sessions=SessionManager(store=store),
        pin_store=SessionPinStore(path=tmp_path / "pins.json"),
        worker_state_store=WorkerStateStore(path=tmp_path / "worker_state.json"),
    )


def _stub_worker(worker_id: str):
    """Minimal stand-in for WorkerInfo — DispatchDecision.to_dict only
    reads ``.id`` off of it."""
    return SimpleNamespace(id=worker_id)


def _seed_decisions(gateway: GatewayServer) -> None:
    """Push a known mix of decisions into the dispatcher's history buffer
    so the filter tests have something to chew on. The shape mirrors what
    a real ``select_for_session`` would produce."""
    assert gateway._dispatcher is not None
    decisions = [
        DispatchDecision(
            worker=_stub_worker("gpu-host"),
            intent="chat",
            reason="pin",
            session_id="sess-1",
            candidates_considered=1,
        ),
        DispatchDecision(
            worker=_stub_worker("pi-host"),
            intent="chat",
            reason="capability_fallback",
            session_id="sess-2",
            candidates_considered=3,
            quality_degraded=True,
        ),
        DispatchDecision(
            worker=_stub_worker("gpu-host"),
            intent="chat",
            reason="affinity",
            session_id="sess-1",
            candidates_considered=2,
        ),
        DispatchDecision(
            worker=_stub_worker("other-host"),
            intent="task",
            reason="task_match",
            session_id="sess-3",
            candidates_considered=4,
            affinity_missed=True,
            previous_worker_id="gpu-host",
        ),
    ]
    for d in decisions:
        gateway._dispatcher._record(d)


class TestDispatchRecentFilters:
    def test_no_filters_returns_all_recent(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?limit=10").json()
        assert len(resp["decisions"]) == 4
        assert resp["total_matching"] == 4

    def test_reason_filter(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?reason=affinity").json()
        assert len(resp["decisions"]) == 1
        assert resp["decisions"][0]["reason"] == "affinity"

    def test_worker_filter(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?worker=gpu-host").json()
        assert len(resp["decisions"]) == 2
        assert {d["worker_id"] for d in resp["decisions"]} == {"gpu-host"}

    def test_session_filter(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?session=sess-1").json()
        assert len(resp["decisions"]) == 2
        for d in resp["decisions"]:
            assert d["session_id"] == "sess-1"

    def test_previous_worker_filter(self, gateway):
        """`previous_worker=X` complements `worker=X`: where the
        latter surfaces decisions that PICKED X, this one surfaces
        decisions that BYPASSED or REPLACED X (retry_empty_text rows
        whose primary was X, affinity_missed rows whose previous
        host was X, pin_missed rows whose pin pointed at X).

        Operators triaging "why does my fleet keep retrying off
        worker X?" had to eyeball every entry without this filter —
        the previous_worker_id field was already in the response,
        just not filterable."""
        assert gateway._dispatcher is not None
        # Three retry decisions: two from gpu-host as primary, one
        # from pi-host. The chosen worker (alt) is the same in both
        # cases — the question is "which primary did we fall off of?"
        from towel.gateway.dispatcher import REASON_RETRY_EMPTY
        for prev in ("gpu-host", "gpu-host", "pi-host"):
            d = DispatchDecision(
                worker=_stub_worker("alt"),
                intent="chat",
                reason=REASON_RETRY_EMPTY,
                notes=f"retry after empty response from {prev}",
                session_id=f"sess-{prev}",
                candidates_considered=1,
                previous_worker_id=prev,
            )
            gateway._dispatcher._record(d)

        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?previous_worker=gpu-host").json()
        assert len(resp["decisions"]) == 2
        for d in resp["decisions"]:
            assert d["previous_worker_id"] == "gpu-host"
            assert d["worker_id"] == "alt"

        resp = client.get("/dispatch/recent?previous_worker=pi-host").json()
        assert len(resp["decisions"]) == 1
        assert resp["decisions"][0]["previous_worker_id"] == "pi-host"

        # Unknown previous_worker matches nothing — no false positives.
        resp = client.get("/dispatch/recent?previous_worker=ghost-host").json()
        assert resp["decisions"] == []

    def test_previous_worker_length_cap(self, gateway):
        """Same 256-char cap as the other string filters — bogus
        100KB inputs would otherwise bloat the request line in any
        access log even though they match nothing."""
        client = TestClient(gateway._build_http_app())
        resp = client.get(
            "/dispatch/recent?previous_worker=" + "x" * 257
        )
        assert resp.status_code == 400
        assert "256 chars" in resp.json().get("error", "")

    def test_only_degraded(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?only_degraded=1").json()
        assert len(resp["decisions"]) == 1
        assert resp["decisions"][0]["quality_degraded"] is True

    def test_only_affinity_missed(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?only_affinity_missed=1").json()
        assert len(resp["decisions"]) == 1
        assert resp["decisions"][0]["affinity_missed"] is True

    def test_only_pin_missed(self, gateway):
        """The pin_missed filter surfaces decisions where the session
        had an explicit pin that was silently bypassed (busy /
        draining / disabled worker at dispatch time). Without this an
        operator had no fast way to ask "where are my pins being
        ignored?"; the bypassed decision looked like a normal route."""
        assert gateway._dispatcher is not None
        # Two decisions: one with a bypassed pin, one without.
        gateway._dispatcher._record(
            DispatchDecision(
                worker=_stub_worker("free-host"),
                intent="chat",
                reason="role_match",
                session_id="sess-pin",
                candidates_considered=1,
                pin_missed=True,
                pinned_worker_id="busy-host",
            )
        )
        gateway._dispatcher._record(
            DispatchDecision(
                worker=_stub_worker("other-host"),
                intent="chat",
                reason="role_match",
                session_id="sess-no-pin",
                candidates_considered=1,
            )
        )
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?only_pin_missed=1").json()
        assert len(resp["decisions"]) == 1
        entry = resp["decisions"][0]
        assert entry["pin_missed"] is True
        assert entry["pinned_worker_id"] == "busy-host"
        assert entry["session_id"] == "sess-pin"

    def test_min_total_ms_filters_slow_requests(self, gateway):
        """Operators triaging slow / timed-out dispatches previously
        had to eyeball every entry's total_ms by hand. The
        `min_total_ms` filter lets them ask "show me everything that
        took > 60s" in one query — pairs naturally with the
        worker_inference_timeout (300s) and chat_fast_timeout (60s)
        bounds so timeout-class entries surface immediately."""
        assert gateway._dispatcher is not None
        # Three decisions: fast, slow-but-under-threshold, slow.
        for sid, ms in (("fast", 100.0), ("med", 5000.0), ("slow", 300000.0)):
            d = DispatchDecision(
                worker=_stub_worker("w"),
                intent="chat",
                reason="role_match",
                session_id=sid,
                candidates_considered=1,
            )
            d.total_ms = ms
            gateway._dispatcher._record(d)
        # Also a decision with no total_ms (in-flight) — must be
        # excluded from the >=N filter.
        gateway._dispatcher._record(
            DispatchDecision(
                worker=_stub_worker("w"),
                intent="chat",
                reason="role_match",
                session_id="no-timing",
                candidates_considered=1,
            )
        )

        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?min_total_ms=60000").json()
        sids = {d["session_id"] for d in resp["decisions"]}
        assert sids == {"slow"}, sids

        # A lower threshold catches both slow and medium.
        resp = client.get("/dispatch/recent?min_total_ms=1000").json()
        sids = {d["session_id"] for d in resp["decisions"]}
        assert sids == {"med", "slow"}, sids

    def test_min_total_ms_rejects_non_numeric(self, gateway):
        """Bogus input should error fast rather than silently fall
        through (same defensive shape as `limit=abc`)."""
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?min_total_ms=abc")
        assert resp.status_code == 400

    def test_min_total_ms_rejects_negative(self, gateway):
        """Negative thresholds make no sense — surface as 400 instead
        of accepting them and silently matching everything."""
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?min_total_ms=-1")
        assert resp.status_code == 400

    def test_combined_filters(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        # gpu-host AND session sess-1 — both decisions match.
        resp = client.get(
            "/dispatch/recent?worker=gpu-host&session=sess-1"
        ).json()
        assert len(resp["decisions"]) == 2
        # gpu-host AND a reason it never had — empty.
        resp = client.get(
            "/dispatch/recent?worker=gpu-host&reason=capability_fallback"
        ).json()
        assert resp["decisions"] == []
        assert resp["total_matching"] == 0

    def test_filter_applies_before_limit(self, gateway):
        _seed_decisions(gateway)
        client = TestClient(gateway._build_http_app())
        # Even with limit=1, the gpu-host filter should still see both
        # gpu-host decisions in total_matching even though only 1 is returned.
        resp = client.get("/dispatch/recent?worker=gpu-host&limit=1").json()
        assert len(resp["decisions"]) == 1
        assert resp["total_matching"] == 2

    def test_invalid_limit_returns_400(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?limit=abc")
        assert resp.status_code == 400

    def test_intent_filter(self, gateway):
        """Filter dispatch decisions by intent class so operators can
        ask "show me only chat traffic" or "only tool dispatches"
        without grepping the response client-side. Matches the
        existing chat/tool/task split used by /dispatch/explain."""
        assert gateway._dispatcher is not None
        for sid, intent in (
            ("chat-1", "chat"), ("chat-2", "chat"),
            ("task-1", "task"), ("tool-1", "tool"),
        ):
            gateway._dispatcher._record(
                DispatchDecision(
                    worker=_stub_worker("w"),
                    intent=intent,
                    reason="role_match",
                    session_id=sid,
                    candidates_considered=1,
                )
            )
        client = TestClient(gateway._build_http_app())

        chat = client.get("/dispatch/recent?intent=chat").json()
        assert {d["session_id"] for d in chat["decisions"]} == {"chat-1", "chat-2"}

        tool = client.get("/dispatch/recent?intent=tool").json()
        assert {d["session_id"] for d in tool["decisions"]} == {"tool-1"}

    def test_filter_lengths_capped(self, gateway):
        """String filters (`reason`, `worker`, `session`) cap at
        256 chars to keep absurd inputs out of access logs. Each
        rejects with a clear 400."""
        client = TestClient(gateway._build_http_app())
        long = "a" * 300
        for f in ("reason", "worker", "session"):
            resp = client.get(f"/dispatch/recent?{f}={long}")
            assert resp.status_code == 400, f"accepted long {f}"
            assert "256 chars" in resp.json()["error"]

    def test_intent_filter_rejects_unknown(self, gateway):
        """A typo like ?intent=tools would otherwise silently match
        nothing and look like an empty log — fail fast with 400."""
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent?intent=tools")
        assert resp.status_code == 400

    def test_log_status_includes_oldest_age_seconds(self, gateway):
        """`log_status.oldest_age_seconds` is meant to drive the UI's
        log-freshness indicator, but the original code looked for an
        `ts` field that to_dict() never emits — the actual field is
        `timestamp` (Unix float). datetime.fromisoformat("") raised
        ValueError, the bare except swallowed it, and the field was
        silently missing from every payload.

        With a few recorded decisions, the freshness number should
        now appear and be non-negative."""
        import time
        # Two decisions: one a few seconds old, one fresh.
        old_decision = DispatchDecision(
            worker=_stub_worker("w"),
            intent="chat",
            reason="role_match",
            session_id="old",
            candidates_considered=1,
        )
        old_decision.timestamp = time.time() - 5.0
        gateway._dispatcher._record(old_decision)
        gateway._dispatcher._record(
            DispatchDecision(
                worker=_stub_worker("w"),
                intent="chat",
                reason="role_match",
                session_id="fresh",
                candidates_considered=1,
            )
        )

        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent").json()
        assert "oldest_age_seconds" in resp["log_status"], resp["log_status"]
        # Sanity: oldest_age_seconds is non-negative and at least
        # a couple of seconds for the 5s-old entry.
        assert resp["log_status"]["oldest_age_seconds"] >= 0
        assert resp["log_status"]["oldest_age_seconds"] >= 4

    def test_log_status_includes_empty_text_retries_by_worker(self, gateway):
        """A flaky chat worker can drive empty-text retries every
        time it gets routed — costing the user the full primary
        latency before the retry runs. Operators viewing
        /dispatch/recent need the per-primary tally surfaced
        directly so "worker X had N empty-text retries" is visible
        without grepping every entry by hand."""
        # Always-present key (even with zero retries) so the UI
        # doesn't have to special-case missing data.
        client = TestClient(gateway._build_http_app())
        resp = client.get("/dispatch/recent").json()
        assert resp["log_status"]["empty_text_retries_by_worker"] == {}

        # Push three empty-text retries (two from gpu-host, one
        # from pi-host) into the buffer. The previous_worker_id
        # surfaces in the tally; the alt worker (newly-picked) does
        # not.
        from towel.gateway.dispatcher import REASON_RETRY_EMPTY
        for prev in ("gpu-host", "gpu-host", "pi-host"):
            d = DispatchDecision(
                worker=_stub_worker("alt"),
                intent="chat",
                reason=REASON_RETRY_EMPTY,
                notes=f"retry after empty response from {prev}",
                session_id=f"sess-{prev}",
                candidates_considered=1,
                previous_worker_id=prev,
            )
            gateway._dispatcher._record(d)

        resp = client.get("/dispatch/recent").json()
        assert resp["log_status"]["empty_text_retries_by_worker"] == {
            "gpu-host": 2, "pi-host": 1,
        }

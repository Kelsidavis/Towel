"""Coverage for the standalone /v1/stream SSE handlers.

The handlers live as closures inside ``build_sse_routes`` and don't
have a clean injection point — so these tests rely on source
inspection to confirm the event-dispatch chain covers every event
type the agent runtime emits. The behaviour is exercised end-to-end
by the existing test_streaming_protocol-adjacent tests on the gateway
side; this file locks in the contract that ERROR / CANCELLED events
make it into the SSE output (not silently dropped).
"""

from __future__ import annotations

import inspect

from towel.agent.streaming_protocol import build_sse_routes


def _stream_post_source() -> str:
    """Find the ``stream_post`` closure's source via the routes."""
    # The handlers are closures returned by build_sse_routes — we can
    # reach them through the Route objects' .endpoint attribute.
    from unittest.mock import MagicMock

    agent = MagicMock()
    config = MagicMock()
    routes = build_sse_routes(agent, config)
    for r in routes:
        if r.path == "/v1/stream" and "POST" in (r.methods or set()):
            return inspect.getsource(r.endpoint)
    raise AssertionError("POST /v1/stream route not found")


def _stream_get_source() -> str:
    from unittest.mock import MagicMock

    agent = MagicMock()
    config = MagicMock()
    routes = build_sse_routes(agent, config)
    for r in routes:
        if r.path == "/v1/stream" and "GET" in (r.methods or set()):
            return inspect.getsource(r.endpoint)
    raise AssertionError("GET /v1/stream route not found")


class TestStreamPostEventCoverage:
    """The POST /v1/stream handler must cover every event type
    the agent runtime emits, not just the happy path. A missing
    branch silently drops the event — the client sees [DONE]
    with no failure / cancel signal."""

    def test_handles_token_events(self):
        src = _stream_post_source()
        assert "EventType.TOKEN" in src

    def test_handles_response_complete_events(self):
        src = _stream_post_source()
        assert "EventType.RESPONSE_COMPLETE" in src

    def test_handles_error_events(self):
        """ERROR events used to drop silently — the stream ended
        via [DONE] with no error signal. Confirm the branch exists
        and emits an `error` SSE payload."""
        src = _stream_post_source()
        assert "EventType.ERROR" in src
        assert "'type': 'error'" in src or "\"type\": \"error\"" in src

    def test_handles_cancelled_events(self):
        """Symmetric to ERROR — cancellations need a visible signal
        instead of a bare [DONE]."""
        src = _stream_post_source()
        assert "EventType.CANCELLED" in src
        assert "'type': 'cancelled'" in src or "\"type\": \"cancelled\"" in src


class TestSseErrorTermination:
    """SSE error responses must include [DONE] so clients don't hang."""

    def test_get_error_includes_done(self):
        src = _stream_get_source()
        assert "[DONE]" in src

    def test_post_error_includes_done(self):
        src = _stream_post_source()
        assert "[DONE]" in src

    def test_get_uses_safe_data_access(self):
        """event.data access must use .get() to avoid KeyError on
        malformed events."""
        src = _stream_get_source()
        assert "event.data.get(" in src

    def test_post_uses_safe_data_access(self):
        src = _stream_post_source()
        assert "event.data.get(" in src


class TestStreamGetEventCoverage:
    """The GET /v1/stream handler also needs the same coverage —
    EventSource clients reading from /v1/stream?prompt=... need the
    same failure/cancel signals POST clients get, otherwise a
    cancelled run looks like a normal finish to the GET caller."""

    def test_handles_token_events(self):
        src = _stream_get_source()
        assert "EventType.TOKEN" in src

    def test_handles_error_events(self):
        src = _stream_get_source()
        assert "EventType.ERROR" in src

    def test_handles_cancelled_events(self):
        """The GET handler used to drop CANCELLED silently — the
        SSE stream just ended at [DONE] when a user cancelled mid-
        generation. EventSource clients keying on
        `type === 'cancelled'` to clear loading indicators were
        getting no signal. Confirm the branch and payload exist
        for parity with POST."""
        src = _stream_get_source()
        assert "EventType.CANCELLED" in src
        assert "'type': 'cancelled'" in src or "\"type\": \"cancelled\"" in src

    def test_handles_response_complete_events(self):
        """RESPONSE_COMPLETE is the normal success terminator —
        without the branch the GET handler would never emit the
        `done` payload that EventSource clients use to flush the
        accumulated content."""
        src = _stream_get_source()
        assert "EventType.RESPONSE_COMPLETE" in src

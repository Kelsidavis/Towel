"""Tests for generation cancellation."""

from towel.agent.events import AgentEvent, EventType
from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig


class TestCancelFlag:
    def test_cancel_sets_flag(self):
        config = TowelConfig()
        agent = AgentRuntime(config)
        assert not agent.is_cancelled
        agent.cancel()
        assert agent.is_cancelled

    def test_cancel_flag_resets(self):
        config = TowelConfig()
        agent = AgentRuntime(config)
        agent.cancel()
        assert agent.is_cancelled
        agent._cancel.clear()
        assert not agent.is_cancelled


class TestCancelledEvent:
    def test_cancelled_event_type(self):
        e = AgentEvent.cancelled("partial text", {"reason": "user_cancelled"})
        assert e.type == EventType.CANCELLED
        assert e.data["content"] == "partial text"
        assert e.data["metadata"]["reason"] == "user_cancelled"

    def test_cancelled_to_ws_message(self):
        e = AgentEvent.cancelled("partial", {"reason": "user_cancelled"})
        msg = e.to_ws_message("session-1")
        assert msg["type"] == "cancelled"
        assert msg["session"] == "session-1"
        assert msg["content"] == "partial"


class TestWebUICancel:
    """Test that the web UI has cancel support."""

    def test_stop_button_exists(self):
        from pathlib import Path

        html = (Path(__file__).parent.parent / "src" / "towel" / "web" / "index.html").read_text()
        assert "stop-btn" in html
        assert "stopGeneration" in html

    def test_escape_key_handler(self):
        from pathlib import Path

        html = (Path(__file__).parent.parent / "src" / "towel" / "web" / "index.html").read_text()
        assert "Escape" in html

    def test_cancel_message_sent(self):
        from pathlib import Path

        html = (Path(__file__).parent.parent / "src" / "towel" / "web" / "index.html").read_text()
        assert "'cancel'" in html

    def test_cancelled_event_handled(self):
        from pathlib import Path

        html = (Path(__file__).parent.parent / "src" / "towel" / "web" / "index.html").read_text()
        assert "'cancelled'" in html
        assert "generation stopped" in html


class TestGatewayCancel:
    """Test that the gateway handles cancel messages."""

    def test_gateway_has_cancel_handler(self):
        import inspect

        from towel.gateway.server import GatewayServer

        source = inspect.getsource(GatewayServer._handle_ws)
        assert "cancel" in source

    def test_gateway_tracks_active_tasks(self):
        from towel.gateway.server import GatewayServer

        config = TowelConfig()
        agent = AgentRuntime(config)
        gw = GatewayServer(config=config, agent=agent)
        assert hasattr(gw, "_active_tasks")
        assert isinstance(gw._active_tasks, dict)

    def test_gateway_ws_loop_tolerates_malformed_frames(self):
        """A worker (or a probing client) that sends a malformed JSON
        frame or a non-object should not kill the WebSocket connection
        — that would force the worker to reconnect and re-sync state.
        Verify the handler swallows the offending frame and keeps
        running."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # Malformed JSON is logged-and-skipped, not raised.
        assert "JSONDecodeError" in src
        assert "Ignoring malformed JSON" in src
        # Non-dict frames likewise.
        assert "isinstance(msg, dict)" in src
        assert "Ignoring non-object WS frame" in src

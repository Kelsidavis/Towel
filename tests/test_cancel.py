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

    def test_settings_panel_closes_on_escape(self):
        """Every overlay (fleet, memory, replay, memory-inspect) closes on
        Escape — settings was missing this and could only be dismissed by
        clicking outside it."""
        from pathlib import Path

        html = (Path(__file__).parent.parent / "src" / "towel" / "web" / "index.html").read_text()
        assert "settingsOverlay.style.display==='flex'){closeSettingsPanel();" in html


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

    def test_handle_ws_only_cancels_own_streaming_tasks(self):
        """The _active_tasks dict is coordinator-wide (keyed by
        session_id). If one WS connection's finally iterated all
        tasks and cancelled them, a second WS client streaming a
        response would have its task killed when the first client
        disconnected. Verify the handler tracks per-connection
        session ids and only cancels those."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # Per-connection set is built up as streaming starts.
        assert "my_session_tasks" in src
        # The finally iterates THAT set, not self._active_tasks.values().
        assert "for sid in my_session_tasks:" in src
        # The blind-cancel-all-tasks pattern must NOT be present.
        assert "for task in self._active_tasks.values()" not in src

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

    def test_gateway_ws_loop_tolerates_non_object_capabilities(self):
        """A register / heartbeat / memory_sync message with a non-
        object inner field (capabilities, mutations) shouldn't crash
        the loop either — same reconnect-storm avoidance."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # Register coerces non-dict capabilities to {}.
        assert "non-object capabilities" in src
        # Heartbeat skips the update on non-dict capabilities.
        assert "non-object" in src and "capabilities" in src
        # memory_sync coerces non-list mutations to [].
        assert "non-list" in src and "mutations" in src

    def test_gateway_ws_register_coerces_bad_id_types(self):
        """A worker that registers with `id=42` or `id=null` would
        previously land in the connections / workers dicts under a
        non-string key, then vanish from /workers/{id} HTTP lookups
        because the URL gives a string "42" not the integer 42. The
        register handler must coerce non-string ids to a string
        fallback (and cap length to keep /workers JSON sane)."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # The coercion lives in the register branch — verify it
        # checks isinstance(..., str) and caps to 256.
        assert "isinstance(raw_id, str)" in src
        assert "256" in src

    def test_gateway_ws_supports_verify_for_nonstreaming(self):
        """Parity with /api/ask and /v1/chat/completions — verify is
        reachable over WS too, with the same constraints (no
        streaming, mutually exclusive with ensemble)."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # Opt-in field is read from the WS frame.
        assert 'msg.get("verify"' in src
        # Mutually exclusive with ensemble at the WS layer.
        assert "not ensemble_flag" in src
        # Wired through to _verify_pass.
        assert "_verify_pass" in src

    def test_gateway_ws_supports_ensemble_for_nonstreaming(self):
        """Web UI / WS clients can opt into multi-worker ensemble on
        non-streaming requests. Streaming is intentionally not
        supported (synthesis can't be streamed). Source-level check
        guards the wiring."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # Opt-in field is read from the WS message frame.
        assert "ensemble_flag" in src
        assert 'msg.get("ensemble"' in src
        # Wires through to the shared helper.
        assert "_ensemble_dispatch" in src
        # Streaming requests still fall through to single-worker path.
        assert "ensemble_flag and not stream" in src

    def test_gateway_ws_message_coerces_bad_field_types(self):
        """A client sending {"type":"message","session":42} previously
        crashed deep in ConversationStore._path_for (iterating the int
        char-by-char), exited the read loop, and killed the WebSocket
        connection — a single bad client could disconnect itself.
        Coerce session_id / content / channel at the handler boundary
        so the read loop survives."""
        import inspect

        from towel.gateway.server import GatewayServer

        src = inspect.getsource(GatewayServer._handle_ws)
        # In the "message" branch, session_id, content, channel must
        # have type checks.
        assert "not isinstance(session_id, str)" in src
        assert "not isinstance(content, str)" in src
        assert "not isinstance(channel, str)" in src

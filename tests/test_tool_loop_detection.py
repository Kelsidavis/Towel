"""Tests for the shared tool-loop detection helpers in agent.runtime."""

from __future__ import annotations

from towel.agent.runtime import (
    TOOL_LOOP_REPEAT_LIMIT,
    _check_tool_loop,
    _tool_call_fingerprint,
)


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


class TestFingerprint:
    def test_fingerprint_stable_for_same_call(self):
        a = _tool_call_fingerprint([_FakeToolCall("read_file", {"path": "/a"})])
        b = _tool_call_fingerprint([_FakeToolCall("read_file", {"path": "/a"})])
        assert a == b

    def test_fingerprint_differs_for_different_args(self):
        a = _tool_call_fingerprint([_FakeToolCall("read_file", {"path": "/a"})])
        b = _tool_call_fingerprint([_FakeToolCall("read_file", {"path": "/b"})])
        assert a != b

    def test_fingerprint_differs_for_different_name(self):
        a = _tool_call_fingerprint([_FakeToolCall("read_file", {"path": "/a"})])
        b = _tool_call_fingerprint([_FakeToolCall("write_file", {"path": "/a"})])
        assert a != b

    def test_fingerprint_stable_under_key_reordering(self):
        # Same args, different dict insertion order — fingerprint must
        # still match so legitimate-looking variation in serialization
        # doesn't defeat detection.
        a = _tool_call_fingerprint([_FakeToolCall("x", {"a": 1, "b": 2})])
        b = _tool_call_fingerprint([_FakeToolCall("x", {"b": 2, "a": 1})])
        assert a == b


class TestCheckToolLoop:
    def test_returns_false_until_threshold(self):
        hist: list[str] = []
        for _ in range(TOOL_LOOP_REPEAT_LIMIT - 1):
            assert _check_tool_loop(hist, "same") is False
        # The LIMIT-th identical entry trips it.
        assert _check_tool_loop(hist, "same") is True

    def test_alternating_fingerprints_dont_trip(self):
        """A loop of two different calls (A B A B A B) shouldn't trip
        the detector — that's still progress, not a repeat-the-same-
        thing failure."""
        hist: list[str] = []
        for fp in ("a", "b", "a", "b", "a", "b"):
            assert _check_tool_loop(hist, fp) is False

    def test_history_bounded(self):
        """History list mustn't grow without bound — operators
        shouldn't pay memory for ancient iterations."""
        hist: list[str] = []
        for i in range(50):
            _check_tool_loop(hist, f"call-{i}")
        assert len(hist) <= TOOL_LOOP_REPEAT_LIMIT

    def test_resets_after_a_different_call(self):
        """3 of A → trip. Then a B breaks the streak — next 2 A's
        shouldn't trip (only 2 in a row again)."""
        hist: list[str] = []
        _check_tool_loop(hist, "a")
        _check_tool_loop(hist, "a")
        assert _check_tool_loop(hist, "a") is True  # 3 a's
        # Break the streak.
        assert _check_tool_loop(hist, "b") is False
        # Now: hist = ["a", "a", "b"] (sliding window).
        # Two more a's: hist = ["a", "b", "a"] → not all same.
        assert _check_tool_loop(hist, "a") is False
        # hist = ["b", "a", "a"] — still not all same.
        assert _check_tool_loop(hist, "a") is False


class TestStepStreamingPersistsTerminalMessages:
    """The streaming path has no return value — its callers (the WS
    handler, OpenAI-compat SSE) just forward events. So the agent
    must mutate the conversation itself when the loop terminates,
    not just emit a complete event. Otherwise replay of the saved
    transcript loses the assistant turn that ended the run. Same
    asymmetry that bit _stream_remote_inference in 803d1b4."""

    def test_step_streaming_persists_terminal_messages(self):
        import inspect

        from towel.agent.runtime import AgentRuntime

        src = inspect.getsource(AgentRuntime.step_streaming)
        # Both terminal branches must append to the conversation.
        assert "conversation.add(Role.ASSISTANT, stuck_msg)" in src
        assert "conversation.add(Role.ASSISTANT, max_iter_msg)" in src

"""Tests for agent events."""

from towel.agent.events import AgentEvent, EventType


def test_token_event():
    e = AgentEvent.token("hello")
    assert e.type == EventType.TOKEN
    assert e.data["content"] == "hello"


def test_tool_call_event():
    e = AgentEvent.tool_call("read_file", {"path": "/tmp/x"})
    assert e.type == EventType.TOOL_CALL
    assert e.data["tool"] == "read_file"
    assert e.data["arguments"]["path"] == "/tmp/x"


def test_tool_result_event():
    e = AgentEvent.tool_result("read_file", "file contents here")
    assert e.type == EventType.TOOL_RESULT
    assert e.data["result"] == "file contents here"


def test_complete_event():
    e = AgentEvent.complete("final answer", {"tps": 42.0, "tokens": 100})
    assert e.type == EventType.RESPONSE_COMPLETE
    assert e.data["content"] == "final answer"
    assert e.data["metadata"]["tps"] == 42.0


def test_error_event():
    e = AgentEvent.error("something broke")
    assert e.type == EventType.ERROR
    assert e.data["message"] == "something broke"


def test_to_ws_message():
    e = AgentEvent.token("hi")
    msg = e.to_ws_message("session-42")
    assert msg["type"] == "token"
    assert msg["session"] == "session-42"
    assert msg["content"] == "hi"


def test_complete_to_ws_message():
    e = AgentEvent.complete("done", {"tps": 10.0})
    msg = e.to_ws_message("s1")
    assert msg["type"] == "response_complete"
    assert msg["content"] == "done"
    assert msg["metadata"]["tps"] == 10.0

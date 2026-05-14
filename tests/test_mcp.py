"""Tests for the MCP server façade over the memory store."""

import json

import pytest

from towel.mcp.server import (
    MCP_PROTOCOL_VERSION,
    MemoryMCPServer,
    _tool_definitions,
)
from towel.memory.store import MemoryStore


@pytest.fixture
def server(tmp_path):
    return MemoryMCPServer(store=MemoryStore(store_dir=tmp_path))


# ── handshake ─────────────────────────────────────────────────────────


class TestHandshake:
    def test_initialize_returns_capabilities(self, server):
        reply = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            }
        )
        assert reply["jsonrpc"] == "2.0"
        assert reply["id"] == 1
        assert reply["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
        # Tools capability is what the spec uses to advertise tools/*.
        assert "tools" in reply["result"]["capabilities"]
        assert reply["result"]["serverInfo"]["name"] == "towel-memory"

    def test_initialized_notification_yields_no_reply(self, server):
        # Notifications carry no id and must never be replied to.
        reply = server.handle_request(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        assert reply is None

    def test_ping_works(self, server):
        reply = server.handle_request(
            {"jsonrpc": "2.0", "id": 99, "method": "ping"}
        )
        assert reply["result"] == {}


# ── tools/list ────────────────────────────────────────────────────────


class TestToolsList:
    def test_lists_seven_tools(self, server):
        reply = server.handle_request(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        )
        tools = reply["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {
            "memory_search",
            "memory_recall",
            "memory_list",
            "memory_remember",
            "memory_forget",
            "memory_related",
            "memory_stats",
        }

    def test_each_tool_has_input_schema(self):
        for tool in _tool_definitions():
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"


# ── tools/call dispatch ───────────────────────────────────────────────


def _call(server, name: str, arguments: dict):
    return server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )


def _text(reply: dict) -> str:
    return reply["result"]["content"][0]["text"]


class TestRemember:
    def test_remember_then_recall(self, server):
        _call(server, "memory_remember", {"key": "k", "content": "v"})
        reply = _call(server, "memory_recall", {"key": "k"})
        payload = json.loads(_text(reply))
        assert payload["content"] == "v"

    def test_remember_tags_mcp_source(self, server):
        _call(server, "memory_remember", {"key": "k", "content": "v"})
        assert server.store.recall("k").source == "mcp"

    def test_invalid_type_defaults_to_fact(self, server):
        _call(
            server,
            "memory_remember",
            {"key": "k", "content": "v", "type": "bogus"},
        )
        assert server.store.recall("k").memory_type == "fact"


class TestRememberTags:
    def test_remember_with_tags(self, server):
        _call(
            server,
            "memory_remember",
            {"key": "k", "content": "v", "tags": ["work", "urgent"]},
        )
        assert server.store.recall("k").tags == ["work", "urgent"]


class TestSearch:
    def test_search_returns_bm25_hit(self, server):
        server.store.remember("vim", "user edits with neovim", "preference")
        server.store.remember("noise", "completely unrelated", "fact")
        reply = _call(server, "memory_search", {"query": "neovim"})
        text = _text(reply)
        assert "vim" in text
        assert "completely unrelated" not in text

    def test_search_empty_no_results(self, server):
        reply = _call(server, "memory_search", {"query": "xyzzy"})
        assert "No matching" in _text(reply)


class TestList:
    def test_list_returns_json_array(self, server):
        server.store.remember("a", "1", "fact")
        server.store.remember("b", "2", "user")
        reply = _call(server, "memory_list", {})
        data = json.loads(_text(reply))
        assert {row["key"] for row in data} == {"a", "b"}

    def test_list_filters_by_type(self, server):
        server.store.remember("a", "1", "fact")
        server.store.remember("b", "2", "user")
        reply = _call(server, "memory_list", {"type": "user"})
        data = json.loads(_text(reply))
        assert [row["key"] for row in data] == ["b"]


class TestForget:
    def test_forget_known_key(self, server):
        server.store.remember("k", "v")
        reply = _call(server, "memory_forget", {"key": "k"})
        assert "Forgot" in _text(reply)
        assert server.store.recall("k") is None

    def test_forget_unknown_key(self, server):
        reply = _call(server, "memory_forget", {"key": "missing"})
        assert "No such key" in _text(reply)


class TestRelated:
    def test_related_returns_linked_entries(self, server):
        server.store.remember("a", "x", "fact")
        server.store.remember("b", "y", "fact")
        server.store._bump_recall(["a", "b"])
        reply = _call(server, "memory_related", {"key": "a"})
        data = json.loads(_text(reply))
        assert any(row["key"] == "b" for row in data)


class TestStats:
    def test_stats_returns_breakdown(self, server):
        server.store.remember("a", "x", "fact")
        server.store.remember("b", "y", "user", source="auto_capture:role")
        reply = _call(server, "memory_stats", {})
        data = json.loads(_text(reply))
        assert data["total"] == 2
        assert data["by_type"] == {"fact": 1, "user": 1}
        # 'b' came from auto_capture, 'a' is operator-set (empty source).
        assert "auto_capture:role" in data["by_source"]
        assert "operator" in data["by_source"]


# ── error paths ───────────────────────────────────────────────────────


class TestErrors:
    def test_unknown_method(self, server):
        reply = server.handle_request(
            {"jsonrpc": "2.0", "id": 5, "method": "nope/unknown"}
        )
        assert reply["error"]["code"] == -32601

    def test_unknown_tool(self, server):
        reply = _call(server, "imaginary_tool", {})
        assert reply["error"]["code"] == -32602

    def test_missing_required_argument(self, server):
        # memory_recall requires `key` — leaving it out should report
        # invalid params, not crash.
        reply = _call(server, "memory_recall", {})
        assert "error" in reply
        assert reply["error"]["code"] == -32602

    def test_tool_runtime_error_is_isError(self, server, monkeypatch):
        # Force the store to blow up so we can verify the convention
        # that tool errors come back via result.isError, not the
        # JSON-RPC error frame.
        def boom(*a, **kw):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(server.store, "recall_all", boom)
        reply = _call(server, "memory_stats", {})
        assert "result" in reply
        assert reply["result"]["isError"] is True
        assert "disk on fire" in reply["result"]["content"][0]["text"]

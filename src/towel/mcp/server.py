"""MCP server façade for the Towel memory store.

Implements the subset of the Model Context Protocol needed to expose
a memory backend: ``initialize`` handshake, ``tools/list``, and
``tools/call``. JSON-RPC 2.0 framing follows the MCP stdio transport
spec — one JSON object per line on stdin/stdout, with stderr reserved
for diagnostics so it doesn't corrupt the protocol stream.

The server holds a ``MemoryStore`` instance and translates each
incoming tool call into a method call on the store. Results come back
as MCP ``content`` arrays of ``text`` blocks; structured payloads
(stats, link weights) are JSON-encoded in the text body so a client
can either render them as-is or re-parse.

We hand-roll the protocol rather than depending on the official
``mcp`` Python SDK so Towel keeps zero new runtime deps for this
feature. Trade-off: we have to track protocol revisions ourselves;
the surface here targets the 2024-11-05 spec.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

from towel.memory.store import MEMORY_TYPES, MemoryStore, salience

log = logging.getLogger("towel.mcp")


MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "towel-memory"
# Match towel's package version so MCP clients can pin or warn.
try:
    from towel import __version__ as _TOWEL_VERSION
except Exception:
    _TOWEL_VERSION = "0.0.0"


# ── tool schemas ──────────────────────────────────────────────────────


def _tool_definitions() -> list[dict[str, Any]]:
    """Return MCP-shaped tool definitions for every exported operation."""
    return [
        {
            "name": "memory_search",
            "description": (
                "Search persistent memory by BM25 + vector + graph "
                "fusion. Returns up to `limit` entries ranked by "
                "combined relevance. Pass `tag` to restrict the "
                "result set to memories carrying that label, or "
                "`scope` to restrict to a specific project."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-form search string.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50,
                    },
                    "tag": {
                        "type": "string",
                        "description": "Restrict to memories carrying this exact tag.",
                    },
                    "scope": {
                        "type": "string",
                        "description": (
                            "Restrict to a specific project scope. "
                            'Pass "" for global only; omit to use '
                            "the store's default (project + global)."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "memory_recall",
            "description": "Fetch a single memory by exact key.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Memory key."},
                },
                "required": ["key"],
            },
        },
        {
            "name": "memory_list",
            "description": (
                "List memories, optionally filtered by type. Use this "
                "for full enumeration; prefer memory_search when looking "
                "for something specific."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": list(MEMORY_TYPES),
                        "description": "Filter by memory type.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
            },
        },
        {
            "name": "memory_remember",
            "description": (
                "Store or update a memory. Use for user preferences, "
                "project context, or anything worth keeping across "
                "sessions. Existing keys are updated in place; tags "
                "merge with any already present."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Short identifier (e.g. 'role').",
                    },
                    "content": {
                        "type": "string",
                        "description": "What to remember.",
                    },
                    "type": {
                        "type": "string",
                        "enum": list(MEMORY_TYPES),
                        "default": "fact",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Free-form labels for grouping.",
                    },
                    "scope": {
                        "type": "string",
                        "description": (
                            "Project scope. Default: store's default. "
                            'Pass "" for global (visible everywhere).'
                        ),
                    },
                },
                "required": ["key", "content"],
            },
        },
        {
            "name": "memory_forget",
            "description": "Delete a memory by key. Returns whether it existed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
        },
        {
            "name": "memory_related",
            "description": (
                "Return memories linked to `key` by co-retrieval weight. "
                "Useful for exploring the neighborhood of a known entry."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["key"],
            },
        },
        {
            "name": "memory_stats",
            "description": (
                "Aggregate counts and breakdowns: total entries, recall "
                "fraction, by-type and by-source distributions."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


# ── server core ───────────────────────────────────────────────────────


class MemoryMCPServer:
    """Stateless-ish MCP wrapper around a MemoryStore.

    One instance per process is plenty; the SQLite connection is
    short-lived per call (the store handles that internally) so
    long-running servers don't hold open transactions.
    """

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or MemoryStore()
        self._initialized = False
        # Mapping registered up-front so dispatch is a single dict lookup.
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "memory_search": self._t_search,
            "memory_recall": self._t_recall,
            "memory_list": self._t_list,
            "memory_remember": self._t_remember,
            "memory_forget": self._t_forget,
            "memory_related": self._t_related,
            "memory_stats": self._t_stats,
        }

    # ── tool implementations ─────────────────────────────────────────

    def _t_search(self, args: dict[str, Any]) -> str:
        query = args.get("query", "")
        limit = int(args.get("limit", 5))
        tag = args.get("tag") or None
        # scope=None means "use the store's default" (project + global)
        # which is usually what the caller wants. Explicit "" or
        # "proj:..." overrides; the empty string sentinel preserves
        # the store's contract for "global only".
        scope = args.get("scope")
        # fused_search runs the same BM25 + vector + graph RRF the
        # in-process runtime uses, so MCP clients get parity with the
        # local agent without duplicating retrieval logic.
        entries = self.store.fused_search(query, limit=limit, tag=tag, scope=scope)
        if not entries:
            return "No matching memories."
        lines = [f"Found {len(entries)} memor(ies):"]
        for e in entries:
            tag_str = f" {{{','.join(e.tags)}}}" if e.tags else ""
            lines.append(f"  [{e.memory_type}]{tag_str} {e.key}: {e.content}")
        return "\n".join(lines)

    def _t_recall(self, args: dict[str, Any]) -> str:
        key = args["key"]
        e = self.store.recall(key)
        if e is None:
            return f"No memory with key {key!r}."
        # Same shape as memory_inspect on the CLI — content + metadata
        # so a client can render or parse as it likes.
        return json.dumps(
            {
                **e.to_dict(),
                "salience": salience(e),
            },
            indent=2,
            ensure_ascii=False,
        )

    def _t_list(self, args: dict[str, Any]) -> str:
        mtype = args.get("type")
        limit = int(args.get("limit", 50))
        entries = self.store.recall_all(memory_type=mtype)[:limit]
        if not entries:
            return "No memories."
        return json.dumps(
            [e.to_dict() for e in entries], indent=2, ensure_ascii=False
        )

    def _t_remember(self, args: dict[str, Any]) -> str:
        key = args["key"]
        content = args["content"]
        mtype = args.get("type", "fact")
        if mtype not in MEMORY_TYPES:
            mtype = "fact"
        raw_tags = args.get("tags")
        tags = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else None
        scope = args.get("scope")
        # Tag with mcp so memory stats can see what landed via this path.
        e = self.store.remember(
            key, content, memory_type=mtype, source="mcp",
            tags=tags, scope=scope,
        )
        return f"Remembered [{e.memory_type}] {e.key}: {e.content}"

    def _t_forget(self, args: dict[str, Any]) -> str:
        key = args["key"]
        return "Forgot." if self.store.forget(key) else "No such key."

    def _t_related(self, args: dict[str, Any]) -> str:
        key = args["key"]
        limit = int(args.get("limit", 5))
        related = self.store.recall_related(key, limit=limit)
        if not related:
            return f"No related memories for {key!r}."
        return json.dumps(
            [
                {"weight": w, **rel.to_dict()}
                for rel, w in related
            ],
            indent=2,
            ensure_ascii=False,
        )

    def _t_stats(self, args: dict[str, Any]) -> str:
        entries = self.store.recall_all()
        by_type: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for e in entries:
            by_type[e.memory_type] = by_type.get(e.memory_type, 0) + 1
            src = (e.source or "") or "operator"
            by_source[src] = by_source.get(src, 0) + 1
        return json.dumps(
            {
                "total": len(entries),
                "recalled": sum(1 for e in entries if e.recall_count > 0),
                "total_recall_events": sum(e.recall_count for e in entries),
                "by_type": by_type,
                "by_source": by_source,
            },
            indent=2,
        )

    # ── JSON-RPC dispatch ────────────────────────────────────────────

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Process one JSON-RPC message, return the reply (or None for notifications)."""
        method = request.get("method")
        msg_id = request.get("id")
        params = request.get("params") or {}

        # Notifications carry no id; the protocol forbids replies to them.
        is_notification = msg_id is None

        try:
            if method == "initialize":
                self._initialized = True
                result = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": _TOWEL_VERSION},
                }
            elif method == "notifications/initialized":
                return None  # ack-only notification
            elif method == "tools/list":
                result = {"tools": _tool_definitions()}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                arguments = params.get("arguments") or {}
                handler = self._handlers.get(tool_name)
                if handler is None:
                    return self._error(msg_id, -32602, f"Unknown tool: {tool_name}")
                try:
                    text = handler(arguments)
                except KeyError as exc:
                    return self._error(msg_id, -32602, f"Missing required argument: {exc}")
                except Exception as exc:
                    log.exception("Tool %s failed", tool_name)
                    # Convention: tool-level failures come back via the
                    # result payload with isError=True, NOT via a JSON-RPC
                    # error frame. Lets the client distinguish protocol
                    # bugs from tool runtime issues.
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": f"Error: {exc}"}],
                            "isError": True,
                        },
                    }
                result = {"content": [{"type": "text", "text": text}]}
            elif method == "ping":
                result = {}
            else:
                return self._error(msg_id, -32601, f"Method not found: {method}")

            if is_notification:
                return None
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as exc:
            log.exception("Request handler crashed")
            return self._error(msg_id, -32603, f"Internal error: {exc}")

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }


def serve_stdio(store: MemoryStore | None = None) -> None:
    """Run the MCP server forever over stdin/stdout (the standard MCP transport).

    One JSON object per line on stdin → one JSON object per line on
    stdout. Diagnostics go to stderr so they never collide with the
    protocol stream. Returns when stdin closes (client disconnected).
    """
    server = MemoryMCPServer(store=store)
    log.info(
        "Towel MCP server starting on stdio (protocol=%s, version=%s)",
        MCP_PROTOCOL_VERSION, _TOWEL_VERSION,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"towel mcp: bad JSON: {exc}\n")
            continue
        reply = server.handle_request(request)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()

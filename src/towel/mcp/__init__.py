"""Towel MCP server — exposes the memory store to MCP clients.

Speaks JSON-RPC 2.0 over stdio per the Model Context Protocol spec,
letting external agents (Claude Code, Cursor, OpenCode, Gemini CLI,
any MCP-compatible client) read and write the same persistent memory
that Towel's own agent runtime uses. No extra dependency — the
protocol is small enough to implement against the spec directly.

Wire it up by adding to the client's MCP config, e.g. for Claude
Code's ``.mcp.json``::

    {
      "mcpServers": {
        "towel-memory": {
          "command": "towel",
          "args": ["mcp"]
        }
      }
    }

After restart, the client sees the seven memory tools and can call
them like any other native tool.
"""

from towel.mcp.server import MemoryMCPServer, serve_stdio

__all__ = ["MemoryMCPServer", "serve_stdio"]

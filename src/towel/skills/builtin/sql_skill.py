"""SQL skill — query SQLite databases, explain queries, format results."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

MAX_ROWS = 100
MAX_CELL = 500


class SqlSkill(Skill):
    @property
    def name(self) -> str:
        return "sql"

    @property
    def description(self) -> str:
        return "Query SQLite databases, inspect schema, and format results"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="sql_query",
                description="Run a SELECT query on a SQLite database and return results",
                parameters={
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "Path to .db or .sqlite file"},
                        "query": {"type": "string", "description": "SQL SELECT query"},
                        "limit": {"type": "integer", "description": "Max rows (default: 50)"},
                    },
                    "required": ["database", "query"],
                },
            ),
            ToolDefinition(
                name="sql_schema",
                description="Show tables and column definitions in a SQLite database",
                parameters={
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "Path to database"},
                        "table": {"type": "string", "description": "Specific table (optional, shows all if omitted)"},
                    },
                    "required": ["database"],
                },
            ),
            ToolDefinition(
                name="sql_explain",
                description="Show the query execution plan (EXPLAIN QUERY PLAN)",
                parameters={
                    "type": "object",
                    "properties": {
                        "database": {"type": "string", "description": "Path to database"},
                        "query": {"type": "string", "description": "SQL query to explain"},
                    },
                    "required": ["database", "query"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "sql_query":
                return self._query(
                    arguments["database"], arguments["query"],
                    arguments.get("limit", 50),
                )
            case "sql_schema":
                return self._schema(arguments["database"], arguments.get("table"))
            case "sql_explain":
                return self._explain(arguments["database"], arguments["query"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _connect(self, db_path: str) -> sqlite3.Connection | str:
        p = Path(db_path).expanduser()
        if not p.is_file():
            return f"Database not found: {db_path}"
        try:
            conn = sqlite3.connect(str(p))
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            return f"Connection error: {e}"

    def _query(self, db_path: str, query: str, limit: int) -> str:
        # Safety: only allow SELECT and PRAGMA
        stripped = query.strip().upper()
        if not (stripped.startswith("SELECT") or stripped.startswith("PRAGMA") or stripped.startswith("WITH")):
            return "Only SELECT, WITH, and PRAGMA queries are allowed (read-only)."

        conn = self._connect(db_path)
        if isinstance(conn, str):
            return conn

        try:
            limit = min(limit, MAX_ROWS)
            cursor = conn.execute(query)
            rows = cursor.fetchmany(limit + 1)

            if not rows:
                return "Query returned 0 rows."

            cols = [desc[0] for desc in cursor.description]
            truncated = len(rows) > limit
            rows = rows[:limit]

            # Format as table
            lines = [" | ".join(cols)]
            lines.append("-+-".join("-" * max(len(c), 5) for c in cols))
            for row in rows:
                vals = []
                for v in row:
                    s = str(v) if v is not None else "NULL"
                    if len(s) > MAX_CELL:
                        s = s[:MAX_CELL] + "..."
                    vals.append(s)
                lines.append(" | ".join(vals))

            result = "\n".join(lines)
            if truncated:
                result += f"\n\n... (limited to {limit} rows)"
            else:
                result = f"{len(rows)} row(s):\n\n{result}"
            return result

        except sqlite3.Error as e:
            return f"SQL error: {e}"
        finally:
            conn.close()

    def _schema(self, db_path: str, table: str | None) -> str:
        conn = self._connect(db_path)
        if isinstance(conn, str):
            return conn

        try:
            if table:
                cursor = conn.execute(f"PRAGMA table_info({table})")
                cols = cursor.fetchall()
                if not cols:
                    return f"Table not found: {table}"
                lines = [f"Table: {table}\n"]
                for c in cols:
                    pk = " PRIMARY KEY" if c["pk"] else ""
                    nn = " NOT NULL" if c["notnull"] else ""
                    default = f" DEFAULT {c['dflt_value']}" if c["dflt_value"] else ""
                    lines.append(f"  {c['name']} {c['type']}{pk}{nn}{default}")
                return "\n".join(lines)
            else:
                cursor = conn.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
                items = cursor.fetchall()
                if not items:
                    return "Database has no tables."
                lines = [f"Database: {Path(db_path).name}\n"]
                for item in items:
                    cols = conn.execute(f"PRAGMA table_info({item['name']})").fetchall()
                    col_names = ", ".join(c["name"] for c in cols[:8])
                    if len(cols) > 8:
                        col_names += f", ... (+{len(cols)-8})"
                    lines.append(f"  {item['type']} {item['name']} ({len(cols)} cols: {col_names})")
                return "\n".join(lines)

        except sqlite3.Error as e:
            return f"Error: {e}"
        finally:
            conn.close()

    def _explain(self, db_path: str, query: str) -> str:
        conn = self._connect(db_path)
        if isinstance(conn, str):
            return conn

        try:
            cursor = conn.execute(f"EXPLAIN QUERY PLAN {query}")
            rows = cursor.fetchall()
            if not rows:
                return "No execution plan available."
            lines = ["Query plan:"]
            for row in rows:
                detail = row["detail"] if "detail" in row.keys() else str(dict(row))
                lines.append(f"  {detail}")
            return "\n".join(lines)
        except sqlite3.Error as e:
            return f"Error: {e}"
        finally:
            conn.close()

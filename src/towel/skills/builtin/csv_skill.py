"""CSV power tools — filter, sort, aggregate, and transform CSV data."""

from __future__ import annotations

import csv
import io
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class CsvSkill(Skill):
    @property
    def name(self) -> str:
        return "csv_tools"

    @property
    def description(self) -> str:
        return "Advanced CSV operations — filter, sort, aggregate, pivot"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="csv_filter",
                description="Filter CSV rows by column value",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV text"},
                        "column": {"type": "string", "description": "Column name to filter on"},
                        "value": {
                            "type": "string",
                            "description": "Value to match (supports >, <, >=, <=, != prefixes)",
                        },
                    },
                    "required": ["data", "column", "value"],
                },
            ),
            ToolDefinition(
                name="csv_sort",
                description="Sort CSV rows by a column",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV text"},
                        "column": {"type": "string", "description": "Column to sort by"},
                        "descending": {
                            "type": "boolean",
                            "description": "Sort descending (default: false)",
                        },
                    },
                    "required": ["data", "column"],
                },
            ),
            ToolDefinition(
                name="csv_aggregate",
                description="Aggregate CSV data — count, sum, avg, min, max per group",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV text"},
                        "group_by": {"type": "string", "description": "Column to group by"},
                        "value_column": {"type": "string", "description": "Column to aggregate"},
                        "operation": {
                            "type": "string",
                            "enum": ["count", "sum", "avg", "min", "max"],
                            "description": "Aggregation",
                        },
                    },
                    "required": ["data", "group_by"],
                },
            ),
            ToolDefinition(
                name="csv_columns",
                description="Show column names, types, and sample values from CSV",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV text"},
                    },
                    "required": ["data"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "csv_filter":
                return self._filter(arguments["data"], arguments["column"], arguments["value"])
            case "csv_sort":
                return self._sort(
                    arguments["data"], arguments["column"], arguments.get("descending", False)
                )
            case "csv_aggregate":
                return self._aggregate(
                    arguments["data"],
                    arguments["group_by"],
                    arguments.get("value_column"),
                    arguments.get("operation", "count"),
                )
            case "csv_columns":
                return self._columns(arguments["data"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _parse(self, data: str) -> tuple[list[str], list[dict]]:
        reader = csv.DictReader(io.StringIO(data))
        rows = [dict(r) for r in reader]
        cols = list(rows[0].keys()) if rows else []
        return cols, rows

    def _to_csv(self, cols: list[str], rows: list[dict]) -> str:
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        w.writerows(rows[:200])
        return out.getvalue()

    def _filter(self, data: str, column: str, value: str) -> str:
        cols, rows = self._parse(data)
        if column not in cols:
            return f"Column not found: {column}"
        op, val = "==", value
        for prefix in [">=", "<=", "!=", ">", "<"]:
            if value.startswith(prefix):
                op, val = prefix, value[len(prefix) :]
                break
        filtered = []
        for r in rows:
            cell = r.get(column, "")
            try:
                cn, vn = float(cell), float(val)
                match op:
                    case "==":
                        ok = cn == vn
                    case "!=":
                        ok = cn != vn
                    case ">":
                        ok = cn > vn
                    case "<":
                        ok = cn < vn
                    case ">=":
                        ok = cn >= vn
                    case "<=":
                        ok = cn <= vn
                    case _:
                        ok = False
            except ValueError:
                match op:
                    case "==":
                        ok = cell == val
                    case "!=":
                        ok = cell != val
                    case _:
                        ok = cell == val
            if ok:
                filtered.append(r)
        return f"{len(filtered)} rows (from {len(rows)}):\n\n{self._to_csv(cols, filtered)}"

    def _sort(self, data: str, column: str, desc: bool) -> str:
        cols, rows = self._parse(data)
        if column not in cols:
            return f"Column not found: {column}"

        def key(r):
            v = r.get(column, "")
            try:
                return (0, float(v))
            except ValueError:
                return (1, v.lower())

        rows.sort(key=key, reverse=desc)
        return self._to_csv(cols, rows)

    def _aggregate(self, data: str, group_by: str, value_col: str | None, op: str) -> str:
        cols, rows = self._parse(data)
        if group_by not in cols:
            return f"Column not found: {group_by}"
        groups: dict[str, list] = {}
        for r in rows:
            key = r.get(group_by, "")
            groups.setdefault(key, []).append(r)
        lines = [f"Aggregation: {op} by {group_by}" + (f" on {value_col}" if value_col else "")]
        for gk in sorted(groups.keys()):
            gr = groups[gk]
            if op == "count":
                lines.append(f"  {gk}: {len(gr)}")
            elif value_col and value_col in cols:
                vals = []
                for r in gr:
                    try:
                        vals.append(float(r.get(value_col, 0)))
                    except ValueError:
                        pass
                if not vals:
                    lines.append(f"  {gk}: (no numeric values)")
                    continue
                match op:
                    case "sum":
                        lines.append(f"  {gk}: {sum(vals):,.2f}")
                    case "avg":
                        lines.append(f"  {gk}: {sum(vals) / len(vals):,.2f}")
                    case "min":
                        lines.append(f"  {gk}: {min(vals):,.2f}")
                    case "max":
                        lines.append(f"  {gk}: {max(vals):,.2f}")
                    case _:
                        lines.append(f"  {gk}: {len(gr)}")
            else:
                lines.append(f"  {gk}: {len(gr)}")
        return "\n".join(lines)

    def _columns(self, data: str) -> str:
        cols, rows = self._parse(data)
        if not cols:
            return "No columns found."
        lines = [f"Columns ({len(cols)}):"]
        for c in cols:
            vals = [r.get(c, "") for r in rows[:10]]
            sample = vals[0] if vals else ""
            is_num = all(v.replace(".", "", 1).replace("-", "", 1).isdigit() for v in vals if v)
            dtype = "numeric" if is_num and vals else "text"
            lines.append(f"  {c} ({dtype}): {sample!r}")
        lines.append(f"\nRows: {len(rows)}")
        return "\n".join(lines)

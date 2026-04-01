"""Data skill — parse, query, and transform structured data."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from towel.skills.base import Skill, ToolDefinition

MAX_OUTPUT = 50_000


class DataSkill(Skill):
    @property
    def name(self) -> str:
        return "data"

    @property
    def description(self) -> str:
        return "Parse, query, and transform JSON, CSV, and structured data"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="parse_json",
                description=(
                    "Parse a JSON string and extract data using a "
                    "key path (e.g., 'users.0.name' or 'items.*.id')"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON string to parse"},
                        "path": {
                            "type": "string",
                            "description": (
                                "Dot-separated key path (optional). "
                                "Use * for array wildcard."
                            ),
                        },
                    },
                    "required": ["data"],
                },
            ),
            ToolDefinition(
                name="parse_csv",
                description="Parse CSV data and return as structured records",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "CSV string to parse"},
                        "delimiter": {
                            "type": "string",
                            "description": "Field delimiter (default: comma)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max rows to return (default: 50)",
                        },
                    },
                    "required": ["data"],
                },
            ),
            ToolDefinition(
                name="format_json",
                description="Pretty-print or compact JSON data",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON string"},
                        "compact": {
                            "type": "boolean",
                            "description": "Compact output (default: false, pretty-print)",
                        },
                    },
                    "required": ["data"],
                },
            ),
            ToolDefinition(
                name="calculate",
                description=(
                    "Evaluate a mathematical expression safely. "
                    "Supports +, -, *, /, **, %, abs, round, "
                    "min, max, sum, len."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate",
                        },
                    },
                    "required": ["expression"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "parse_json":
                return self._parse_json(arguments["data"], arguments.get("path"))
            case "parse_csv":
                return self._parse_csv(
                    arguments["data"],
                    arguments.get("delimiter", ","),
                    arguments.get("limit", 50),
                )
            case "format_json":
                return self._format_json(arguments["data"], arguments.get("compact", False))
            case "calculate":
                return self._calculate(arguments["expression"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _parse_json(self, data: str, path: str | None) -> str:
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        if not path:
            return json.dumps(obj, indent=2, ensure_ascii=False)[:MAX_OUTPUT]

        # Navigate the path
        result = self._query_path(obj, path.split("."))
        return json.dumps(result, indent=2, ensure_ascii=False)[:MAX_OUTPUT]

    def _query_path(self, obj: Any, keys: list[str]) -> Any:
        """Navigate a dot-path through nested data. Supports * wildcard for arrays."""
        current = obj
        for key in keys:
            if key == "*" and isinstance(current, list):
                remaining = keys[keys.index(key) + 1 :]
                return [self._query_path(item, remaining) for item in current]
            elif isinstance(current, dict):
                current = current.get(key)
                if current is None:
                    return None
            elif isinstance(current, list):
                try:
                    current = current[int(key)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return current

    def _parse_csv(self, data: str, delimiter: str, limit: int) -> str:
        try:
            reader = csv.DictReader(io.StringIO(data), delimiter=delimiter)
            rows = []
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                rows.append(dict(row))

            if not rows:
                return "No data rows found."

            result = {
                "columns": list(rows[0].keys()),
                "row_count": len(rows),
                "rows": rows,
            }
            return json.dumps(result, indent=2, ensure_ascii=False)[:MAX_OUTPUT]

        except Exception as e:
            return f"CSV parse error: {e}"

    def _format_json(self, data: str, compact: bool) -> str:
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        if compact:
            return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)[:MAX_OUTPUT]
        return json.dumps(obj, indent=2, ensure_ascii=False)[:MAX_OUTPUT]

    def _calculate(self, expression: str) -> str:
        """Safely evaluate a math expression."""
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "int": int,
            "float": float,
            "pow": pow,
        }
        try:
            # Only allow safe operations
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

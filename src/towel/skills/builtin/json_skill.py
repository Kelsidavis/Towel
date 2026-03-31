"""JSON power tools — diff, patch, flatten, validate, and generate schemas."""

from __future__ import annotations

import json
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class JsonSkill(Skill):
    @property
    def name(self) -> str:
        return "json_tools"

    @property
    def description(self) -> str:
        return "Advanced JSON operations: diff, flatten, validate, schema generation"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="json_diff",
                description="Compare two JSON objects and show the differences",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "string", "description": "First JSON string"},
                        "b": {"type": "string", "description": "Second JSON string"},
                    },
                    "required": ["a", "b"],
                },
            ),
            ToolDefinition(
                name="json_flatten",
                description="Flatten nested JSON into dot-notation key-value pairs",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON string to flatten"},
                        "separator": {"type": "string", "description": "Key separator (default: '.')"},
                    },
                    "required": ["data"],
                },
            ),
            ToolDefinition(
                name="json_schema",
                description="Generate a JSON Schema from a sample JSON object",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Sample JSON to generate schema from"},
                    },
                    "required": ["data"],
                },
            ),
            ToolDefinition(
                name="json_validate",
                description="Check if a JSON string is valid and report any syntax errors",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON string to validate"},
                    },
                    "required": ["data"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "json_diff":
                return self._diff(arguments["a"], arguments["b"])
            case "json_flatten":
                return self._flatten(arguments["data"], arguments.get("separator", "."))
            case "json_schema":
                return self._schema(arguments["data"])
            case "json_validate":
                return self._validate(arguments["data"])
            case _:
                return f"Unknown tool: {tool_name}"

    @staticmethod
    def _parse_json_input(data: str | dict | list) -> Any:
        """Parse JSON input — accept both strings and already-parsed objects."""
        if isinstance(data, (dict, list, int, float, bool)):
            return data
        return json.loads(data)

    def _diff(self, a_str: str, b_str: str) -> str:
        try:
            a = self._parse_json_input(a_str)
            b = self._parse_json_input(b_str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        changes = []
        self._compare("$", a, b, changes)
        if not changes:
            return "Objects are identical."
        return f"{len(changes)} difference(s):\n" + "\n".join(changes[:50])

    def _compare(self, path: str, a: Any, b: Any, changes: list[str]) -> None:
        if type(a) != type(b):
            changes.append(f"  ~ {path}: type changed {type(a).__name__} -> {type(b).__name__}")
            return
        if isinstance(a, dict):
            all_keys = set(a.keys()) | set(b.keys())
            for k in sorted(all_keys):
                child = f"{path}.{k}"
                if k not in a:
                    changes.append(f"  + {child}: {json.dumps(b[k])[:80]}")
                elif k not in b:
                    changes.append(f"  - {child}: {json.dumps(a[k])[:80]}")
                else:
                    self._compare(child, a[k], b[k], changes)
        elif isinstance(a, list):
            for i in range(max(len(a), len(b))):
                child = f"{path}[{i}]"
                if i >= len(a):
                    changes.append(f"  + {child}: {json.dumps(b[i])[:80]}")
                elif i >= len(b):
                    changes.append(f"  - {child}: {json.dumps(a[i])[:80]}")
                else:
                    self._compare(child, a[i], b[i], changes)
        elif a != b:
            changes.append(f"  ~ {path}: {json.dumps(a)[:40]} -> {json.dumps(b)[:40]}")

    def _flatten(self, data_str: str | dict | list, sep: str) -> str:
        try:
            obj = self._parse_json_input(data_str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        flat: dict[str, Any] = {}
        self._flatten_obj("", obj, flat, sep)
        return json.dumps(flat, indent=2, ensure_ascii=False)[:10000]

    def _flatten_obj(self, prefix: str, obj: Any, out: dict, sep: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}{sep}{k}" if prefix else k
                self._flatten_obj(key, v, out, sep)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                key = f"{prefix}[{i}]"
                self._flatten_obj(key, v, out, sep)
        else:
            out[prefix] = obj

    def _schema(self, data_str: str | dict | list) -> str:
        try:
            obj = self._parse_json_input(data_str)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        schema = self._infer_schema(obj)
        return json.dumps(schema, indent=2, ensure_ascii=False)[:10000]

    def _infer_schema(self, obj: Any) -> dict:
        if obj is None:
            return {"type": "null"}
        if isinstance(obj, bool):
            return {"type": "boolean"}
        if isinstance(obj, int):
            return {"type": "integer"}
        if isinstance(obj, float):
            return {"type": "number"}
        if isinstance(obj, str):
            return {"type": "string"}
        if isinstance(obj, list):
            if not obj:
                return {"type": "array", "items": {}}
            return {"type": "array", "items": self._infer_schema(obj[0])}
        if isinstance(obj, dict):
            props = {k: self._infer_schema(v) for k, v in obj.items()}
            return {
                "type": "object",
                "properties": props,
                "required": list(obj.keys()),
            }
        return {}

    def _validate(self, data_str: str | dict | list) -> str:
        try:
            obj = self._parse_json_input(data_str)
            kind = type(obj).__name__
            if isinstance(obj, dict):
                return f"Valid JSON object with {len(obj)} key(s)."
            elif isinstance(obj, list):
                return f"Valid JSON array with {len(obj)} element(s)."
            else:
                return f"Valid JSON ({kind}): {json.dumps(obj)[:100]}"
        except json.JSONDecodeError as e:
            line = e.lineno
            col = e.colno
            return f"Invalid JSON at line {line}, column {col}: {e.msg}"

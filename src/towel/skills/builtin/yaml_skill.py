"""YAML skill — parse, validate, and convert YAML."""

from __future__ import annotations

import json
from typing import Any

from towel.skills.base import Skill, ToolDefinition

try:
    import toml as _toml
except ImportError:
    _toml = None

# Minimal YAML parser (no PyYAML dependency) for simple cases
def _simple_yaml_parse(text: str) -> dict | list | str:
    """Parse simple YAML (flat key-value, lists). Falls back to raw text."""
    lines = text.strip().splitlines()
    if not lines: return {}

    # Check if it's a simple list
    if all(l.strip().startswith("- ") for l in lines if l.strip()):
        return [l.strip()[2:] for l in lines if l.strip()]

    # Try key-value pairs
    result: dict[str, Any] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"): continue
        if ": " in line:
            key, _, val = line.partition(": ")
            key = key.strip()
            val = val.strip()
            if val.lower() in ("true","yes"): val = True
            elif val.lower() in ("false","no"): val = False
            elif val.replace(".","",1).lstrip("-").isdigit():
                val = float(val) if "." in val else int(val)
            elif val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip("'\"") for v in val[1:-1].split(",")]
            elif val.startswith("'") or val.startswith('"'):
                val = val.strip("'\"")
            result[key] = val
        elif line.endswith(":"): result[line[:-1].strip()] = {}
    return result if result else text


class YamlSkill(Skill):
    @property
    def name(self) -> str: return "yaml"
    @property
    def description(self) -> str: return "Parse YAML, convert between YAML/JSON/TOML"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="yaml_parse", description="Parse YAML text and return as structured JSON",
                parameters={"type":"object","properties":{
                    "text":{"type":"string","description":"YAML text to parse"},
                },"required":["text"]}),
            ToolDefinition(name="yaml_to_json", description="Convert YAML to JSON",
                parameters={"type":"object","properties":{
                    "text":{"type":"string","description":"YAML text"},
                },"required":["text"]}),
            ToolDefinition(name="json_to_yaml", description="Convert JSON to YAML-like format",
                parameters={"type":"object","properties":{
                    "text":{"type":"string","description":"JSON text"},
                },"required":["text"]}),
            ToolDefinition(name="yaml_validate", description="Check if YAML text is valid",
                parameters={"type":"object","properties":{
                    "text":{"type":"string","description":"YAML text to validate"},
                },"required":["text"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "yaml_parse": return self._parse(arguments["text"])
            case "yaml_to_json": return self._to_json(arguments["text"])
            case "json_to_yaml": return self._to_yaml(arguments["text"])
            case "yaml_validate": return self._validate(arguments["text"])
            case _: return f"Unknown tool: {tool_name}"

    def _parse(self, text: str) -> str:
        result = _simple_yaml_parse(text)
        return json.dumps(result, indent=2, ensure_ascii=False)

    def _to_json(self, text: str) -> str:
        result = _simple_yaml_parse(text)
        return json.dumps(result, indent=2, ensure_ascii=False)

    def _to_yaml(self, text: str) -> str:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"
        return self._render_yaml(obj, 0)

    def _render_yaml(self, obj: Any, depth: int) -> str:
        indent = "  " * depth
        lines: list[str] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{indent}{k}:")
                    lines.append(self._render_yaml(v, depth + 1))
                elif isinstance(v, bool):
                    lines.append(f"{indent}{k}: {'true' if v else 'false'}")
                elif v is None:
                    lines.append(f"{indent}{k}: null")
                else:
                    lines.append(f"{indent}{k}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}-")
                    lines.append(self._render_yaml(item, depth + 1))
                else:
                    lines.append(f"{indent}- {item}")
        else:
            lines.append(f"{indent}{obj}")
        return "\n".join(lines)

    def _validate(self, text: str) -> str:
        try:
            result = _simple_yaml_parse(text)
            if isinstance(result, dict):
                return f"Valid YAML: {len(result)} top-level keys"
            elif isinstance(result, list):
                return f"Valid YAML: list with {len(result)} items"
            else:
                return "Parsed as raw text (complex YAML may need PyYAML)"
        except Exception as e:
            return f"Parse error: {e}"

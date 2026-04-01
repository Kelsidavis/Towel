"""Markdown skill — generate tables, TOCs, and convert formats."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class MarkdownSkill(Skill):
    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return "Generate markdown tables, TOCs, and format conversions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="md_table",
                description="Generate a markdown table from JSON array or CSV data",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "JSON array of objects or CSV text",
                        },
                        "alignment": {
                            "type": "string",
                            "description": "Column alignment: left, center, right (default: left)",
                        },
                    },
                    "required": ["data"],
                },
            ),
            ToolDefinition(
                name="md_toc",
                description="Generate a table of contents from markdown headings",
                parameters={
                    "type": "object",
                    "properties": {
                        "markdown": {
                            "type": "string",
                            "description": "Markdown text to generate TOC from",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max heading depth (default: 3)",
                        },
                    },
                    "required": ["markdown"],
                },
            ),
            ToolDefinition(
                name="md_checklist",
                description="Generate a markdown checklist from a list of items",
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of checklist items",
                        },
                        "checked": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Indices of checked items (0-based)",
                        },
                    },
                    "required": ["items"],
                },
            ),
            ToolDefinition(
                name="json_to_md",
                description="Convert a JSON object to a readable markdown document",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON string to convert"},
                    },
                    "required": ["data"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "md_table":
                return self._table(arguments["data"], arguments.get("alignment", "left"))
            case "md_toc":
                return self._toc(arguments["markdown"], arguments.get("max_depth", 3))
            case "md_checklist":
                return self._checklist(arguments["items"], arguments.get("checked", []))
            case "json_to_md":
                return self._json_to_md(arguments["data"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _table(self, data: str, alignment: str) -> str:
        # Try JSON first
        rows: list[dict] = []
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                rows = parsed
        except json.JSONDecodeError:
            pass

        # Try CSV
        if not rows:
            try:
                reader = csv.DictReader(io.StringIO(data))
                rows = [dict(r) for r in reader]
            except Exception:
                return "Could not parse data as JSON array or CSV."

        if not rows:
            return "No data rows found."

        cols = list(rows[0].keys())
        align_map = {"left": ":---", "center": ":---:", "right": "---:"}
        sep = align_map.get(alignment, ":---")

        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(sep for _ in cols) + " |")
        for row in rows[:100]:
            vals = [str(row.get(c, "")).replace("|", "\\|") for c in cols]
            lines.append("| " + " | ".join(vals) + " |")

        return "\n".join(lines)

    def _toc(self, markdown: str, max_depth: int) -> str:
        import re

        lines = []
        for line in markdown.splitlines():
            m = re.match(r"^(#{1,6})\s+(.+)", line)
            if m:
                level = len(m.group(1))
                if level > max_depth:
                    continue
                title = m.group(2).strip()
                slug = re.sub(r"[^\w\s-]", "", title.lower()).replace(" ", "-")
                indent = "  " * (level - 1)
                lines.append(f"{indent}- [{title}](#{slug})")

        if not lines:
            return "No headings found."
        return "## Table of Contents\n\n" + "\n".join(lines)

    def _checklist(self, items: list[str], checked: list[int]) -> str:
        checked_set = set(checked)
        lines = []
        for i, item in enumerate(items):
            mark = "x" if i in checked_set else " "
            lines.append(f"- [{mark}] {item}")
        return "\n".join(lines)

    def _json_to_md(self, data: str) -> str:
        try:
            obj = json.loads(data)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"
        return self._render_obj(obj, depth=0)

    def _render_obj(self, obj: Any, depth: int) -> str:
        lines: list[str] = []
        prefix = "#" * min(depth + 2, 6)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix} {k}\n")
                    lines.append(self._render_obj(v, depth + 1))
                else:
                    lines.append(f"- **{k}:** {v}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, dict):
                    lines.append(self._render_obj(item, depth))
                    if i < len(obj) - 1:
                        lines.append("---")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append(str(obj))
        return "\n".join(lines)

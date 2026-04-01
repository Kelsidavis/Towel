"""Diff skill — compare files and text."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class DiffSkill(Skill):
    @property
    def name(self) -> str:
        return "diff"

    @property
    def description(self) -> str:
        return "Compare files or text and show differences"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="diff_files",
                description="Compare two files and show a unified diff",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_a": {"type": "string", "description": "Path to first file"},
                        "file_b": {"type": "string", "description": "Path to second file"},
                        "context": {
                            "type": "integer",
                            "description": "Lines of context (default: 3)",
                        },
                    },
                    "required": ["file_a", "file_b"],
                },
            ),
            ToolDefinition(
                name="diff_text",
                description="Compare two text strings and show differences",
                parameters={
                    "type": "object",
                    "properties": {
                        "text_a": {"type": "string", "description": "First text"},
                        "text_b": {"type": "string", "description": "Second text"},
                        "label_a": {
                            "type": "string",
                            "description": "Label for first text (default: 'a')",
                        },
                        "label_b": {
                            "type": "string",
                            "description": "Label for second text (default: 'b')",
                        },
                    },
                    "required": ["text_a", "text_b"],
                },
            ),
            ToolDefinition(
                name="diff_stats",
                description=(
                    "Get statistics about differences between "
                    "two files (additions, deletions, changed lines)"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_a": {"type": "string", "description": "Path to first file"},
                        "file_b": {"type": "string", "description": "Path to second file"},
                    },
                    "required": ["file_a", "file_b"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "diff_files":
                return self._diff_files(
                    arguments["file_a"],
                    arguments["file_b"],
                    arguments.get("context", 3),
                )
            case "diff_text":
                return self._diff_text(
                    arguments["text_a"],
                    arguments["text_b"],
                    arguments.get("label_a", "a"),
                    arguments.get("label_b", "b"),
                )
            case "diff_stats":
                return self._diff_stats(arguments["file_a"], arguments["file_b"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _read(self, path: str) -> list[str] | str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"File not found: {path}"
        try:
            return p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except OSError as e:
            return f"Error reading {path}: {e}"

    def _diff_files(self, file_a: str, file_b: str, context: int) -> str:
        lines_a = self._read(file_a)
        if isinstance(lines_a, str):
            return lines_a
        lines_b = self._read(file_b)
        if isinstance(lines_b, str):
            return lines_b

        diff = list(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=file_a,
                tofile=file_b,
                n=context,
            )
        )
        if not diff:
            return "Files are identical."
        return "".join(diff[:500])

    def _diff_text(self, text_a: str, text_b: str, label_a: str, label_b: str) -> str:
        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)
        diff = list(difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b))
        if not diff:
            return "Texts are identical."
        return "".join(diff[:500])

    def _diff_stats(self, file_a: str, file_b: str) -> str:
        lines_a = self._read(file_a)
        if isinstance(lines_a, str):
            return lines_a
        lines_b = self._read(file_b)
        if isinstance(lines_b, str):
            return lines_b

        sm = difflib.SequenceMatcher(None, lines_a, lines_b)
        adds = dels = changes = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "insert":
                adds += j2 - j1
            elif tag == "delete":
                dels += i2 - i1
            elif tag == "replace":
                changes += max(i2 - i1, j2 - j1)

        ratio = sm.ratio()
        return (
            f"Diff stats ({file_a} vs {file_b}):\n"
            f"  Additions: +{adds} lines\n"
            f"  Deletions: -{dels} lines\n"
            f"  Changed:   ~{changes} lines\n"
            f"  Similarity: {ratio:.1%}\n"
            f"  File A: {len(lines_a)} lines\n"
            f"  File B: {len(lines_b)} lines"
        )

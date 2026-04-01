"""Regex skill — test, match, and replace with regular expressions."""

from __future__ import annotations

import re
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class RegexSkill(Skill):
    @property
    def name(self) -> str:
        return "regex"

    @property
    def description(self) -> str:
        return "Test, match, extract, and replace using regular expressions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="regex_test",
                description=(
                    "Test if a regex pattern matches a string. "
                    "Returns match details or 'no match'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regular expression pattern"},
                        "text": {"type": "string", "description": "Text to test against"},
                        "flags": {
                            "type": "string",
                            "description": (
                                "Flags: i=ignorecase, m=multiline, "
                                "s=dotall (e.g., 'im')"
                            ),
                        },
                    },
                    "required": ["pattern", "text"],
                },
            ),
            ToolDefinition(
                name="regex_findall",
                description=(
                    "Find all matches of a pattern in text. "
                    "Returns list of matches with positions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regular expression pattern"},
                        "text": {"type": "string", "description": "Text to search"},
                        "flags": {
                            "type": "string",
                            "description": "Flags: i=ignorecase, m=multiline, s=dotall",
                        },
                    },
                    "required": ["pattern", "text"],
                },
            ),
            ToolDefinition(
                name="regex_replace",
                description="Replace all matches of a pattern in text with a replacement string.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regular expression pattern"},
                        "replacement": {
                            "type": "string",
                            "description": "Replacement string (supports \\1, \\2 backreferences)",
                        },
                        "text": {"type": "string", "description": "Text to transform"},
                        "flags": {
                            "type": "string",
                            "description": "Flags: i=ignorecase, m=multiline, s=dotall",
                        },
                    },
                    "required": ["pattern", "replacement", "text"],
                },
            ),
            ToolDefinition(
                name="regex_split",
                description="Split text by a regex pattern.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Delimiter pattern"},
                        "text": {"type": "string", "description": "Text to split"},
                        "maxsplit": {
                            "type": "integer",
                            "description": "Max splits (0=unlimited, default)",
                        },
                    },
                    "required": ["pattern", "text"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "regex_test":
                return self._test(
                    arguments["pattern"], arguments["text"], arguments.get("flags", "")
                )
            case "regex_findall":
                return self._findall(
                    arguments["pattern"], arguments["text"], arguments.get("flags", "")
                )
            case "regex_replace":
                return self._replace(
                    arguments["pattern"],
                    arguments["replacement"],
                    arguments["text"],
                    arguments.get("flags", ""),
                )
            case "regex_split":
                return self._split(
                    arguments["pattern"], arguments["text"], arguments.get("maxsplit", 0)
                )
            case _:
                return f"Unknown tool: {tool_name}"

    def _parse_flags(self, flags_str: str) -> int:
        f = 0
        for c in flags_str.lower():
            if c == "i":
                f |= re.IGNORECASE
            elif c == "m":
                f |= re.MULTILINE
            elif c == "s":
                f |= re.DOTALL
        return f

    def _test(self, pattern: str, text: str, flags: str) -> str:
        try:
            regex = re.compile(pattern, self._parse_flags(flags))
        except re.error as e:
            return f"Invalid regex: {e}"

        m = regex.search(text)
        if not m:
            return "No match."

        lines = [f"Match: '{m.group()}'", f"Position: {m.start()}-{m.end()}"]
        if m.groups():
            for i, g in enumerate(m.groups(), 1):
                lines.append(f"  Group {i}: '{g}'")
        if m.groupdict():
            for name, val in m.groupdict().items():
                lines.append(f"  Named '{name}': '{val}'")
        return "\n".join(lines)

    def _findall(self, pattern: str, text: str, flags: str) -> str:
        try:
            regex = re.compile(pattern, self._parse_flags(flags))
        except re.error as e:
            return f"Invalid regex: {e}"

        matches = list(regex.finditer(text))
        if not matches:
            return "No matches found."

        lines = [f"Found {len(matches)} match(es):"]
        for i, m in enumerate(matches[:50]):
            lines.append(f"  {i + 1}. '{m.group()}' at {m.start()}-{m.end()}")
        if len(matches) > 50:
            lines.append(f"  ... and {len(matches) - 50} more")
        return "\n".join(lines)

    def _replace(self, pattern: str, replacement: str, text: str, flags: str) -> str:
        try:
            regex = re.compile(pattern, self._parse_flags(flags))
        except re.error as e:
            return f"Invalid regex: {e}"

        result, count = regex.subn(replacement, text)
        if count == 0:
            return "No matches to replace."
        return f"Replaced {count} match(es):\n{result[:10000]}"

    def _split(self, pattern: str, text: str, maxsplit: int) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex: {e}"

        parts = regex.split(text, maxsplit=maxsplit)
        lines = [f"Split into {len(parts)} part(s):"]
        for i, p in enumerate(parts[:50]):
            preview = p[:100].replace("\n", "\\n")
            if len(p) > 100:
                preview += "..."
            lines.append(f"  {i + 1}. '{preview}'")
        return "\n".join(lines)

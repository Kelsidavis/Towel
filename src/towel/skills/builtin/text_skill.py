"""Text skill — word count, line stats, character frequency, text transforms."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class TextSkill(Skill):
    @property
    def name(self) -> str:
        return "text"

    @property
    def description(self) -> str:
        return "Text analysis and transformation — word count, stats, transforms"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="text_stats",
                description="Get statistics about text: characters, words, lines, sentences, reading time",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze"},
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="text_transform",
                description="Transform text: uppercase, lowercase, title case, snake_case, camelCase, reverse",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to transform"},
                        "transform": {
                            "type": "string",
                            "enum": ["upper", "lower", "title", "snake", "camel", "reverse", "sort_lines", "unique_lines", "number_lines"],
                            "description": "Transformation to apply",
                        },
                    },
                    "required": ["text", "transform"],
                },
            ),
            ToolDefinition(
                name="text_frequency",
                description="Get word or character frequency analysis",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze"},
                        "mode": {"type": "string", "enum": ["words", "chars"], "description": "Count words or characters (default: words)"},
                        "top": {"type": "integer", "description": "Number of top items (default: 20)"},
                    },
                    "required": ["text"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "text_stats":
                return self._stats(arguments["text"])
            case "text_transform":
                return self._transform(arguments["text"], arguments["transform"])
            case "text_frequency":
                return self._frequency(arguments["text"], arguments.get("mode", "words"), arguments.get("top", 20))
            case _:
                return f"Unknown tool: {tool_name}"

    def _stats(self, text: str) -> str:
        chars = len(text)
        chars_no_space = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        words = len(text.split())
        lines = text.count("\n") + (1 if text else 0)
        sentences = len(re.findall(r"[.!?]+", text))
        paragraphs = len([p for p in text.split("\n\n") if p.strip()])
        read_min = words / 200  # avg reading speed

        return (
            f"Text statistics:\n"
            f"  Characters: {chars:,} ({chars_no_space:,} without spaces)\n"
            f"  Words: {words:,}\n"
            f"  Lines: {lines:,}\n"
            f"  Sentences: {sentences:,}\n"
            f"  Paragraphs: {paragraphs:,}\n"
            f"  Reading time: ~{read_min:.1f} min"
        )

    def _transform(self, text: str, transform: str) -> str:
        match transform:
            case "upper": return text.upper()
            case "lower": return text.lower()
            case "title": return text.title()
            case "snake":
                s = re.sub(r"([A-Z])", r"_\1", text).lower()
                return re.sub(r"[^a-z0-9]+", "_", s).strip("_")
            case "camel":
                words = re.split(r"[_\s-]+", text)
                return words[0].lower() + "".join(w.capitalize() for w in words[1:])
            case "reverse": return text[::-1]
            case "sort_lines": return "\n".join(sorted(text.splitlines()))
            case "unique_lines":
                seen: set[str] = set()
                result = []
                for line in text.splitlines():
                    if line not in seen:
                        seen.add(line)
                        result.append(line)
                return "\n".join(result)
            case "number_lines":
                return "\n".join(f"{i+1:4d}  {line}" for i, line in enumerate(text.splitlines()))
            case _: return f"Unknown transform: {transform}"

    def _frequency(self, text: str, mode: str, top: int) -> str:
        if mode == "chars":
            counts = Counter(c for c in text if not c.isspace())
        else:
            counts = Counter(re.findall(r"\b\w+\b", text.lower()))

        if not counts:
            return "No data to analyze."

        items = counts.most_common(top)
        lines = [f"Top {len(items)} {mode}:"]
        max_count = items[0][1]
        for item, count in items:
            bar = "█" * int(count / max_count * 20)
            lines.append(f"  {item:>15s}  {count:>5d}  {bar}")
        return "\n".join(lines)

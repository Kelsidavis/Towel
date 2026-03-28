"""Template engine — discovers, loads, and renders prompt templates.

Templates are text files in ~/.towel/templates/ with {{variable}} placeholders.
Built-in templates ship with Towel. User templates override built-ins.

Template format:
    First line starting with # is the description (stripped from output).
    {{input}} — replaced with the user's input text
    {{file}} — replaced with @file-expanded content
    {{lang}} — replaced with --var lang=value
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.templates")

TEMPLATES_DIR = TOWEL_HOME / "templates"

# Built-in templates
BUILTIN_TEMPLATES: dict[str, str] = {
    "review": (
        "# Code review — analyze code for bugs, style, and improvements\n"
        "Review the following code. Identify bugs, suggest improvements, "
        "and rate the overall quality (1-10). Be specific and constructive.\n\n"
        "{{input}}"
    ),
    "explain": (
        "# Explain code or concepts in plain language\n"
        "Explain the following in clear, simple terms. "
        "Use analogies where helpful. Assume the reader is a developer "
        "but may not know this specific domain.\n\n"
        "{{input}}"
    ),
    "summarize": (
        "# Summarize text concisely\n"
        "Summarize the following text. Keep the key points and important details. "
        "Be concise but don't lose critical information.\n\n"
        "{{input}}"
    ),
    "translate": (
        "# Translate text to another language\n"
        "Translate the following to {{lang|English}}. "
        "Preserve tone and meaning. Only output the translation.\n\n"
        "{{input}}"
    ),
    "fix": (
        "# Fix bugs in code\n"
        "The following code has a bug. Find it, explain what's wrong, "
        "and provide the corrected version.\n\n"
        "{{input}}"
    ),
    "test": (
        "# Generate tests for code\n"
        "Write comprehensive tests for the following code. "
        "Cover edge cases, error conditions, and typical usage. "
        "Use the appropriate test framework for the language.\n\n"
        "{{input}}"
    ),
    "commit": (
        "# Generate a git commit message\n"
        "Write a concise, descriptive git commit message for the following diff. "
        "Use conventional commits format (feat/fix/refactor/docs/test). "
        "First line under 72 chars, then a blank line, then details if needed.\n\n"
        "{{input}}"
    ),
    "refactor": (
        "# Refactor code for clarity and maintainability\n"
        "Refactor the following code to improve readability, reduce complexity, "
        "and follow best practices. Explain your changes.\n\n"
        "{{input}}"
    ),
}


class TemplateEngine:
    """Discovers and renders prompt templates."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        self.templates_dir = templates_dir or TEMPLATES_DIR

    def list_templates(self) -> dict[str, str]:
        """Return {name: description} for all available templates."""
        result: dict[str, str] = {}

        # Built-ins first
        for name, content in BUILTIN_TEMPLATES.items():
            result[name] = self._extract_description(content)

        # User templates override
        if self.templates_dir.is_dir():
            for f in sorted(self.templates_dir.glob("*.txt")):
                name = f.stem
                content = f.read_text(encoding="utf-8", errors="replace")
                result[name] = self._extract_description(content)

        return result

    def get(self, name: str) -> str | None:
        """Get a template's raw content by name."""
        # User template takes priority
        user_file = self.templates_dir / f"{name}.txt"
        if user_file.is_file():
            return user_file.read_text(encoding="utf-8", errors="replace")

        return BUILTIN_TEMPLATES.get(name)

    def render(
        self,
        name: str,
        input_text: str = "",
        variables: dict[str, str] | None = None,
    ) -> str | None:
        """Render a template with the given input and variables.

        Returns None if the template doesn't exist.
        """
        raw = self.get(name)
        if raw is None:
            return None

        # Strip description line
        lines = raw.splitlines()
        content_lines = [l for l in lines if not l.startswith("#")]
        content = "\n".join(content_lines).strip()

        # Replace {{input}}
        content = content.replace("{{input}}", input_text)

        # Replace {{var}} and {{var|default}}
        vars_ = variables or {}

        def _replace_var(match: re.Match[str]) -> str:
            expr = match.group(1)
            if "|" in expr:
                var_name, default = expr.split("|", 1)
                return vars_.get(var_name.strip(), default.strip())
            return vars_.get(expr.strip(), match.group(0))

        content = re.sub(r"\{\{(\w+(?:\|[^}]*)?)\}\}", _replace_var, content)

        return content

    def _extract_description(self, content: str) -> str:
        """Extract the # description line from a template."""
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return ""

"""Lint skill — basic code style checks without external tools."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_CHECKS = [
    ("trailing_whitespace", re.compile(r" +$", re.MULTILINE), "Trailing whitespace"),
    ("tabs", re.compile(r"\t"), "Tab characters (use spaces)"),
    ("long_lines", re.compile(r"^.{121,}$", re.MULTILINE), "Line exceeds 120 chars"),
    ("debug_print", re.compile(r"\bprint\s*\("), "print() statement (use logging?)"),
    ("todo", re.compile(r"(?i)\bTODO\b"), "TODO comment"),
    ("fixme", re.compile(r"(?i)\bFIXME\b"), "FIXME comment"),
    ("bare_except", re.compile(r"except\s*:"), "Bare except clause"),
    ("import_star", re.compile(r"from\s+\S+\s+import\s+\*"), "Wildcard import"),
    (
        "hardcoded_ip",
        re.compile(r"\b(?:192\.168|10\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b"),
        "Hardcoded private IP",
    ),
    ("console_log", re.compile(r"console\.log\s*\("), "console.log() (remove before prod)"),
]


class LintSkill(Skill):
    @property
    def name(self) -> str:
        return "lint"

    @property
    def description(self) -> str:
        return "Basic code style and quality checks without external tools"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="lint_file",
                description="Run basic lint checks on a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File to check"},
                        "checks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific checks (default: all)",
                        },
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="lint_text",
                description="Run lint checks on text/code directly",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to check"},
                        "language": {
                            "type": "string",
                            "description": "Language hint (python, javascript, etc.)",
                        },
                    },
                    "required": ["code"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "lint_file":
                return self._lint_file(arguments["path"], arguments.get("checks"))
            case "lint_text":
                return self._lint_text(arguments["code"], arguments.get("language", ""))
            case _:
                return f"Unknown tool: {tool_name}"

    def _lint_file(self, path: str, checks: list[str] | None) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Not found: {path}"
        code = p.read_text(encoding="utf-8", errors="replace")
        return self._run_checks(code, p.name, checks)

    def _lint_text(self, code: str, language: str) -> str:
        return self._run_checks(code, f"<{language or 'code'}>", None)

    def _run_checks(self, code: str, name: str, filter_checks: list[str] | None) -> str:
        findings: list[str] = []
        lines = code.splitlines()
        for check_name, pattern, desc in _CHECKS:
            if filter_checks and check_name not in filter_checks:
                continue
            for i, line in enumerate(lines):
                if pattern.search(line):
                    findings.append(f"  {name}:{i + 1}: [{check_name}] {desc}")
        if not findings:
            return f"No issues found in {name} ({len(lines)} lines checked)."
        return f"{len(findings)} issue(s) in {name}:\n" + "\n".join(findings[:50])

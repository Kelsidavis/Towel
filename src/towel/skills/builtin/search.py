"""Search skill — recursive file content search (grep-like)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

MAX_RESULTS = 50
MAX_LINE_LEN = 300
BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
                     ".ttf", ".eot", ".zip", ".tar", ".gz", ".bin", ".exe",
                     ".pyc", ".pyo", ".so", ".dylib", ".o", ".class"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox",
             "dist", "build", ".eggs", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


class SearchSkill(Skill):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search file contents recursively (grep-like)"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_files",
                description="Search for a pattern across files in a directory. Returns matching lines with context.",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern (regex supported)"},
                        "path": {"type": "string", "description": "Directory to search (default: cwd)"},
                        "glob": {"type": "string", "description": "File glob filter (e.g., '*.py', '*.ts')"},
                        "context": {"type": "integer", "description": "Lines of context around matches (default: 2)"},
                        "case_sensitive": {"type": "boolean", "description": "Case sensitive (default: false)"},
                    },
                    "required": ["pattern"],
                },
            ),
            ToolDefinition(
                name="find_files",
                description="Find files by name pattern in a directory tree",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern (e.g., '*.py', 'test_*.py', '**/*.tsx')"},
                        "path": {"type": "string", "description": "Directory to search (default: cwd)"},
                    },
                    "required": ["pattern"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "search_files":
                return await self._search(
                    arguments["pattern"],
                    arguments.get("path", "."),
                    arguments.get("glob"),
                    arguments.get("context", 2),
                    arguments.get("case_sensitive", False),
                )
            case "find_files":
                return self._find(arguments["pattern"], arguments.get("path", "."))
            case _:
                return f"Unknown tool: {tool_name}"

    async def _search(
        self, pattern: str, path: str, file_glob: str | None,
        context: int, case_sensitive: bool,
    ) -> str:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex: {e}"

        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return f"Not a directory: {root}"

        results: list[str] = []
        files_searched = 0
        match_count = 0

        for fpath in self._walk_files(root, file_glob):
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            files_searched += 1
            file_matches: list[str] = []

            for i, line in enumerate(lines):
                if regex.search(line):
                    # Gather context lines
                    start = max(0, i - context)
                    end = min(len(lines), i + context + 1)
                    for j in range(start, end):
                        prefix = ">>" if j == i else "  "
                        text = lines[j][:MAX_LINE_LEN]
                        file_matches.append(f"  {prefix} {j+1}: {text}")
                    if context > 0:
                        file_matches.append("")
                    match_count += 1

                    if match_count >= MAX_RESULTS:
                        break

            if file_matches:
                rel = fpath.relative_to(root) if fpath.is_relative_to(root) else fpath
                results.append(f"{rel}:")
                results.extend(file_matches)

            if match_count >= MAX_RESULTS:
                break

        if not results:
            return f"No matches for '{pattern}' in {files_searched} files."

        header = f"Found {match_count} match(es) in {files_searched} files searched:"
        if match_count >= MAX_RESULTS:
            header += f" (limited to {MAX_RESULTS})"
        return header + "\n\n" + "\n".join(results)

    def _find(self, pattern: str, path: str) -> str:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return f"Not a directory: {root}"

        matches: list[str] = []
        for fpath in sorted(root.rglob(pattern)):
            if any(skip in fpath.parts for skip in SKIP_DIRS):
                continue
            rel = fpath.relative_to(root) if fpath.is_relative_to(root) else fpath
            kind = "d" if fpath.is_dir() else "f"
            matches.append(f"  {kind} {rel}")
            if len(matches) >= 100:
                matches.append("  ... (truncated at 100)")
                break

        if not matches:
            return f"No files matching '{pattern}' in {root}"
        return f"Found {len(matches)} match(es):\n" + "\n".join(matches)

    def _walk_files(self, root: Path, file_glob: str | None):
        """Yield files, respecting skip dirs and binary extensions."""
        if file_glob:
            for fpath in sorted(root.rglob(file_glob)):
                if any(skip in fpath.parts for skip in SKIP_DIRS):
                    continue
                if fpath.is_file() and fpath.suffix not in BINARY_EXTENSIONS:
                    yield fpath
        else:
            for fpath in sorted(root.rglob("*")):
                if any(skip in fpath.parts for skip in SKIP_DIRS):
                    continue
                if fpath.is_file() and fpath.suffix not in BINARY_EXTENSIONS:
                    yield fpath

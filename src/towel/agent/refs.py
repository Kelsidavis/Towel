"""@ file references — expand @path tokens in user messages.

Syntax:
    @path/to/file        Inject the full file content
    @path/to/file:10-20  Inject lines 10-20 only
    @dir/*.py            Inject all matching files (glob)
    @url                 Ignored (not a file reference)

References are expanded inline, replacing the @token with a fenced
code block containing the file content. This is faster than tool calls
because the content goes straight into the user message.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger("towel.agent.refs")

# Match @path tokens — must start with ./ or ../ or / or look like a file path
# Excludes things that look like email addresses or URLs
_REF_PATTERN = re.compile(
    r"(?<!\w)@((?:[./~*][\w./~*?-]*|[\w][\w./~*?-]*\.[\w]+)(?::(\d+)(?:-(\d+))?)?)",
)

MAX_FILE_SIZE = 500_000  # 500KB per file
MAX_TOTAL_INJECT = 1_000_000  # 1MB total injection per message
MAX_GLOB_FILES = 20


class FileRef(NamedTuple):
    """A parsed @ file reference."""

    raw: str  # the full match including @
    path: str  # the file path or glob pattern
    line_start: int | None  # optional start line
    line_end: int | None  # optional end line


def parse_refs(text: str) -> list[FileRef]:
    """Extract @file references from a message."""
    refs: list[FileRef] = []
    for match in _REF_PATTERN.finditer(text):
        full = match.group(1)
        # Split off line range if present
        path_part = full.split(":")[0]
        line_start = int(match.group(2)) if match.group(2) else None
        line_end = int(match.group(3)) if match.group(3) else None
        refs.append(FileRef(
            raw=f"@{full}",
            path=path_part,
            line_start=line_start,
            line_end=line_end,
        ))
    return refs


def expand_refs(text: str) -> str:
    """Expand all @file references in a message, replacing them with file content."""
    refs = parse_refs(text)
    if not refs:
        return text

    total_injected = 0
    result = text

    for ref in refs:
        expanded = _resolve_ref(ref)
        if expanded is None:
            continue  # leave the @token as-is if resolution fails

        total_injected += len(expanded)
        if total_injected > MAX_TOTAL_INJECT:
            expanded = "[File content truncated — total injection limit reached]"

        result = result.replace(ref.raw, expanded, 1)

    return result


def _resolve_ref(ref: FileRef) -> str | None:
    """Resolve a single @file reference to its content block."""
    path_str = ref.path
    expanded = Path(path_str).expanduser()

    # Handle glob patterns
    if "*" in path_str or "?" in path_str:
        return _resolve_glob(expanded, ref)

    # Handle single file
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded

    if not expanded.is_file():
        return None  # not a file — leave the @token alone

    return _read_file(expanded, ref.line_start, ref.line_end)


def _resolve_glob(pattern_path: Path, ref: FileRef) -> str | None:
    """Resolve a glob pattern to multiple file contents."""
    if not pattern_path.is_absolute():
        base = Path.cwd()
    else:
        base = pattern_path.parent
        pattern_path = Path(pattern_path.name)

    matches = sorted(base.glob(str(pattern_path)))[:MAX_GLOB_FILES]
    if not matches:
        return None

    blocks: list[str] = []
    for path in matches:
        if path.is_file():
            content = _read_file(path, ref.line_start, ref.line_end)
            if content:
                blocks.append(content)

    if not blocks:
        return None

    if len(matches) > MAX_GLOB_FILES:
        blocks.append(f"\n[... and {len(matches) - MAX_GLOB_FILES} more files]")

    return "\n\n".join(blocks)


def _read_file(path: Path, line_start: int | None, line_end: int | None) -> str | None:
    """Read a file (or a line range) and return a fenced code block."""
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return f"\n**{path.name}** (too large — {path.stat().st_size} bytes, max {MAX_FILE_SIZE})\n"

        content = path.read_text(encoding="utf-8", errors="replace")

        # Apply line range if specified
        if line_start is not None:
            lines = content.splitlines()
            start = max(0, line_start - 1)  # 1-indexed to 0-indexed
            end = line_end if line_end else start + 1
            content = "\n".join(lines[start:end])
            range_info = f" (lines {line_start}-{end})"
        else:
            range_info = ""

        # Detect language from extension for syntax highlighting
        ext = path.suffix.lstrip(".")
        lang = _ext_to_lang(ext)

        return f"\n**{path.name}**{range_info}:\n```{lang}\n{content}\n```"

    except OSError as e:
        log.warning(f"Failed to read {path}: {e}")
        return None


def _ext_to_lang(ext: str) -> str:
    """Map file extension to markdown code fence language."""
    mapping = {
        "py": "python", "js": "javascript", "ts": "typescript",
        "tsx": "tsx", "jsx": "jsx", "rs": "rust", "go": "go",
        "c": "c", "h": "c", "cpp": "cpp", "hpp": "cpp",
        "java": "java", "rb": "ruby", "sh": "bash", "zsh": "bash",
        "yml": "yaml", "yaml": "yaml", "toml": "toml", "json": "json",
        "md": "markdown", "html": "html", "css": "css", "sql": "sql",
        "swift": "swift", "kt": "kotlin", "r": "r",
    }
    return mapping.get(ext, ext)

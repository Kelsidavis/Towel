"""@ references — expand @path and @url tokens in user messages.

Syntax:
    @path/to/file        Inject the full file content
    @path/to/file:10-20  Inject lines 10-20 only
    @dir/*.py            Inject all matching files (glob)
    @https://example.com Fetch URL content and inject inline

References are expanded inline, replacing the @token with a fenced
code block containing the content. This is faster than tool calls
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
MAX_URL_SIZE = 200_000  # 200KB per URL fetch
MAX_TOTAL_INJECT = 1_000_000  # 1MB total injection per message
MAX_GLOB_FILES = 20
URL_TIMEOUT = 10  # seconds

# Match @https://... and @http://... URLs
_URL_PATTERN = re.compile(
    r"(?<!\w)@(https?://[^\s\])<>\"']+)",
)


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
        refs.append(
            FileRef(
                raw=f"@{full}",
                path=path_part,
                line_start=line_start,
                line_end=line_end,
            )
        )
    return refs


def expand_refs(text: str) -> str:
    """Expand all @file and @url references in a message."""
    # Expand URLs first (so they don't get matched as file paths)
    url_matches = list(_URL_PATTERN.finditer(text))
    total_injected = 0
    result = text

    for match in url_matches:
        url = match.group(1)
        token = f"@{url}"
        fetched = _fetch_url(url)
        if fetched is None:
            continue

        total_injected += len(fetched)
        if total_injected > MAX_TOTAL_INJECT:
            fetched = "[Content truncated — total injection limit reached]"

        result = result.replace(token, fetched, 1)

    # Then expand file refs
    refs = parse_refs(result)
    for ref in refs:
        expanded = _resolve_ref(ref)
        if expanded is None:
            continue

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
            size = path.stat().st_size
            return (
                f"\n**{path.name}** (too large — "
                f"{size} bytes, max {MAX_FILE_SIZE})\n"
            )

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


def _fetch_url(url: str) -> str | None:
    """Fetch a URL and return its content as a fenced block."""
    try:
        import httpx as _httpx
    except ImportError:
        log.warning("httpx not installed — cannot expand @url references")
        return None

    try:
        with _httpx.Client(timeout=URL_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Towel/1.0"})
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            body = resp.text[:MAX_URL_SIZE]

            if len(resp.text) > MAX_URL_SIZE:
                body += f"\n\n[... truncated at {MAX_URL_SIZE} bytes]"

            # Guess language from URL or content-type
            lang = ""
            if "json" in content_type or url.endswith(".json"):
                lang = "json"
            elif "yaml" in content_type or url.endswith((".yml", ".yaml")):
                lang = "yaml"
            elif "html" in content_type:
                lang = "html"
            elif "xml" in content_type or url.endswith(".xml"):
                lang = "xml"
            elif url.endswith(".py"):
                lang = "python"
            elif url.endswith(".js"):
                lang = "javascript"
            elif url.endswith(".ts"):
                lang = "typescript"
            elif url.endswith((".md", ".markdown")):
                lang = "markdown"

            # Shorten display URL
            display = url if len(url) <= 80 else url[:77] + "..."

            return f"\n**{display}**:\n```{lang}\n{body}\n```"

    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return f"\n[Failed to fetch {url}: {e}]\n"


def _ext_to_lang(ext: str) -> str:
    """Map file extension to markdown code fence language."""
    mapping = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "jsx": "jsx",
        "rs": "rust",
        "go": "go",
        "c": "c",
        "h": "c",
        "cpp": "cpp",
        "hpp": "cpp",
        "java": "java",
        "rb": "ruby",
        "sh": "bash",
        "zsh": "bash",
        "yml": "yaml",
        "yaml": "yaml",
        "toml": "toml",
        "json": "json",
        "md": "markdown",
        "html": "html",
        "css": "css",
        "sql": "sql",
        "swift": "swift",
        "kt": "kotlin",
        "r": "r",
    }
    return mapping.get(ext, ext)

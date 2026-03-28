"""Project context loader — discovers and loads .towel.md files.

Walks up from the current working directory looking for .towel.md,
then loads all of them (child first, parent last) to build a
project context block for the system prompt.

Also supports .towel/ directories containing multiple context files
for larger projects.

Discovery order (all loaded, child wins on conflict):
  1. .towel.md in cwd
  2. .towel.md in parent directories (up to home or root)
  3. .towel/*.md files in the nearest .towel/ directory
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("towel.agent.project")

CONTEXT_FILENAME = ".towel.md"
CONTEXT_DIR = ".towel"
MAX_CONTEXT_BYTES = 50_000  # cap to avoid blowing up the context window


def find_project_contexts(start_dir: Path | None = None) -> list[Path]:
    """Find .towel.md files walking up from start_dir.

    Returns paths in order from most specific (deepest) to least.
    """
    start = (start_dir or Path.cwd()).resolve()
    home = Path.home().resolve()
    found: list[Path] = []

    current = start
    while True:
        # Check for .towel.md file
        candidate = current / CONTEXT_FILENAME
        if candidate.is_file():
            found.append(candidate)

        # Check for .towel/ directory with markdown files
        context_dir = current / CONTEXT_DIR
        if context_dir.is_dir():
            for md_file in sorted(context_dir.glob("*.md")):
                if md_file.is_file():
                    found.append(md_file)

        # Stop at home directory or root
        if current == home or current == current.parent:
            break
        current = current.parent

    return found


def load_project_context(start_dir: Path | None = None) -> str:
    """Load and combine all project context files into a prompt block.

    Returns empty string if no context files found.
    """
    paths = find_project_contexts(start_dir)
    if not paths:
        return ""

    sections: list[str] = []
    total_bytes = 0

    for path in paths:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue

            # Track size to avoid blowing up context
            total_bytes += len(content)
            if total_bytes > MAX_CONTEXT_BYTES:
                remaining = MAX_CONTEXT_BYTES - (total_bytes - len(content))
                if remaining > 0:
                    content = content[:remaining] + "\n\n[... truncated ...]"
                    sections.append(content)
                log.warning(f"Project context truncated at {MAX_CONTEXT_BYTES} bytes")
                break

            sections.append(content)
            log.debug(f"Loaded project context: {path} ({len(content)} bytes)")

        except OSError as e:
            log.warning(f"Failed to read {path}: {e}")

    if not sections:
        return ""

    combined = "\n\n---\n\n".join(sections)
    return f"\n\n## Project Context\nThe following context describes the project you're working in:\n\n{combined}"

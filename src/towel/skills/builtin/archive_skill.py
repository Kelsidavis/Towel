"""Archive skill — create and inspect zip/tar archives."""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

MAX_LIST = 200


class ArchiveSkill(Skill):
    @property
    def name(self) -> str:
        return "archive"

    @property
    def description(self) -> str:
        return "Create and inspect zip/tar archives"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="archive_list",
                description="List contents of a zip or tar archive",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to archive file"},
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="archive_create",
                description="Create a zip archive from files or a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "output": {"type": "string", "description": "Output archive path (e.g., backup.zip)"},
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files or directories to include",
                        },
                    },
                    "required": ["output", "sources"],
                },
            ),
            ToolDefinition(
                name="archive_extract",
                description="Extract a zip or tar archive to a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to archive"},
                        "dest": {"type": "string", "description": "Destination directory (default: current dir)"},
                    },
                    "required": ["path"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "archive_list":
                return self._list(arguments["path"])
            case "archive_create":
                return self._create(arguments["output"], arguments["sources"])
            case "archive_extract":
                return self._extract(arguments["path"], arguments.get("dest", "."))
            case _:
                return f"Unknown tool: {tool_name}"

    def _list(self, path: str) -> str:
        p = Path(path).expanduser()
        if not p.is_file():
            return f"Not found: {path}"

        entries: list[str] = []
        total_size = 0

        try:
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    for info in zf.infolist()[:MAX_LIST]:
                        size = info.file_size
                        total_size += size
                        kind = "d" if info.is_dir() else "f"
                        entries.append(f"  {kind} {info.filename}  ({size:,} bytes)")
                    count = len(zf.infolist())
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as tf:
                    members = tf.getmembers()
                    for m in members[:MAX_LIST]:
                        total_size += m.size
                        kind = "d" if m.isdir() else "f"
                        entries.append(f"  {kind} {m.name}  ({m.size:,} bytes)")
                    count = len(members)
            else:
                return f"Not a recognized archive: {path}"
        except Exception as e:
            return f"Error reading archive: {e}"

        header = f"Archive: {p.name} ({count} entries, {total_size:,} bytes total)"
        if count > MAX_LIST:
            entries.append(f"  ... and {count - MAX_LIST} more")
        return header + "\n" + "\n".join(entries)

    def _create(self, output: str, sources: list[str]) -> str:
        out = Path(output).expanduser()
        try:
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
                count = 0
                for src in sources:
                    sp = Path(src).expanduser()
                    if sp.is_file():
                        zf.write(sp, sp.name)
                        count += 1
                    elif sp.is_dir():
                        for fp in sorted(sp.rglob("*")):
                            if fp.is_file():
                                zf.write(fp, fp.relative_to(sp.parent))
                                count += 1
                    else:
                        return f"Not found: {src}"

            size = out.stat().st_size
            return f"Created: {out} ({count} files, {size:,} bytes)"
        except Exception as e:
            return f"Error creating archive: {e}"

    def _extract(self, path: str, dest: str) -> str:
        p = Path(path).expanduser()
        d = Path(dest).expanduser()
        if not p.is_file():
            return f"Not found: {path}"

        d.mkdir(parents=True, exist_ok=True)
        try:
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as zf:
                    zf.extractall(d)
                    count = len(zf.infolist())
            elif tarfile.is_tarfile(p):
                with tarfile.open(p) as tf:
                    tf.extractall(d, filter="data")
                    count = len(tf.getmembers())
            else:
                return f"Not a recognized archive: {path}"

            return f"Extracted {count} entries to {d}"
        except Exception as e:
            return f"Error extracting: {e}"

"""Persistent memory store — the agent's long-term brain.

Memories persist across sessions in ~/.towel/memory/ as JSON files.
They're automatically injected into the system prompt so the agent
always knows what it has learned about the user and their work.

Memory types:
  - user:      Facts about the user (role, preferences, expertise)
  - project:   Ongoing work, goals, deadlines
  - fact:      Learned facts the agent should remember
  - preference: How the user likes things done
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.memory")

DEFAULT_MEMORY_DIR = TOWEL_HOME / "memory"

MEMORY_TYPES = ("user", "project", "fact", "preference")


@dataclass
class MemoryEntry:
    """A single memory."""

    key: str
    content: str
    memory_type: str = "fact"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "type": self.memory_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            key=data["key"],
            content=data["content"],
            memory_type=data.get("type", "fact"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def __str__(self) -> str:
        return f"[{self.memory_type}] {self.key}: {self.content}"


class MemoryStore:
    """File-backed persistent memory store."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or DEFAULT_MEMORY_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, MemoryEntry] | None = None

    def _index_path(self) -> Path:
        return self.store_dir / "memories.json"

    def _load_all(self) -> dict[str, MemoryEntry]:
        if self._cache is not None:
            return self._cache
        path = self._index_path()
        if not path.exists():
            self._cache = {}
            return self._cache
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._cache = {
                k: MemoryEntry.from_dict(v) for k, v in data.items()
            }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning(f"Failed to load memories: {e}")
            self._cache = {}
        return self._cache

    def _save_all(self) -> None:
        entries = self._load_all()
        data = {k: v.to_dict() for k, v in entries.items()}
        self._index_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def remember(self, key: str, content: str, memory_type: str = "fact") -> MemoryEntry:
        """Store or update a memory."""
        entries = self._load_all()
        now = datetime.now(timezone.utc)

        if key in entries:
            entry = entries[key]
            entry.content = content
            entry.memory_type = memory_type
            entry.updated_at = now
        else:
            entry = MemoryEntry(
                key=key,
                content=content,
                memory_type=memory_type,
                created_at=now,
                updated_at=now,
            )
            entries[key] = entry

        self._save_all()
        log.info(f"Remembered: {key}")
        return entry

    def forget(self, key: str) -> bool:
        """Remove a memory. Returns True if it existed."""
        entries = self._load_all()
        if key in entries:
            del entries[key]
            self._save_all()
            log.info(f"Forgot: {key}")
            return True
        return False

    def recall(self, key: str) -> MemoryEntry | None:
        """Get a specific memory by key."""
        return self._load_all().get(key)

    def recall_all(self, memory_type: str | None = None) -> list[MemoryEntry]:
        """Get all memories, optionally filtered by type."""
        entries = self._load_all()
        if memory_type:
            return [e for e in entries.values() if e.memory_type == memory_type]
        return list(entries.values())

    def search(self, query: str) -> list[MemoryEntry]:
        """Search memories by key or content."""
        q = query.lower()
        return [
            e for e in self._load_all().values()
            if q in e.key.lower() or q in e.content.lower()
        ]

    def to_prompt_block(self) -> str:
        """Render all memories as a block for injection into the system prompt."""
        entries = self.recall_all()
        if not entries:
            return ""

        lines = ["\n\n## Your Memory\nYou have the following persistent memories from past sessions:\n"]

        by_type: dict[str, list[MemoryEntry]] = {}
        for e in entries:
            by_type.setdefault(e.memory_type, []).append(e)

        for mtype in ["user", "preference", "project", "fact"]:
            group = by_type.get(mtype, [])
            if not group:
                continue
            lines.append(f"\n**{mtype.title()}:**")
            for e in group:
                lines.append(f"- {e.key}: {e.content}")

        lines.append(
            "\nYou can use the `remember` and `forget` tools to update your memory. "
            "Proactively remember useful facts about the user and their work."
        )
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._load_all())

"""Conversation store — persists conversations to JSON files.

Storage layout:
    ~/.towel/conversations/
        {conversation_id}.json

Each file is a self-contained JSON document with the full conversation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from towel.agent.conversation import Conversation, Role
from towel.config import TOWEL_HOME

log = logging.getLogger("towel.persistence")

DEFAULT_STORE_DIR = TOWEL_HOME / "conversations"


class ConversationStore:
    """File-based conversation persistence."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or DEFAULT_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, conversation_id: str) -> Path:
        # Sanitize ID to prevent path traversal
        safe_id = "".join(c for c in conversation_id if c.isalnum() or c in "-_")
        return self.store_dir / f"{safe_id}.json"

    def save(self, conversation: Conversation) -> Path:
        """Save a conversation to disk. Returns the file path."""
        path = self._path_for(conversation.id)
        data = conversation.to_dict()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.debug(f"Saved conversation {conversation.id} ({len(conversation)} messages)")
        return path

    def load(self, conversation_id: str) -> Conversation | None:
        """Load a conversation by ID. Returns None if not found."""
        path = self._path_for(conversation_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Conversation.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.warning(f"Failed to load conversation {conversation_id}: {e}")
            return None

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed."""
        path = self._path_for(conversation_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def delete_all(self) -> int:
        """Delete all conversations. Returns count deleted."""
        count = 0
        for path in self.store_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count

    def list_conversations(self, limit: int = 50) -> list[ConversationSummary]:
        """List saved conversations, most recent first."""
        summaries: list[ConversationSummary] = []
        json_files = sorted(
            self.store_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        for path in json_files[:limit]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                conv = Conversation.from_dict(data)
                summaries.append(
                    ConversationSummary(
                        id=conv.id,
                        title=conv.title,
                        channel=conv.channel,
                        created_at=conv.created_at.isoformat(),
                        message_count=len(conv),
                        summary=conv.display_title,
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                log.warning(f"Skipping corrupt conversation file: {path.name}")
                continue

        return summaries

    def rename(self, conversation_id: str, title: str) -> bool:
        """Set a custom title on a conversation. Returns True on success."""
        conv = self.load(conversation_id)
        if not conv:
            return False
        conv.title = title
        self.save(conv)
        return True

    def exists(self, conversation_id: str) -> bool:
        return self._path_for(conversation_id).exists()

    @property
    def count(self) -> int:
        """Number of stored conversations."""
        return len(list(self.store_dir.glob("*.json")))

    def search(
        self,
        query: str,
        limit: int = 20,
        role_filter: Role | None = None,
        regex: bool = False,
    ) -> list[SearchResult]:
        """Search across all conversations for messages matching the query.

        Args:
            query: Search string (case-insensitive) or regex pattern.
            limit: Maximum results to return.
            role_filter: Only search messages with this role (e.g., Role.USER).
            regex: Treat query as a regex pattern.

        Returns:
            List of SearchResult, sorted by relevance (match count per conversation).
        """
        if regex:
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error as e:
                log.warning(f"Invalid regex pattern: {e}")
                return []
        else:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        results: list[SearchResult] = []
        json_files = sorted(
            self.store_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for path in json_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                conv = Conversation.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

            matches: list[SearchMatch] = []
            for msg in conv.messages:
                if role_filter and msg.role != role_filter:
                    continue
                if pattern.search(msg.content):
                    # Extract a snippet around the match
                    snippet = _extract_snippet(msg.content, pattern)
                    matches.append(
                        SearchMatch(
                            message_id=msg.id,
                            role=msg.role.value,
                            snippet=snippet,
                            timestamp=msg.timestamp.isoformat(),
                        )
                    )

            if matches:
                results.append(
                    SearchResult(
                        conversation_id=conv.id,
                        channel=conv.channel,
                        created_at=conv.created_at.isoformat(),
                        summary=conv.summary,
                        matches=matches,
                    )
                )

            if len(results) >= limit:
                break

        # Sort by match count (most relevant first)
        results.sort(key=lambda r: len(r.matches), reverse=True)
        return results


def _extract_snippet(text: str, pattern: re.Pattern[str], context_chars: int = 80) -> str:
    """Extract a text snippet around the first match."""
    match = pattern.search(text)
    if not match:
        return text[:160]

    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)

    snippet = text[start:end].replace("\n", " ")
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


class SearchMatch:
    """A single matching message within a conversation."""

    def __init__(self, message_id: str, role: str, snippet: str, timestamp: str) -> None:
        self.message_id = message_id
        self.role = role
        self.snippet = snippet
        self.timestamp = timestamp


class SearchResult:
    """Search result — a conversation with matching messages."""

    def __init__(
        self,
        conversation_id: str,
        channel: str,
        created_at: str,
        summary: str,
        matches: list[SearchMatch],
    ) -> None:
        self.conversation_id = conversation_id
        self.channel = channel
        self.created_at = created_at
        self.summary = summary
        self.matches = matches


class ConversationSummary:
    """Lightweight summary of a stored conversation."""

    def __init__(
        self,
        id: str,
        channel: str,
        created_at: str,
        message_count: int,
        summary: str,
        title: str = "",
    ) -> None:
        self.id = id
        self.title = title
        self.channel = channel
        self.created_at = created_at
        self.message_count = message_count
        self.summary = summary  # display_title (title if set, else auto)

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
from datetime import UTC, datetime
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
        """Save a conversation to disk. Returns the file path.

        Atomic write: dumps to a sibling .tmp then renames. Without
        this, a kill / disk-full mid-write leaves a half-written
        JSON that load() rejects with JSONDecodeError — the
        conversation then appears empty / missing on next read.
        Same pattern memory/store.py adopted in 5512834.
        """
        path = self._path_for(conversation.id)
        data = conversation.to_dict()
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.debug("Saved conversation %s (%d messages)", conversation.id, len(conversation))
        return path

    def load(self, conversation_id: str) -> Conversation | None:
        """Load a conversation by ID. Returns None if not found.

        On corruption, renames the bad file to a sibling
        ``.corrupted-<timestamp>`` before returning None. Without
        this, the next save() for this id would overwrite the
        corrupt file with fresh empty content — silently destroying
        whatever was there. Same pattern memory/store.py adopted
        in 5512834.
        """
        path = self._path_for(conversation_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Conversation.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            from datetime import UTC, datetime
            backup = path.with_name(
                f"{path.name}.corrupted-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
            )
            try:
                path.replace(backup)
                log.warning(
                    "Failed to load conversation %s: %s. Backed up the "
                    "bad file to %s — the next save would otherwise "
                    "overwrite it with fresh empty content.",
                    conversation_id, e, backup,
                )
            except OSError as rename_exc:
                log.warning(
                    "Failed to load conversation %s: %s. Also failed to "
                    "back up the corrupt file (%s) — data may be lost on "
                    "the next save.",
                    conversation_id, e, rename_exc,
                )
            return None

    def resolve_id(self, prefix: str) -> str | None:
        """Resolve a conversation ID prefix to a full ID.

        Returns the full ID if exactly one conversation matches, or
        None if zero or more than one match.  Exact matches always win.
        """
        exact = self._path_for(prefix)
        if exact.exists():
            return prefix
        matches = [
            p.stem for p in self.store_dir.glob("*.json")
            if p.stem.startswith(prefix)
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation. Returns True if it existed.

        Also unlinks the corresponding ``.json.tmp`` sibling if one
        was left behind by an interrupted atomic save — otherwise
        the orphan persists invisibly (list_conversations globs for
        ``*.json``, not ``*.json.tmp``) until the next bulk delete.
        Same housekeeping delete_all does (commit 0595b39).
        """
        path = self._path_for(conversation_id)
        tmp = path.with_name(path.name + ".tmp")
        existed = path.exists()
        if existed:
            path.unlink()
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return existed

    def delete_all(self) -> int:
        """Delete all conversations. Returns count deleted.

        Also cleans up any leaked ``*.json.tmp`` files from interrupted
        atomic saves — those are otherwise invisible to the
        conversation list (the glob excludes them) and accumulate
        forever after a kill/disk-full event. Not counted in the
        returned total since they were never readable conversations.
        """
        count = 0
        for path in self.store_dir.glob("*.json"):
            path.unlink()
            count += 1
        for tmp in self.store_dir.glob("*.json.tmp"):
            tmp.unlink()
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
                mtime = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=UTC
                ).isoformat()
                summaries.append(
                    ConversationSummary(
                        id=conv.id,
                        title=conv.title,
                        channel=conv.channel,
                        created_at=conv.created_at.isoformat(),
                        updated_at=mtime,
                        message_count=len(conv),
                        summary=conv.display_title,
                        tags=conv.tags,
                    )
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                log.warning("Skipping corrupt conversation file: %s", path.name)
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
                log.warning("Invalid regex pattern: %s", e)
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
                        title=conv.title or "",
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
        title: str = "",
    ) -> None:
        self.conversation_id = conversation_id
        self.channel = channel
        self.created_at = created_at
        self.summary = summary
        self.matches = matches
        # Title is the operator-visible name in the conversations
        # list. Search results that omit it forced UIs to fall back
        # to the conversation_id (e.g. "openai-chatcmpl-abc123") in
        # the results panel — useless for browsing. Default to ""
        # for backwards compatibility with callers that construct
        # SearchResult directly.
        self.title = title


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
        tags: list[str] | None = None,
        updated_at: str = "",
    ) -> None:
        self.id = id
        self.title = title
        self.channel = channel
        self.created_at = created_at
        self.updated_at = updated_at
        self.message_count = message_count
        self.summary = summary  # display_title (title if set, else auto)
        # Tags hoisted to the summary so callers (e.g. /api/sessions)
        # don't have to re-read each conversation JSON to render the
        # session list — observed in profiling: 50 sessions × one
        # file-read each per /api/sessions call.
        self.tags = list(tags) if tags else []

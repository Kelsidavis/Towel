"""Incremental context synchronization for LAN cluster workers.

Instead of sending the entire conversation JSON on every job, delta sync
tracks which messages each worker has already seen and sends only the new
ones. This dramatically reduces WebSocket traffic for long conversations.

Protocol:
  1. Controller tracks a cursor (last synced message ID) per worker+session pair
  2. On job dispatch, controller sends only messages after the cursor
  3. Worker applies the delta to its local conversation copy
  4. On job completion, worker sends back only the new/updated messages
  5. Controller merges the delta and advances the cursor

Falls back to full sync when:
  - Worker has never seen this conversation (cursor is empty)
  - Conversation was compacted (messages before the cursor were removed)
  - Worker explicitly requests full sync (e.g., after restart)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from towel.agent.conversation import Conversation, Message

log = logging.getLogger("towel.gateway.context_sync")


@dataclass
class SyncCursor:
    """Tracks what a worker has already seen for a given session."""

    worker_id: str
    session_id: str
    last_message_id: str = ""
    last_message_index: int = -1
    conversation_hash: str = ""
    synced_message_count: int = 0


@dataclass
class ConversationDelta:
    """A patch representing changes to a conversation since a cursor."""

    session_id: str
    new_messages: list[dict[str, Any]] = field(default_factory=list)
    updated_messages: list[dict[str, Any]] = field(default_factory=list)
    removed_message_ids: list[str] = field(default_factory=list)
    is_full_sync: bool = False
    base_message_id: str = ""
    conversation_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return (
            not self.new_messages
            and not self.updated_messages
            and not self.removed_message_ids
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "is_full_sync": self.is_full_sync,
            "base_message_id": self.base_message_id,
        }
        if self.new_messages:
            d["new_messages"] = self.new_messages
        if self.updated_messages:
            d["updated_messages"] = self.updated_messages
        if self.removed_message_ids:
            d["removed_message_ids"] = self.removed_message_ids
        if self.conversation_metadata:
            d["conversation_metadata"] = self.conversation_metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationDelta:
        return cls(
            session_id=data.get("session_id", ""),
            new_messages=data.get("new_messages", []),
            updated_messages=data.get("updated_messages", []),
            removed_message_ids=data.get("removed_message_ids", []),
            is_full_sync=data.get("is_full_sync", False),
            base_message_id=data.get("base_message_id", ""),
            conversation_metadata=data.get("conversation_metadata", {}),
        )


def _conversation_hash(conversation: Conversation) -> str:
    """Quick hash of message IDs to detect structural changes."""
    ids = "|".join(m.id for m in conversation.messages)
    return hashlib.md5(ids.encode()).hexdigest()[:12]


class ContextSyncManager:
    """Manages incremental context synchronization between controller and workers.

    The controller maintains one SyncCursor per (worker_id, session_id) pair.
    When dispatching a job, it computes a delta from the cursor. When the
    job completes, it merges the worker's response delta and advances the cursor.
    """

    def __init__(self) -> None:
        # Key: (worker_id, session_id)
        self._cursors: dict[tuple[str, str], SyncCursor] = {}

    def get_cursor(self, worker_id: str, session_id: str) -> SyncCursor | None:
        return self._cursors.get((worker_id, session_id))

    def _set_cursor(self, cursor: SyncCursor) -> None:
        self._cursors[(cursor.worker_id, cursor.session_id)] = cursor

    def clear_worker(self, worker_id: str) -> int:
        """Remove all cursors for a worker (e.g., on disconnect). Returns count removed."""
        keys = [k for k in self._cursors if k[0] == worker_id]
        for k in keys:
            del self._cursors[k]
        return len(keys)

    def clear_session(self, session_id: str) -> int:
        """Remove all cursors for a session (e.g., on delete). Returns count removed."""
        keys = [k for k in self._cursors if k[1] == session_id]
        for k in keys:
            del self._cursors[k]
        return len(keys)

    def compute_delta(
        self,
        worker_id: str,
        session_id: str,
        conversation: Conversation,
    ) -> ConversationDelta:
        """Compute the delta a worker needs to catch up to the current conversation.

        Returns a ConversationDelta with is_full_sync=True if the worker has
        never seen this conversation or if the conversation structure has changed
        in ways that can't be expressed as appends (compaction, edits).
        """
        cursor = self._cursors.get((worker_id, session_id))
        conv_hash = _conversation_hash(conversation)
        metadata = {
            "id": conversation.id,
            "title": conversation.title,
            "channel": conversation.channel,
        }

        # No cursor → full sync
        if cursor is None or cursor.last_message_id == "":
            return ConversationDelta(
                session_id=session_id,
                new_messages=[m.to_dict() for m in conversation.messages],
                is_full_sync=True,
                conversation_metadata=metadata,
            )

        # Find where the cursor points in the current conversation
        cursor_index = None
        for i, msg in enumerate(conversation.messages):
            if msg.id == cursor.last_message_id:
                cursor_index = i
                break

        # Cursor message was removed (compaction happened) → full sync
        if cursor_index is None:
            log.info(
                "Cursor message %s not found for %s/%s — falling back to full sync",
                cursor.last_message_id,
                worker_id,
                session_id,
            )
            return ConversationDelta(
                session_id=session_id,
                new_messages=[m.to_dict() for m in conversation.messages],
                is_full_sync=True,
                conversation_metadata=metadata,
            )

        # Conversation was truncated/reorganized
        if cursor.synced_message_count != cursor_index + 1:
            return ConversationDelta(
                session_id=session_id,
                new_messages=[m.to_dict() for m in conversation.messages],
                is_full_sync=True,
                conversation_metadata=metadata,
            )

        # Happy path: append-only delta
        new_messages = conversation.messages[cursor_index + 1 :]
        if not new_messages:
            return ConversationDelta(
                session_id=session_id,
                base_message_id=cursor.last_message_id,
                conversation_metadata=metadata,
            )

        return ConversationDelta(
            session_id=session_id,
            new_messages=[m.to_dict() for m in new_messages],
            is_full_sync=False,
            base_message_id=cursor.last_message_id,
            conversation_metadata=metadata,
        )

    def advance_cursor(
        self,
        worker_id: str,
        session_id: str,
        conversation: Conversation,
    ) -> SyncCursor:
        """Advance the cursor to the end of the conversation after a successful sync."""
        if not conversation.messages:
            cursor = SyncCursor(worker_id=worker_id, session_id=session_id)
            self._set_cursor(cursor)
            return cursor

        last_msg = conversation.messages[-1]
        cursor = SyncCursor(
            worker_id=worker_id,
            session_id=session_id,
            last_message_id=last_msg.id,
            last_message_index=len(conversation.messages) - 1,
            conversation_hash=_conversation_hash(conversation),
            synced_message_count=len(conversation.messages),
        )
        self._set_cursor(cursor)
        return cursor


def apply_delta(
    conversation: Conversation,
    delta: ConversationDelta,
) -> Conversation:
    """Apply a delta to a local conversation copy (worker side).

    For full syncs, replaces all messages.
    For incremental syncs, appends new messages after the base.
    """
    if delta.is_full_sync:
        conversation.messages = [Message.from_dict(m) for m in delta.new_messages]
        if delta.conversation_metadata:
            meta = delta.conversation_metadata
            if "title" in meta:
                conversation.title = meta["title"]
            if "channel" in meta:
                conversation.channel = meta["channel"]
        return conversation

    # Remove messages by ID
    if delta.removed_message_ids:
        removed_set = set(delta.removed_message_ids)
        conversation.messages = [m for m in conversation.messages if m.id not in removed_set]

    # Update existing messages
    if delta.updated_messages:
        updates = {m["id"]: m for m in delta.updated_messages}
        for i, msg in enumerate(conversation.messages):
            if msg.id in updates:
                conversation.messages[i] = Message.from_dict(updates[msg.id])

    # Append new messages
    if delta.new_messages:
        for msg_data in delta.new_messages:
            conversation.messages.append(Message.from_dict(msg_data))

    return conversation


def compute_response_delta(
    before_count: int,
    conversation: Conversation,
) -> ConversationDelta:
    """Compute the delta a worker sends back after executing a job.

    The worker records the message count before execution, then calls
    this to capture any new messages that were added during the job.
    """
    new_messages = conversation.messages[before_count:]
    return ConversationDelta(
        session_id=conversation.id,
        new_messages=[m.to_dict() for m in new_messages],
        is_full_sync=False,
    )

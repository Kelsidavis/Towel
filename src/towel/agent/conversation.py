"""Conversation state management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""

    role: Role
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    pinned: bool = False

    def to_chat_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
        if self.pinned:
            d["pinned"] = True
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        # Tolerant of partial dicts. Messages assembled on the wire rather than
        # persisted — e.g. the coordinator's injected memory system message —
        # may omit id/timestamp. Falling back to the dataclass defaults (a
        # generated id, now()) instead of indexing keeps a missing field from
        # raising KeyError inside the worker's job deserialization, which would
        # crash _run_job and hang the request until the inference timeout.
        kwargs: dict[str, Any] = {
            "role": Role(data["role"]),
            "content": data.get("content", ""),
            "metadata": data.get("metadata", {}),
            "pinned": data.get("pinned", False),
        }
        if data.get("id"):
            kwargs["id"] = data["id"]
        if data.get("timestamp"):
            kwargs["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**kwargs)


@dataclass
class Conversation:
    """An ordered sequence of messages forming a conversation."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    title: str = ""
    tags: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    channel: str = "cli"

    def add(self, role: Role, content: str, **metadata: Any) -> Message:
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        return msg

    def to_chat_messages(self) -> list[dict[str, str]]:
        return [m.to_chat_dict() for m in self.messages]

    @property
    def last(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    def latest_user_query(self) -> str:
        """Return the most recent user message text, or "" if there is none.

        Used by the system-prompt builders to fetch query-relevant
        memories — the user's last turn is the strongest signal for
        which memories to surface this round.
        """
        for msg in reversed(self.messages):
            if msg.role == Role.USER:
                return msg.content
        return ""

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def display_title(self) -> str:
        """Title if set, otherwise auto-generated from first user message."""
        if self.title:
            return self.title
        return self.summary

    @property
    def summary(self) -> str:
        """First user message, truncated — useful as fallback title."""
        for msg in self.messages:
            if msg.role == Role.USER:
                text = msg.content.strip().replace("\n", " ")
                # Strip @file references from summary
                text = __import__("re").sub(r"@[\w./~*?:-]+", "", text).strip()
                return text[:80] + "..." if len(text) > 80 else text
        return "(empty)"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "channel": self.channel,
            "created_at": self.created_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.title:
            d["title"] = self.title
        if self.tags:
            d["tags"] = self.tags
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        return cls(
            id=data["id"],
            title=data.get("title", ""),
            tags=data.get("tags", []),
            channel=data.get("channel", "cli"),
            created_at=datetime.fromisoformat(data["created_at"]),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
        )

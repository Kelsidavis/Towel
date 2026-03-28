"""Conversation state management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
        return cls(
            id=data["id"],
            role=Role(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            pinned=data.get("pinned", False),
        )


@dataclass
class Conversation:
    """An ordered sequence of messages forming a conversation."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    title: str = ""
    tags: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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

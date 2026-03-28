"""Agent events — typed events emitted during streaming generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    TOKEN = "token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESPONSE_COMPLETE = "response_complete"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class AgentEvent:
    """A single event emitted during agent execution."""

    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_ws_message(self, session_id: str) -> dict[str, Any]:
        """Serialize for WebSocket transmission."""
        return {
            "type": self.type.value,
            "session": session_id,
            **self.data,
        }

    @staticmethod
    def token(text: str) -> AgentEvent:
        return AgentEvent(type=EventType.TOKEN, data={"content": text})

    @staticmethod
    def tool_call(name: str, arguments: dict[str, Any]) -> AgentEvent:
        return AgentEvent(
            type=EventType.TOOL_CALL,
            data={"tool": name, "arguments": arguments},
        )

    @staticmethod
    def tool_result(name: str, result: str) -> AgentEvent:
        return AgentEvent(
            type=EventType.TOOL_RESULT,
            data={"tool": name, "result": result},
        )

    @staticmethod
    def complete(content: str, metadata: dict[str, Any] | None = None) -> AgentEvent:
        return AgentEvent(
            type=EventType.RESPONSE_COMPLETE,
            data={"content": content, "metadata": metadata or {}},
        )

    @staticmethod
    def cancelled(content: str, metadata: dict[str, Any] | None = None) -> AgentEvent:
        return AgentEvent(
            type=EventType.CANCELLED,
            data={"content": content, "metadata": metadata or {}},
        )

    @staticmethod
    def error(message: str) -> AgentEvent:
        return AgentEvent(type=EventType.ERROR, data={"message": message})

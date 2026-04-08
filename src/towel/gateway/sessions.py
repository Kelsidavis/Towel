"""Session management for the gateway."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from towel.agent.conversation import Conversation
from towel.persistence.store import ConversationStore

log = logging.getLogger("towel.gateway.sessions")


@dataclass
class Session:
    """A gateway session — ties a conversation to a connection context."""

    id: str
    conversation: Conversation = field(default_factory=Conversation)
    node_ids: list[str] = field(default_factory=list)


class SessionManager:
    """Manages active sessions with optional persistence."""

    def __init__(self, store: ConversationStore | None = None) -> None:
        self._sessions: dict[str, Session] = {}
        self.store = store

    def get_or_create(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            # Try loading from disk first
            conv = None
            if self.store:
                conv = self.store.load(session_id)
                if conv:
                    log.debug(f"Resumed session {session_id} from disk ({len(conv)} messages)")

            if conv is None:
                conv = Conversation(id=session_id)

            self._sessions[session_id] = Session(id=session_id, conversation=conv)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def save(self, session_id: str) -> None:
        """Persist a session's conversation to disk."""
        session = self._sessions.get(session_id)
        if session and self.store and len(session.conversation) > 0:
            self.store.save(session.conversation)

    def save_all(self) -> int:
        """Persist all active sessions. Returns count saved."""
        if not self.store:
            return 0
        saved = 0
        for session_id in self._sessions:
            self.save(session_id)
            saved += 1
        return saved

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        """Remove all sessions from memory."""
        self._sessions.clear()

    def all(self) -> list[Session]:
        return list(self._sessions.values())

    def __len__(self) -> int:
        return len(self._sessions)

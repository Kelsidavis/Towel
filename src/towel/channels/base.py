"""Base channel interface — all messaging adapters implement this."""

from __future__ import annotations

import abc
import json
from collections.abc import AsyncIterator
from typing import Any

import websockets


class Channel(abc.ABC):
    """Abstract base for messaging channel adapters.

    Each channel connects to the gateway via WebSocket and bridges
    messages between its platform and the Towel agent.
    """

    def __init__(self, gateway_url: str = "ws://127.0.0.1:18742") -> None:
        self.gateway_url = gateway_url
        self._ws: Any = None
        self._running = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Channel identifier (e.g., 'cli', 'telegram', 'discord')."""
        ...

    @abc.abstractmethod
    async def listen(self) -> None:
        """Start listening for incoming messages from the platform."""
        ...

    @abc.abstractmethod
    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a message out to the platform."""
        ...

    async def connect(self) -> None:
        """Connect to the gateway WebSocket."""
        self._ws = await websockets.connect(self.gateway_url)
        await self._ws.send(
            json.dumps(
                {
                    "type": "register",
                    "id": f"channel:{self.name}",
                }
            )
        )
        resp = json.loads(await self._ws.recv())
        assert resp.get("type") == "registered"

    async def send_to_gateway(self, content: str, session: str = "default") -> dict[str, Any]:
        """Send a message to the gateway and wait for the complete response."""
        if not self._ws:
            await self.connect()

        await self._ws.send(
            json.dumps(
                {
                    "type": "message",
                    "channel": self.name,
                    "session": session,
                    "content": content,
                    "stream": False,
                }
            )
        )
        raw = await self._ws.recv()
        return json.loads(raw)

    async def stream_from_gateway(
        self, content: str, session: str = "default"
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message and yield streaming events from the gateway."""
        if not self._ws:
            await self.connect()

        await self._ws.send(
            json.dumps(
                {
                    "type": "message",
                    "channel": self.name,
                    "session": session,
                    "content": content,
                    "stream": True,
                }
            )
        )

        while True:
            raw = await self._ws.recv()
            event = json.loads(raw)
            yield event
            if event.get("type") in ("response_complete", "error"):
                break

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

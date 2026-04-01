"""Matrix bot channel — run Towel on the Matrix protocol.

Uses Matrix client-server API directly (no matrix-nio dependency).
Connects to any Matrix homeserver (Element, Synapse, etc.).

Setup:
  1. Create a Matrix account for your bot
  2. Get an access token
  3. Run: towel matrix --homeserver https://matrix.org --token YOUR_TOKEN
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from towel.channels.base import Channel

log = logging.getLogger("towel.channels.matrix")


class MatrixChannel(Channel):
    """Matrix bot using client-server API (no SDK dependency)."""

    def __init__(
        self,
        homeserver: str,
        access_token: str,
        gateway_url: str = "ws://127.0.0.1:18742",
    ) -> None:
        super().__init__(gateway_url)
        self.homeserver = homeserver.rstrip("/")
        self.access_token = access_token
        self._user_id: str | None = None
        self._next_batch: str = ""

    @property
    def name(self) -> str:
        return "matrix"

    async def listen(self) -> None:
        """Connect to Matrix and start syncing."""
        import httpx

        headers = {"Authorization": f"Bearer {self.access_token}"}

        # Verify token and get user ID
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.homeserver}/_matrix/client/v3/account/whoami", headers=headers)
            data = resp.json()
            if "user_id" not in data:
                log.error(f"Auth failed: {data}")
                return
            self._user_id = data["user_id"]
            log.info(f"Matrix bot: {self._user_id}")

        # Initial sync to get batch token
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{self.homeserver}/_matrix/client/v3/sync",
                headers=headers, params={"timeout": "0"},
            )
            self._next_batch = resp.json().get("next_batch", "")

        # Long-poll sync loop
        log.info("Listening for Matrix messages...")
        async with httpx.AsyncClient(timeout=60) as client:
            while True:
                try:
                    resp = await client.get(
                        f"{self.homeserver}/_matrix/client/v3/sync",
                        headers=headers,
                        params={"since": self._next_batch, "timeout": "30000"},
                    )
                    data = resp.json()
                    self._next_batch = data.get("next_batch", self._next_batch)

                    # Process room events
                    for room_id, room_data in data.get("rooms", {}).get("join", {}).items():
                        for event in room_data.get("timeline", {}).get("events", []):
                            await self._handle_event(room_id, event, headers)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.error(f"Sync error: {e}")
                    await asyncio.sleep(5)

    async def _handle_event(self, room_id: str, event: dict, headers: dict) -> None:
        if event.get("type") != "m.room.message":
            return
        if event.get("sender") == self._user_id:
            return

        content = event.get("content", {})
        body = content.get("body", "").strip()
        if not body:
            return

        log.info(f"Message from {event['sender']}: {body[:50]}")

        try:
            session_id = f"matrix-{room_id}"
            response = await self.send_to_gateway(body, session=session_id)
            reply = response.get("content", "I couldn't generate a response.")
        except Exception as e:
            log.error(f"Gateway error: {e}")
            reply = f"Sorry: {e}"

        await self._send_message(room_id, reply, headers)

    async def _send_message(self, room_id: str, text: str, headers: dict) -> None:
        import httpx
        txn_id = str(int(time.time() * 1000))
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.put(
                    f"{self.homeserver}/_matrix/client/v3/rooms/{room_id}/send/m.room.message/{txn_id}",
                    headers=headers,
                    json={"msgtype": "m.text", "body": text[:60000]},
                )
        except Exception as e:
            log.error(f"Send failed: {e}")

    async def send(self, content: str, **kwargs: Any) -> None:
        room_id = kwargs.get("room_id")
        if room_id:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            await self._send_message(room_id, content, headers)

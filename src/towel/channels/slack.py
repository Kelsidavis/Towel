"""Slack bot channel — run Towel as a Slack app using Socket Mode.

Uses the Slack Web API directly (no slack_bolt dependency).
Socket Mode means no public URL needed — connects outbound via WebSocket.

Setup:
  1. Create a Slack app at api.slack.com/apps
  2. Enable Socket Mode, get an app-level token (xapp-...)
  3. Add bot token scopes: chat:write, app_mentions:read, im:history
  4. Get the bot token (xoxb-...)
  5. Run: towel slack --bot-token xoxb-... --app-token xapp-...
"""

from __future__ import annotations

import json
import logging
from typing import Any

from towel.channels.base import Channel

log = logging.getLogger("towel.channels.slack")


class SlackChannel(Channel):
    """Slack bot using Socket Mode (WebSocket, no public URL needed)."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        gateway_url: str = "ws://127.0.0.1:18742",
    ) -> None:
        super().__init__(gateway_url)
        self.bot_token = bot_token
        self.app_token = app_token
        self._bot_id: str | None = None

    @property
    def name(self) -> str:
        return "slack"

    async def listen(self) -> None:
        """Connect via Socket Mode and listen for events."""
        import httpx
        import websockets

        # Get WebSocket URL via apps.connections.open
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://slack.com/api/apps.connections.open",
                headers={"Authorization": f"Bearer {self.app_token}"},
            )
            data = resp.json()
            if not data.get("ok"):
                log.error(f"Socket Mode auth failed: {data.get('error')}")
                return
            ws_url = data["url"]

        # Get bot user ID
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {self.bot_token}"},
            )
            auth = resp.json()
            self._bot_id = auth.get("user_id")
            log.info(f"Slack bot: {auth.get('user', '?')} ({self._bot_id})")

        # Connect to Socket Mode WebSocket
        log.info("Connecting to Slack Socket Mode...")
        async with websockets.connect(ws_url) as ws:
            async for raw in ws:
                msg = json.loads(raw)

                # Acknowledge envelope
                if "envelope_id" in msg:
                    await ws.send(json.dumps({"envelope_id": msg["envelope_id"]}))

                if msg.get("type") == "events_api":
                    event = msg.get("payload", {}).get("event", {})
                    await self._handle_event(event)
                elif msg.get("type") == "disconnect":
                    log.info("Slack requested disconnect")
                    break

    async def _handle_event(self, event: dict) -> None:
        """Handle a Slack event."""
        event_type = event.get("type")

        if event_type == "app_mention" or event_type == "message":
            # Skip bot's own messages
            if event.get("bot_id") or event.get("user") == self._bot_id:
                return

            text = event.get("text", "").strip()
            channel_id = event.get("channel", "")

            # Strip bot mention
            if self._bot_id:
                text = text.replace(f"<@{self._bot_id}>", "").strip()

            if not text:
                return

            log.info(f"Slack message from {event.get('user', '?')}: {text[:50]}")

            # Get response from Towel gateway
            try:
                session_id = f"slack-{channel_id}"
                response = await self.send_to_gateway(text, session=session_id)
                reply = response.get("content", "I couldn't generate a response.")
            except Exception as e:
                log.error(f"Gateway error: {e}")
                reply = f"Sorry, something went wrong: {e}"

            await self._post_message(channel_id, reply)

    async def _post_message(self, channel: str, text: str) -> None:
        """Post a message to a Slack channel."""
        import httpx

        # Slack message limit is ~40000 chars but keep it reasonable
        chunks = [text[i : i + 3900] for i in range(0, len(text), 3900)]

        async with httpx.AsyncClient() as client:
            for chunk in chunks:
                try:
                    resp = await client.post(
                        "https://slack.com/api/chat.postMessage",
                        headers={"Authorization": f"Bearer {self.bot_token}"},
                        json={"channel": channel, "text": chunk, "mrkdwn": True},
                    )
                    data = resp.json()
                    if not data.get("ok"):
                        log.error(f"Slack send error: {data.get('error')}")
                except Exception as e:
                    log.error(f"Failed to post: {e}")

    async def send(self, content: str, **kwargs: Any) -> None:
        channel = kwargs.get("channel")
        if channel:
            await self._post_message(channel, content)

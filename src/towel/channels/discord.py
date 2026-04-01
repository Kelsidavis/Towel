"""Discord bot channel — run Towel as a Discord bot.

Connects to the Towel gateway via WebSocket and bridges messages
between Discord and the agent.

Setup:
  1. Create a Discord bot at https://discord.com/developers
  2. Get the bot token
  3. Run: towel discord --token YOUR_BOT_TOKEN

The bot responds to:
  - Direct messages
  - Messages mentioning the bot
  - Messages starting with a configurable prefix (default: !towel)
"""

from __future__ import annotations

import asyncio
import json
import logging
import platform
from typing import Any

from towel.channels.base import Channel

log = logging.getLogger("towel.channels.discord")

DISCORD_API = "https://discord.com/api/v10"
DISCORD_GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"


class DiscordChannel(Channel):
    """Discord bot channel using raw WebSocket (no discord.py dependency)."""

    def __init__(
        self,
        token: str,
        prefix: str = "!towel",
        gateway_url: str = "ws://127.0.0.1:18742",
    ) -> None:
        super().__init__(gateway_url)
        self.token = token
        self.prefix = prefix
        self._bot_id: str | None = None
        self._heartbeat_interval: float = 41.25
        self._sequence: int | None = None
        self._discord_ws: Any = None

    @property
    def name(self) -> str:
        return "discord"

    async def listen(self) -> None:
        """Connect to Discord gateway and start listening."""
        import websockets

        log.info("Connecting to Discord gateway...")
        async with websockets.connect(DISCORD_GATEWAY) as ws:
            self._discord_ws = ws

            # Receive Hello
            hello = json.loads(await ws.recv())
            self._heartbeat_interval = hello["d"]["heartbeat_interval"] / 1000

            # Start heartbeat
            asyncio.create_task(self._heartbeat(ws))

            # Identify
            await ws.send(
                json.dumps(
                    {
                        "op": 2,
                        "d": {
                            "token": self.token,
                            # GUILDS + GUILD_MESSAGES + DIRECT_MESSAGES + MESSAGE_CONTENT
                            "intents": 33281,
                            "properties": {
                                "os": platform.system().lower(),
                                "browser": "towel",
                                "device": "towel",
                            },
                        },
                    }
                )
            )

            log.info("Connected to Discord. Listening for messages...")

            async for raw in ws:
                msg = json.loads(raw)
                self._sequence = msg.get("s", self._sequence)

                if msg["op"] == 0:  # Dispatch
                    await self._handle_dispatch(msg["t"], msg["d"])
                elif msg["op"] == 11:  # Heartbeat ACK
                    pass
                elif msg["op"] == 7:  # Reconnect
                    log.info("Discord requested reconnect")
                    break

    async def _heartbeat(self, ws: Any) -> None:
        """Send periodic heartbeats to keep connection alive."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await ws.send(json.dumps({"op": 1, "d": self._sequence}))
            except Exception:
                break

    async def _handle_dispatch(self, event: str, data: dict) -> None:
        """Handle Discord gateway dispatch events."""
        if event == "READY":
            self._bot_id = data["user"]["id"]
            log.info(f"Bot ready as {data['user']['username']}#{data['user']['discriminator']}")
            return

        if event != "MESSAGE_CREATE":
            return

        # Ignore own messages
        if data["author"]["id"] == self._bot_id:
            return

        content = data.get("content", "").strip()
        channel_id = data["channel_id"]
        is_dm = data.get("guild_id") is None
        mentioned = self._bot_id and f"<@{self._bot_id}>" in content

        # Check if we should respond
        if not is_dm and not mentioned and not content.startswith(self.prefix):
            return

        # Strip prefix/mention
        if content.startswith(self.prefix):
            content = content[len(self.prefix) :].strip()
        if mentioned:
            content = content.replace(f"<@{self._bot_id}>", "").strip()

        if not content:
            return

        log.info(f"Message from {data['author']['username']}: {content[:50]}")

        # Get response from Towel gateway
        try:
            session_id = f"discord-{channel_id}"
            response = await self.send_to_gateway(content, session=session_id)
            reply = response.get("content", "I couldn't generate a response.")
        except Exception as e:
            log.error(f"Gateway error: {e}")
            reply = f"Sorry, I encountered an error: {e}"

        # Send reply to Discord
        await self._send_message(channel_id, reply)

    async def _send_message(self, channel_id: str, content: str) -> None:
        """Send a message to a Discord channel via REST API."""
        import httpx

        # Discord message limit is 2000 chars
        chunks = [content[i : i + 1990] for i in range(0, len(content), 1990)]

        async with httpx.AsyncClient() as client:
            for chunk in chunks:
                try:
                    await client.post(
                        f"{DISCORD_API}/channels/{channel_id}/messages",
                        headers={
                            "Authorization": f"Bot {self.token}",
                            "Content-Type": "application/json",
                        },
                        json={"content": chunk},
                    )
                except Exception as e:
                    log.error(f"Failed to send Discord message: {e}")

    async def send(self, content: str, **kwargs: Any) -> None:
        """Send a message to a specific channel."""
        channel_id = kwargs.get("channel_id")
        if channel_id:
            await self._send_message(channel_id, content)

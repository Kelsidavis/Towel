"""Telegram bot channel — run Towel as a Telegram bot.

Uses the Telegram Bot API directly (no python-telegram-bot dependency).
Long-polling for updates, sends responses via REST API.

Setup:
  1. Create a bot via @BotFather on Telegram
  2. Get the bot token
  3. Run: towel telegram --token YOUR_BOT_TOKEN
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from towel.channels.base import Channel

log = logging.getLogger("towel.channels.telegram")

API_BASE = "https://api.telegram.org/bot{token}"


class TelegramChannel(Channel):
    """Telegram bot using long-polling (no webhooks needed)."""

    def __init__(
        self,
        token: str,
        gateway_url: str = "ws://127.0.0.1:18742",
    ) -> None:
        super().__init__(gateway_url)
        self.token = token
        self._api = API_BASE.format(token=token)
        self._offset: int = 0

    @property
    def name(self) -> str:
        return "telegram"

    async def listen(self) -> None:
        """Start long-polling for Telegram updates."""
        import httpx

        # Verify bot token
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self._api}/getMe")
            data = resp.json()
            if not data.get("ok"):
                log.error(f"Invalid token: {data}")
                return
            bot = data["result"]
            log.info(f"Telegram bot: @{bot['username']} ({bot['first_name']})")

        # Poll for updates
        log.info("Listening for Telegram messages...")
        async with httpx.AsyncClient(timeout=60) as client:
            while True:
                try:
                    resp = await client.get(
                        f"{self._api}/getUpdates",
                        params={"offset": self._offset, "timeout": 30},
                    )
                    data = resp.json()

                    if not data.get("ok"):
                        log.warning(f"Telegram API error: {data}")
                        await asyncio.sleep(5)
                        continue

                    for update in data.get("result", []):
                        self._offset = update["update_id"] + 1
                        await self._handle_update(update)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.error(f"Polling error: {e}")
                    await asyncio.sleep(5)

    async def _handle_update(self, update: dict) -> None:
        """Process a single Telegram update."""
        message = update.get("message")
        if not message:
            return

        text = message.get("text", "").strip()
        chat_id = message["chat"]["id"]
        username = message.get("from", {}).get("username", "unknown")

        if not text:
            return

        # Skip /start command
        if text == "/start":
            await self._send(chat_id, "Hey! I'm Towel. Don't Panic. Just send me a message.")
            return

        log.info(f"Message from @{username}: {text[:50]}")

        # Get response from Towel gateway
        try:
            session_id = f"telegram-{chat_id}"
            response = await self.send_to_gateway(text, session=session_id)
            reply = response.get("content", "I couldn't generate a response.")
        except Exception as e:
            log.error(f"Gateway error: {e}")
            reply = f"Sorry, something went wrong: {e}"

        await self._send(chat_id, reply)

    async def _send(self, chat_id: int, text: str) -> None:
        """Send a message to a Telegram chat."""
        import httpx

        # Telegram message limit is 4096 chars
        chunks = [text[i : i + 4090] for i in range(0, len(text), 4090)]

        async with httpx.AsyncClient() as client:
            for chunk in chunks:
                try:
                    await client.post(
                        f"{self._api}/sendMessage",
                        json={
                            "chat_id": chat_id,
                            "text": chunk,
                            "parse_mode": "Markdown",
                        },
                    )
                except Exception:
                    # Retry without markdown if it fails
                    try:
                        await client.post(
                            f"{self._api}/sendMessage",
                            json={"chat_id": chat_id, "text": chunk},
                        )
                    except Exception as e2:
                        log.error(f"Failed to send: {e2}")

    async def send(self, content: str, **kwargs: Any) -> None:
        chat_id = kwargs.get("chat_id")
        if chat_id:
            await self._send(chat_id, content)

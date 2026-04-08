"""Gmail skill — read, search, label, and draft emails."""

from __future__ import annotations

import base64
import email.mime.text
import logging
from typing import Any

from towel.skills.base import Skill, ToolDefinition

log = logging.getLogger("towel.skills.gmail")

MAX_BODY_CHARS = 4000


class GmailSkill(Skill):
    """Read, search, and manage Gmail messages."""

    @property
    def name(self) -> str:
        return "gmail"

    @property
    def description(self) -> str:
        return "Gmail — list, read, search, label, and draft emails"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="gmail_unread",
                description="List unread emails. Returns sender, subject, snippet, and ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "max_results": {
                            "type": "integer",
                            "description": "Max emails to return (default: 10)",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="gmail_read",
                description="Read the full body of an email by message ID.",
                parameters={
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "Gmail message ID",
                        },
                    },
                    "required": ["message_id"],
                },
            ),
            ToolDefinition(
                name="gmail_search",
                description="Search Gmail with a query (same syntax as Gmail search bar).",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Gmail search query (e.g., 'from:boss subject:urgent')",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max results (default: 10)",
                        },
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="gmail_send",
                description="Send an email or reply.",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Plain text body"},
                        "reply_to_id": {
                            "type": "string",
                            "description": "Message ID to reply to (optional)",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
            ),
            ToolDefinition(
                name="gmail_label",
                description="Add or remove a label from a message.",
                parameters={
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string", "description": "Gmail message ID"},
                        "add_labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels to add (e.g., ['STARRED'])",
                        },
                        "remove_labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Labels to remove (e.g., ['UNREAD'])",
                        },
                    },
                    "required": ["message_id"],
                },
            ),
            ToolDefinition(
                name="gmail_trash",
                description="Move a message to trash.",
                parameters={
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string", "description": "Gmail message ID"},
                    },
                    "required": ["message_id"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            match tool_name:
                case "gmail_unread":
                    return await self._unread(arguments.get("max_results", 10))
                case "gmail_read":
                    return await self._read(arguments["message_id"])
                case "gmail_search":
                    return await self._search(
                        arguments["query"], arguments.get("max_results", 10)
                    )
                case "gmail_send":
                    return await self._send(
                        arguments["to"],
                        arguments["subject"],
                        arguments["body"],
                        arguments.get("reply_to_id"),
                    )
                case "gmail_label":
                    return await self._label(
                        arguments["message_id"],
                        arguments.get("add_labels", []),
                        arguments.get("remove_labels", []),
                    )
                case "gmail_trash":
                    return await self._trash(arguments["message_id"])
                case _:
                    return f"Unknown tool: {tool_name}"
        except Exception as e:
            return f"Gmail error: {e}"

    def _get_service(self) -> Any:
        from towel.skills.builtin.google_auth import build_gmail_service
        return build_gmail_service()

    def _header(self, headers: list[dict], name: str) -> str:
        for h in headers:
            if h["name"].lower() == name.lower():
                return h["value"]
        return ""

    async def _unread(self, max_results: int) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)
        results = await asyncio.to_thread(
            lambda: svc.users().messages().list(
                userId="me", q="is:unread", maxResults=max_results
            ).execute()
        )
        messages = results.get("messages", [])
        if not messages:
            return "Inbox clear — no unread emails."

        lines = [f"**{len(messages)} unread emails:**\n"]
        for msg_stub in messages:
            msg = await asyncio.to_thread(
                lambda mid=msg_stub["id"]: svc.users().messages().get(
                    userId="me", id=mid, format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                ).execute()
            )
            headers = msg.get("payload", {}).get("headers", [])
            sender = self._header(headers, "From")
            subject = self._header(headers, "Subject")
            date = self._header(headers, "Date")
            snippet = msg.get("snippet", "")
            lines.append(f"- **{subject}**\n  From: {sender} | {date}\n  {snippet}\n  ID: `{msg['id']}`\n")

        return "\n".join(lines)

    async def _read(self, message_id: str) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)
        msg = await asyncio.to_thread(
            lambda: svc.users().messages().get(userId="me", id=message_id, format="full").execute()
        )
        headers = msg.get("payload", {}).get("headers", [])
        subject = self._header(headers, "Subject")
        sender = self._header(headers, "From")
        date = self._header(headers, "Date")

        body = self._extract_body(msg.get("payload", {}))
        if len(body) > MAX_BODY_CHARS:
            body = body[:MAX_BODY_CHARS] + "\n\n[... truncated ...]"

        return f"**{subject}**\nFrom: {sender}\nDate: {date}\n\n{body}"

    def _extract_body(self, payload: dict) -> str:
        """Extract plain text body from Gmail message payload."""
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

        for part in payload.get("parts", []):
            text = self._extract_body(part)
            if text:
                return text
        return "(No plain text body found)"

    async def _search(self, query: str, max_results: int) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)
        results = await asyncio.to_thread(
            lambda: svc.users().messages().list(
                userId="me", q=query, maxResults=max_results
            ).execute()
        )
        messages = results.get("messages", [])
        if not messages:
            return f"No results for: {query}"

        lines = [f"**{len(messages)} results for '{query}':**\n"]
        for msg_stub in messages:
            msg = await asyncio.to_thread(
                lambda mid=msg_stub["id"]: svc.users().messages().get(
                    userId="me", id=mid, format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                ).execute()
            )
            headers = msg.get("payload", {}).get("headers", [])
            sender = self._header(headers, "From")
            subject = self._header(headers, "Subject")
            snippet = msg.get("snippet", "")
            lines.append(f"- **{subject}** from {sender}\n  {snippet}\n  ID: `{msg['id']}`\n")

        return "\n".join(lines)

    async def _send(self, to: str, subject: str, body: str, reply_to_id: str | None) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)

        message = email.mime.text.MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        if reply_to_id:
            orig = await asyncio.to_thread(
                lambda: svc.users().messages().get(
                    userId="me", id=reply_to_id, format="metadata",
                    metadataHeaders=["Message-ID", "Subject"],
                ).execute()
            )
            orig_headers = orig.get("payload", {}).get("headers", [])
            msg_id_header = self._header(orig_headers, "Message-ID")
            if msg_id_header:
                message["In-Reply-To"] = msg_id_header
                message["References"] = msg_id_header
            thread_id = orig.get("threadId")
        else:
            thread_id = None

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_body: dict[str, Any] = {"raw": raw}
        if thread_id:
            send_body["threadId"] = thread_id

        result = await asyncio.to_thread(
            lambda: svc.users().messages().send(userId="me", body=send_body).execute()
        )
        return f"Email sent. Message ID: {result['id']}"

    async def _label(self, message_id: str, add: list[str], remove: list[str]) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)
        body: dict[str, Any] = {}
        if add:
            body["addLabelIds"] = add
        if remove:
            body["removeLabelIds"] = remove
        await asyncio.to_thread(
            lambda: svc.users().messages().modify(
                userId="me", id=message_id, body=body
            ).execute()
        )
        return f"Labels updated on {message_id}"

    async def _trash(self, message_id: str) -> str:
        import asyncio
        svc = await asyncio.to_thread(self._get_service)
        await asyncio.to_thread(
            lambda: svc.users().messages().trash(userId="me", id=message_id).execute()
        )
        return f"Message {message_id} moved to trash."

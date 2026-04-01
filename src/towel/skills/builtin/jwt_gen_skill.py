"""JWT generator skill — create JWTs for testing."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class JwtGenSkill(Skill):
    @property
    def name(self) -> str:
        return "jwt_gen"

    @property
    def description(self) -> str:
        return "Generate JWT tokens for testing"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="jwt_create",
                description="Create a JWT for testing (HS256)",
                parameters={
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "description": "Claims (e.g., {sub: '123', name: 'Test'})",
                        },
                        "secret": {
                            "type": "string",
                            "description": "Signing secret (default: 'towel-test-secret')",
                        },
                        "expiry_hours": {
                            "type": "number",
                            "description": "Hours until expiry (default: 1)",
                        },
                    },
                    "required": ["payload"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "jwt_create":
            return f"Unknown: {tool_name}"
        payload = dict(arguments["payload"])
        secret = arguments.get("secret", "towel-test-secret")
        hours = arguments.get("expiry_hours", 1)
        now = int(time.time())
        payload.setdefault("iat", now)
        payload.setdefault("exp", now + int(hours * 3600))
        header = {"alg": "HS256", "typ": "JWT"}

        def b64(d):
            return (
                base64.urlsafe_b64encode(json.dumps(d, separators=(",", ":")).encode())
                .rstrip(b"=")
                .decode()
            )

        h, p = b64(header), b64(payload)
        sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
        token = f"{h}.{p}.{sig_b64}"
        payload_str = json.dumps(payload, indent=2)
        return (
            f"JWT (HS256, expires in {hours}h):\n{token}"
            f"\n\nPayload:\n{payload_str}"
        )

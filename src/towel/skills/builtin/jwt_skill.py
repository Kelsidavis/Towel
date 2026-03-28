"""JWT skill — decode, inspect, and validate JSON Web Tokens."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from towel.skills.base import Skill, ToolDefinition


def _b64_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4: s += "=" * padding
    return base64.urlsafe_b64decode(s)


class JwtSkill(Skill):
    @property
    def name(self) -> str: return "jwt"
    @property
    def description(self) -> str: return "Decode and inspect JSON Web Tokens (no signature verification)"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="jwt_decode", description="Decode a JWT and show header, payload, and expiry status",
                parameters={"type":"object","properties":{
                    "token":{"type":"string","description":"The JWT string"},
                },"required":["token"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "jwt_decode": return f"Unknown tool: {tool_name}"
        return self._decode(arguments["token"])

    def _decode(self, token: str) -> str:
        token = token.strip()
        parts = token.split(".")
        if len(parts) != 3:
            return f"Invalid JWT: expected 3 parts, got {len(parts)}"

        try:
            header = json.loads(_b64_decode(parts[0]))
            payload = json.loads(_b64_decode(parts[1]))
        except Exception as e:
            return f"Failed to decode JWT: {e}"

        lines = ["JWT Decoded:\n"]
        lines.append("Header:")
        lines.append(json.dumps(header, indent=2))
        lines.append("\nPayload:")
        lines.append(json.dumps(payload, indent=2))

        # Check expiry
        exp = payload.get("exp")
        iat = payload.get("iat")
        now = int(time.time())

        if exp:
            from datetime import datetime, timezone
            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
            if exp < now:
                lines.append(f"\nStatus: EXPIRED (expired {exp_dt.isoformat()})")
            else:
                remaining = exp - now
                h, m = divmod(remaining // 60, 60)
                lines.append(f"\nStatus: VALID (expires {exp_dt.isoformat()}, {h}h {m}m remaining)")

        if iat:
            from datetime import datetime, timezone
            iat_dt = datetime.fromtimestamp(iat, tz=timezone.utc)
            lines.append(f"Issued: {iat_dt.isoformat()}")

        sub = payload.get("sub")
        if sub: lines.append(f"Subject: {sub}")
        iss = payload.get("iss")
        if iss: lines.append(f"Issuer: {iss}")

        lines.append(f"\nAlgorithm: {header.get('alg', 'unknown')}")
        lines.append("(Signature not verified — decode only)")
        return "\n".join(lines)

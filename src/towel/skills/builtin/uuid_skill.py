"""UUID and random data generation skill."""

from __future__ import annotations

import secrets
import uuid
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class UuidSkill(Skill):
    @property
    def name(self) -> str: return "uuid"
    @property
    def description(self) -> str: return "Generate UUIDs, random strings, passwords, and test data"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="generate_uuid", description="Generate one or more UUIDs (v4)",
                parameters={"type":"object","properties":{
                    "count":{"type":"integer","description":"How many (default: 1)"},
                    "format":{"type":"string","enum":["standard","hex","short"],"description":"Format"},
                }}),
            ToolDefinition(name="generate_password", description="Generate a secure random password",
                parameters={"type":"object","properties":{
                    "length":{"type":"integer","description":"Length (default: 24)"},
                    "no_symbols":{"type":"boolean","description":"Alphanumeric only (default: false)"},
                }}),
            ToolDefinition(name="generate_token", description="Generate a random hex or base64 token",
                parameters={"type":"object","properties":{
                    "bytes":{"type":"integer","description":"Number of random bytes (default: 32)"},
                    "encoding":{"type":"string","enum":["hex","base64","urlsafe"],"description":"Encoding (default: hex)"},
                }}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "generate_uuid": return self._uuid(arguments.get("count",1), arguments.get("format","standard"))
            case "generate_password": return self._password(arguments.get("length",24), arguments.get("no_symbols",False))
            case "generate_token": return self._token(arguments.get("bytes",32), arguments.get("encoding","hex"))
            case _: return f"Unknown tool: {tool_name}"

    def _uuid(self, count: int, fmt: str) -> str:
        count = min(count, 50)
        uuids = []
        for _ in range(count):
            u = uuid.uuid4()
            if fmt == "hex": uuids.append(u.hex)
            elif fmt == "short": uuids.append(u.hex[:8])
            else: uuids.append(str(u))
        return "\n".join(uuids)

    def _password(self, length: int, no_symbols: bool) -> str:
        length = max(8, min(length, 128))
        if no_symbols:
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        else:
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*_+-="
        pw = "".join(secrets.choice(chars) for _ in range(length))
        return f"Password ({length} chars): {pw}"

    def _token(self, nbytes: int, encoding: str) -> str:
        nbytes = max(8, min(nbytes, 256))
        raw = secrets.token_bytes(nbytes)
        import base64
        match encoding:
            case "hex": val = raw.hex()
            case "base64": val = base64.b64encode(raw).decode()
            case "urlsafe": val = base64.urlsafe_b64encode(raw).decode().rstrip("=")
            case _: val = raw.hex()
        return f"Token ({nbytes} bytes, {encoding}): {val}"

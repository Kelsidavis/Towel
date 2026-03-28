"""Hash and encoding skill — checksums, base64, URL encoding."""

from __future__ import annotations

import base64
import hashlib
import urllib.parse
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class HashSkill(Skill):
    @property
    def name(self) -> str:
        return "hash"

    @property
    def description(self) -> str:
        return "Compute hashes, encode/decode base64 and URLs"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="hash_text",
                description="Compute hash of text (md5, sha1, sha256, sha512)",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to hash"},
                        "algorithm": {
                            "type": "string",
                            "enum": ["md5", "sha1", "sha256", "sha512"],
                            "description": "Hash algorithm (default: sha256)",
                        },
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="hash_file",
                description="Compute hash of a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"},
                        "algorithm": {
                            "type": "string",
                            "enum": ["md5", "sha1", "sha256", "sha512"],
                            "description": "Hash algorithm (default: sha256)",
                        },
                    },
                    "required": ["path"],
                },
            ),
            ToolDefinition(
                name="base64_encode",
                description="Encode text to base64",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to encode"},
                    },
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="base64_decode",
                description="Decode base64 to text",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Base64 string to decode"},
                    },
                    "required": ["data"],
                },
            ),
            ToolDefinition(
                name="url_encode",
                description="URL-encode or decode a string",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to encode/decode"},
                        "decode": {"type": "boolean", "description": "Decode instead of encode (default: false)"},
                    },
                    "required": ["text"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "hash_text":
                return self._hash_text(arguments["text"], arguments.get("algorithm", "sha256"))
            case "hash_file":
                return self._hash_file(arguments["path"], arguments.get("algorithm", "sha256"))
            case "base64_encode":
                return self._b64_encode(arguments["text"])
            case "base64_decode":
                return self._b64_decode(arguments["data"])
            case "url_encode":
                return self._url_encode(arguments["text"], arguments.get("decode", False))
            case _:
                return f"Unknown tool: {tool_name}"

    def _hash_text(self, text: str, algo: str) -> str:
        try:
            h = hashlib.new(algo)
            h.update(text.encode("utf-8"))
            return f"{algo}: {h.hexdigest()}"
        except ValueError:
            return f"Unknown algorithm: {algo}"

    def _hash_file(self, path: str, algo: str) -> str:
        from pathlib import Path
        target = Path(path).expanduser()
        if not target.is_file():
            return f"File not found: {path}"
        try:
            h = hashlib.new(algo)
            with open(target, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            size = target.stat().st_size
            return f"{algo}: {h.hexdigest()}\nFile: {target.name} ({size:,} bytes)"
        except (OSError, ValueError) as e:
            return f"Error: {e}"

    def _b64_encode(self, text: str) -> str:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return f"Base64: {encoded}"

    def _b64_decode(self, data: str) -> str:
        try:
            decoded = base64.b64decode(data).decode("utf-8", errors="replace")
            return f"Decoded: {decoded}"
        except Exception as e:
            return f"Decode error: {e}"

    def _url_encode(self, text: str, decode: bool) -> str:
        if decode:
            return f"Decoded: {urllib.parse.unquote(text)}"
        return f"Encoded: {urllib.parse.quote(text, safe='')}"

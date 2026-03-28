"""Keychain skill — securely store and retrieve secrets using macOS Keychain."""

from __future__ import annotations

import asyncio
import platform
from typing import Any

from towel.skills.base import Skill, ToolDefinition

SERVICE = "towel-secrets"


class KeychainSkill(Skill):
    @property
    def name(self) -> str: return "keychain"
    @property
    def description(self) -> str: return "Securely store and retrieve secrets via macOS Keychain"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="secret_set", description="Store a secret in the system keychain",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Secret name/key"},
                    "value":{"type":"string","description":"Secret value"},
                },"required":["name","value"]}),
            ToolDefinition(name="secret_get", description="Retrieve a secret from the keychain",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Secret name/key"},
                },"required":["name"]}),
            ToolDefinition(name="secret_delete", description="Delete a secret from the keychain",
                parameters={"type":"object","properties":{
                    "name":{"type":"string","description":"Secret name/key"},
                },"required":["name"]}),
            ToolDefinition(name="secret_list", description="List stored secret names (values hidden)",
                parameters={"type":"object","properties":{}}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if platform.system() != "Darwin":
            return "Keychain skill requires macOS."
        match tool_name:
            case "secret_set": return await self._set(arguments["name"], arguments["value"])
            case "secret_get": return await self._get(arguments["name"])
            case "secret_delete": return await self._delete(arguments["name"])
            case "secret_list": return await self._list()
            case _: return f"Unknown tool: {tool_name}"

    async def _run(self, args: list[str], stdin: str|None = None) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            "security", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin else None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin.encode() if stdin else None), timeout=10
        )
        out = (stdout or stderr).decode("utf-8","replace").strip()
        return proc.returncode or 0, out

    async def _set(self, name: str, value: str) -> str:
        # Delete first (update = delete + add)
        await self._run(["delete-generic-password", "-s", SERVICE, "-a", name])
        code, out = await self._run([
            "add-generic-password", "-s", SERVICE, "-a", name, "-w", value, "-U"
        ])
        return f"Stored: {name}" if code == 0 else f"Failed: {out}"

    async def _get(self, name: str) -> str:
        code, out = await self._run([
            "find-generic-password", "-s", SERVICE, "-a", name, "-w"
        ])
        if code != 0: return f"Not found: {name}"
        return out

    async def _delete(self, name: str) -> str:
        code, out = await self._run([
            "delete-generic-password", "-s", SERVICE, "-a", name
        ])
        return f"Deleted: {name}" if code == 0 else f"Not found: {name}"

    async def _list(self) -> str:
        code, out = await self._run(["dump-keychain"])
        if code != 0: return "Cannot read keychain."
        import re
        names = set()
        for m in re.finditer(r'"svce"<blob>="towel-secrets".*?"acct"<blob>="([^"]+)"', out, re.DOTALL):
            names.add(m.group(1))
        if not names:
            # Fallback: search line by line
            lines = out.splitlines()
            for i, line in enumerate(lines):
                if SERVICE in line:
                    for j in range(max(0,i-3), min(len(lines),i+3)):
                        m2 = re.search(r'"acct"<blob>="([^"]+)"', lines[j])
                        if m2: names.add(m2.group(1))
        if not names: return "No towel secrets stored."
        return f"Stored secrets ({len(names)}):\n" + "\n".join(f"  {n}" for n in sorted(names))

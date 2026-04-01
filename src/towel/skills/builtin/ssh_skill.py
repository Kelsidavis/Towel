"""SSH skill — manage SSH keys, known_hosts, and config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

SSH_DIR = Path.home() / ".ssh"


class SshSkill(Skill):
    @property
    def name(self) -> str:
        return "ssh"

    @property
    def description(self) -> str:
        return "Inspect SSH keys, known_hosts, and config"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="ssh_keys",
                description="List SSH keys in ~/.ssh with fingerprints",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="ssh_config",
                description="Show parsed SSH config (hosts, users, ports)",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="ssh_known_hosts",
                description="List or search known_hosts entries",
                parameters={
                    "type": "object",
                    "properties": {
                        "search": {
                            "type": "string",
                            "description": "Filter by hostname (optional)",
                        },
                    },
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "ssh_keys":
                return self._keys()
            case "ssh_config":
                return self._config()
            case "ssh_known_hosts":
                return self._known_hosts(arguments.get("search"))
            case _:
                return f"Unknown tool: {tool_name}"

    def _keys(self) -> str:
        if not SSH_DIR.exists():
            return "~/.ssh directory not found."
        keys = []
        for f in sorted(SSH_DIR.iterdir()):
            if f.suffix == ".pub":
                try:
                    content = f.read_text().strip()
                    parts = content.split()
                    algo = parts[0] if parts else "?"
                    comment = parts[2] if len(parts) > 2 else ""
                    priv = f.with_suffix("")
                    has_priv = "+" if priv.exists() else "-"
                    keys.append(f"  [{has_priv}] {f.name} ({algo}) {comment}")
                except Exception:
                    pass
        if not keys:
            return "No SSH keys found in ~/.ssh"
        return f"SSH keys ({len(keys)}):\n" + "\n".join(keys)

    def _config(self) -> str:
        cfg = SSH_DIR / "config"
        if not cfg.exists():
            return "No ~/.ssh/config file."
        try:
            content = cfg.read_text()
        except Exception:
            return "Cannot read ~/.ssh/config"
        hosts = []
        current: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("host ") and not line.lower().startswith("host *"):
                if current:
                    hosts.append(current)
                current = {"Host": line.split(None, 1)[1]}
            elif "=" in line or " " in line:
                key, _, val = line.partition(" ")
                if not val:
                    key, _, val = line.partition("=")
                current[key.strip()] = val.strip()
        if current:
            hosts.append(current)
        if not hosts:
            return "No host entries in SSH config."
        lines = [f"SSH config ({len(hosts)} hosts):"]
        for h in hosts:
            name = h.get("Host", "?")
            hostname = h.get("HostName", h.get("Hostname", ""))
            user = h.get("User", "")
            port = h.get("Port", "")
            extra = f" ({user}@{hostname}:{port})" if hostname else ""
            lines.append(f"  {name}{extra}")
        return "\n".join(lines)

    def _known_hosts(self, search: str | None) -> str:
        kh = SSH_DIR / "known_hosts"
        if not kh.exists():
            return "No known_hosts file."
        try:
            entries = kh.read_text().strip().splitlines()
        except Exception:
            return "Cannot read known_hosts."
        if search:
            entries = [e for e in entries if search.lower() in e.lower()]
        if not entries:
            return f"No entries{' matching ' + search if search else ''}."
        lines = [f"known_hosts ({len(entries)} entries):"]
        for e in entries[:30]:
            parts = e.split()
            host = parts[0][:50] if parts else "?"
            algo = parts[1] if len(parts) > 1 else "?"
            lines.append(f"  {host} ({algo})")
        if len(entries) > 30:
            lines.append(f"  ... and {len(entries) - 30} more")
        return "\n".join(lines)

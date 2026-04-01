"""Shell skill — execute shell commands with safety guardrails."""

from __future__ import annotations

import asyncio
from typing import Any

from towel.skills.base import Skill, ToolDefinition

# Commands that are never allowed
_BLOCKED_COMMANDS = {"rm -rf /", "mkfs", "dd if=/dev/zero", ":(){", "fork bomb"}
_BLOCKED_PREFIXES = ("sudo rm -rf", "rm -rf /", "mkfs.", "dd if=/dev")

MAX_OUTPUT_BYTES = 50_000
DEFAULT_TIMEOUT = 30


class ShellSkill(Skill):
    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Execute shell commands and return their output"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="run_command",
                description="Run a shell command and return stdout/stderr. Timeout 30s.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to run"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 120)"},
                    },
                    "required": ["command"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "run_command":
            return f"Unknown tool: {tool_name}"

        command = arguments["command"]
        timeout = min(arguments.get("timeout", DEFAULT_TIMEOUT), 120)

        # Safety check
        cmd_lower = command.lower().strip()
        for blocked in _BLOCKED_PREFIXES:
            if cmd_lower.startswith(blocked):
                return f"Blocked: '{command}' matches a dangerous pattern."

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return f"Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

        out = stdout.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]
        err = stderr.decode("utf-8", errors="replace")[:MAX_OUTPUT_BYTES]

        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(f"[stderr] {err}")
        parts.append(f"[exit: {proc.returncode}]")
        return "\n".join(parts)

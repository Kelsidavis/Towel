"""Man page skill — look up command documentation."""

from __future__ import annotations

import asyncio
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class ManSkill(Skill):
    @property
    def name(self) -> str: return "man"
    @property
    def description(self) -> str: return "Look up command man pages and --help output"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="man_page", description="Show the man page for a command (summary section)",
                parameters={"type":"object","properties":{
                    "command":{"type":"string","description":"Command name (e.g., grep, curl, git)"},
                    "section":{"type":"string","description":"Man section (1-8, default: auto)"},
                },"required":["command"]}),
            ToolDefinition(name="man_help", description="Run --help for a command and return output",
                parameters={"type":"object","properties":{
                    "command":{"type":"string","description":"Command to get help for"},
                },"required":["command"]}),
            ToolDefinition(name="man_tldr", description="Show a concise cheat-sheet style summary of a command",
                parameters={"type":"object","properties":{
                    "command":{"type":"string","description":"Command name"},
                },"required":["command"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "man_page": return await self._man(arguments["command"], arguments.get("section"))
            case "man_help": return await self._help(arguments["command"])
            case "man_tldr": return await self._tldr(arguments["command"])
            case _: return f"Unknown tool: {tool_name}"

    async def _run(self, cmd: list[str], timeout: int = 10) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={"PATH": "/usr/bin:/usr/local/bin:/opt/homebrew/bin", "COLUMNS": "100", "TERM": "dumb", "MAN_KEEP_FORMATTING": "1"},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            out = stdout.decode("utf-8", "replace").strip()
            if not out: out = stderr.decode("utf-8", "replace").strip()
            return out[:10000]
        except FileNotFoundError:
            return f"Command not found: {cmd[0]}"
        except TimeoutError:
            return "Command timed out."
        except Exception as e:
            return f"Error: {e}"

    async def _man(self, command: str, section: str | None) -> str:
        args = ["man"]
        if section: args.append(section)
        args.append(command)
        output = await self._run(args)
        if "No manual entry" in output:
            return f"No man page for: {command}"
        # Extract NAME and DESCRIPTION sections for brevity
        import re
        sections = {}
        current = None
        for line in output.splitlines():
            header = re.match(r'^([A-Z][A-Z ]+)$', line.strip())
            if header:
                current = header.group(1).strip()
                sections[current] = []
            elif current:
                sections[current].append(line)

        result = [f"man {command}:"]
        for key in ["NAME", "SYNOPSIS", "DESCRIPTION"]:
            if key in sections:
                text = "\n".join(sections[key][:20]).strip()
                result.append(f"\n{key}:\n{text}")
        if len(result) == 1:
            return output[:3000]
        return "\n".join(result)

    async def _help(self, command: str) -> str:
        output = await self._run([command, "--help"])
        if "not found" in output.lower():
            output = await self._run([command, "-h"])
        return output or f"No help output for: {command}"

    async def _tldr(self, command: str) -> str:
        # Try tldr command first
        output = await self._run(["tldr", command])
        if "not found" not in output.lower() and output.strip():
            return output
        # Fallback: extract EXAMPLES from man page
        man_output = await self._run(["man", command])
        import re
        in_examples = False
        examples = []
        for line in man_output.splitlines():
            if re.match(r'^EXAMPLES?$', line.strip()):
                in_examples = True
                continue
            elif re.match(r'^[A-Z][A-Z ]+$', line.strip()) and in_examples:
                break
            elif in_examples:
                examples.append(line)
        if examples:
            return f"Examples for {command}:\n" + "\n".join(examples[:30])
        return f"No tldr or examples found for: {command}. Try 'man_help' instead."

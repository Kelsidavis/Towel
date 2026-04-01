"""Changelog generator skill — generate changelogs from git history."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class ChangelogGenSkill(Skill):
    @property
    def name(self) -> str:
        return "changelog_gen"

    @property
    def description(self) -> str:
        return "Generate changelogs from git commit history"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="changelog_generate",
                description="Generate a changelog from recent git commits",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Git repo path (default: cwd)"},
                        "since": {
                            "type": "string",
                            "description": "Since tag or commit (e.g., v1.0.0, HEAD~20)",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["keepachangelog", "simple", "grouped"],
                            "description": "Format (default: grouped)",
                        },
                    },
                    "required": [],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "changelog_generate":
            return f"Unknown: {tool_name}"
        return await self._generate(
            arguments.get("path", "."),
            arguments.get("since", "HEAD~20"),
            arguments.get("format", "grouped"),
        )

    async def _generate(self, path: str, since: str, fmt: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                path,
                "log",
                f"{since}..HEAD",
                "--oneline",
                "--no-merges",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                return f"Git error: {stderr.decode().strip()}"
            commits = stdout.decode().strip().splitlines()
        except Exception as e:
            return f"Error: {e}"

        if not commits:
            return "No commits found."

        if fmt == "simple":
            return "## Changelog\n\n" + "\n".join(f"- {c.split(' ', 1)[1]}" for c in commits)

        # Group by conventional commit type
        groups: dict[str, list[str]] = {
            "Features": [],
            "Fixes": [],
            "Refactoring": [],
            "Documentation": [],
            "Tests": [],
            "Chores": [],
            "Other": [],
        }
        type_map = {
            "feat": "Features",
            "fix": "Fixes",
            "refactor": "Refactoring",
            "docs": "Documentation",
            "test": "Tests",
            "chore": "Chores",
        }

        for commit in commits:
            parts = commit.split(" ", 1)
            msg = parts[1] if len(parts) > 1 else parts[0]
            m = re.match(r"(\w+)(?:\(.+?\))?:\s*(.*)", msg)
            if m:
                ctype = type_map.get(m.group(1).lower(), "Other")
                groups[ctype].append(m.group(2))
            else:
                groups["Other"].append(msg)

        if fmt == "keepachangelog":
            from datetime import date

            lines = [f"## [{date.today()}]\n"]
            section_map = {
                "Features": "Added",
                "Fixes": "Fixed",
                "Refactoring": "Changed",
                "Documentation": "Documentation",
                "Other": "Other",
            }
            for section, label in section_map.items():
                items = groups.get(section, [])
                if items:
                    lines.append(f"### {label}")
                    for item in items:
                        lines.append(f"- {item}")
                    lines.append("")
            return "\n".join(lines)

        # Default: grouped
        lines = ["## Changelog\n"]
        for section, items in groups.items():
            if items:
                lines.append(f"### {section}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")
        return "\n".join(lines)

"""Git skill — structured version control operations."""

from __future__ import annotations

import asyncio
from typing import Any

from towel.skills.base import Skill, ToolDefinition

MAX_OUTPUT = 30_000


class GitSkill(Skill):
    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return "Git version control operations"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="git_status",
                description=(
                    "Show working tree status: branch, staged "
                    "changes, unstaged changes, untracked files"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                },
            ),
            ToolDefinition(
                name="git_diff",
                description=(
                    "Show file diffs. Use staged=true for "
                    "staged changes, or specify a file path."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repo path (default: cwd)"},
                        "staged": {
                            "type": "boolean",
                            "description": "Show staged changes (default: false)",
                        },
                        "file": {
                            "type": "string",
                            "description": "Specific file to diff (optional)",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="git_log",
                description="Show recent commit history",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repo path (default: cwd)"},
                        "limit": {
                            "type": "integer",
                            "description": "Number of commits (default: 10)",
                        },
                        "oneline": {
                            "type": "boolean",
                            "description": "Compact format (default: false)",
                        },
                    },
                },
            ),
            ToolDefinition(
                name="git_commit",
                description="Stage all changes and commit with a message",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                        "path": {"type": "string", "description": "Repo path (default: cwd)"},
                        "files": {
                            "type": "string",
                            "description": (
                                "Specific files to stage "
                                "(space-separated, default: all)"
                            ),
                        },
                    },
                    "required": ["message"],
                },
            ),
            ToolDefinition(
                name="git_branch",
                description="List, create, or switch branches",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Repo path (default: cwd)"},
                        "create": {
                            "type": "string",
                            "description": "Create and switch to a new branch",
                        },
                        "switch": {"type": "string", "description": "Switch to an existing branch"},
                    },
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        path = arguments.get("path", ".")

        match tool_name:
            case "git_status":
                return await self._status(path)
            case "git_diff":
                return await self._diff(path, arguments.get("staged", False), arguments.get("file"))
            case "git_log":
                return await self._log(
                    path, arguments.get("limit", 10), arguments.get("oneline", False)
                )
            case "git_commit":
                return await self._commit(path, arguments["message"], arguments.get("files"))
            case "git_branch":
                return await self._branch(path, arguments.get("create"), arguments.get("switch"))
            case _:
                return f"Unknown tool: {tool_name}"

    async def _run(self, *args: str, cwd: str = ".") -> tuple[str, str, int]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        return (
            stdout.decode("utf-8", errors="replace")[:MAX_OUTPUT],
            stderr.decode("utf-8", errors="replace")[:MAX_OUTPUT],
            proc.returncode or 0,
        )

    async def _status(self, path: str) -> str:
        out, err, rc = await self._run("git", "status", "--porcelain=v1", "-b", cwd=path)
        if rc != 0:
            return f"Error: {err or out}"

        lines = out.strip().splitlines()
        branch = ""
        staged, unstaged, untracked = [], [], []

        for line in lines:
            if line.startswith("## "):
                branch = line[3:].split("...")[0]
                continue
            if len(line) < 2:
                continue
            x, y = line[0], line[1]
            fname = line[3:]
            if x == "?":
                untracked.append(fname)
            else:
                if x != " " and x != "?":
                    staged.append(f"{x} {fname}")
                if y != " " and y != "?":
                    unstaged.append(f"{y} {fname}")

        parts = [f"Branch: {branch}"]
        if staged:
            parts.append(f"Staged ({len(staged)}):\n  " + "\n  ".join(staged))
        if unstaged:
            parts.append(f"Unstaged ({len(unstaged)}):\n  " + "\n  ".join(unstaged))
        if untracked:
            parts.append(f"Untracked ({len(untracked)}):\n  " + "\n  ".join(untracked))
        if not staged and not unstaged and not untracked:
            parts.append("Working tree clean.")

        return "\n".join(parts)

    async def _diff(self, path: str, staged: bool, file: str | None) -> str:
        args = ["git", "diff"]
        if staged:
            args.append("--cached")
        args.append("--stat")
        if file:
            args.extend(["--", file])
        stat_out, _, _ = await self._run(*args, cwd=path)

        # Also get the actual diff (limited)
        args2 = ["git", "diff"]
        if staged:
            args2.append("--cached")
        if file:
            args2.extend(["--", file])
        diff_out, err, rc = await self._run(*args2, cwd=path)
        if rc != 0:
            return f"Error: {err or diff_out}"

        if not diff_out.strip():
            return "No changes." if not staged else "No staged changes."

        return f"{stat_out.strip()}\n\n{diff_out}"

    async def _log(self, path: str, limit: int, oneline: bool) -> str:
        limit = min(limit, 50)
        if oneline:
            args = ["git", "log", f"-{limit}", "--oneline"]
        else:
            args = ["git", "log", f"-{limit}", "--format=%h %ad %an%n  %s", "--date=short"]
        out, err, rc = await self._run(*args, cwd=path)
        if rc != 0:
            return f"Error: {err or out}"
        return out.strip() or "No commits."

    async def _commit(self, path: str, message: str, files: str | None) -> str:
        # Stage
        if files:
            for f in files.split():
                _, err, rc = await self._run("git", "add", f, cwd=path)
                if rc != 0:
                    return f"Failed to stage {f}: {err}"
        else:
            _, err, rc = await self._run("git", "add", "-A", cwd=path)
            if rc != 0:
                return f"Failed to stage: {err}"

        # Commit
        out, err, rc = await self._run("git", "commit", "-m", message, cwd=path)
        if rc != 0:
            if "nothing to commit" in (out + err).lower():
                return "Nothing to commit — working tree clean."
            return f"Commit failed: {err or out}"

        return out.strip()

    async def _branch(self, path: str, create: str | None, switch: str | None) -> str:
        if create:
            out, err, rc = await self._run("git", "checkout", "-b", create, cwd=path)
            if rc != 0:
                return f"Failed to create branch: {err}"
            return f"Created and switched to branch: {create}"

        if switch:
            out, err, rc = await self._run("git", "checkout", switch, cwd=path)
            if rc != 0:
                return f"Failed to switch: {err}"
            return f"Switched to branch: {switch}"

        # List branches
        out, err, rc = await self._run("git", "branch", "-v", cwd=path)
        if rc != 0:
            return f"Error: {err or out}"
        return out.strip()

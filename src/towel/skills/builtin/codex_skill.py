"""Codex skill — external compaction via the local Codex subscription CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_DEFAULT_MODEL = "gpt-5-mini"


def _build_compaction_prompt(text: str, goal: str, max_words: int) -> str:
    prompt = (
        "Compact the following text into a dense plain-text working summary. Preserve only the "
        "highest-value context: user intent, open tasks, decisions made, relevant files or paths, "
        "commands or tool results, errors, constraints, and concrete next steps. Omit chit-chat "
        "and repetition."
    )
    if goal:
        prompt += f"\n\nFocus especially on: {goal}"
    prompt += f"\n\nKeep the result under about {max_words} words."
    prompt += f"\n\nText to compact:\n\n{text}"
    return prompt


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def _codex_logged_in() -> bool:
    try:
        result = subprocess.run(
            ["codex", "login", "status"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return False
    status_text = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0 and "Logged in" in status_text


def _run_codex_exec(prompt: str, model: str) -> str:
    fd, output_path = tempfile.mkstemp(prefix="towel-codex-compact-", suffix=".txt")
    os.close(fd)

    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-o",
        output_path,
        "-m",
        model,
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or f"exit {result.returncode}"
            return f"Error calling Codex compactor: {detail}"

        text = Path(output_path).read_text(encoding="utf-8").strip()
        return text or "(empty response)"
    except Exception as e:
        return f"Error calling Codex compactor: {e}"
    finally:
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass


def codex_compact_available() -> bool:
    """Return whether subscription-backed Codex compaction is usable."""
    return _codex_available() and _codex_logged_in()


def codex_compact_text(
    text: str, goal: str = "", max_words: int = 220, model: str | None = None
) -> str:
    """Compact text using the local Codex subscription CLI."""
    if not _codex_available():
        return "Codex CLI not found. Install Codex to use subscription-backed compaction."
    if not _codex_logged_in():
        return "Codex is not logged in. Run `codex login` to use subscription-backed compaction."
    if not text.strip():
        return "Nothing to compact."

    selected_model = str(model or os.environ.get("TOWEL_CODEX_MODEL") or _DEFAULT_MODEL)
    prompt = _build_compaction_prompt(text, goal.strip(), int(max_words or 220))
    return _run_codex_exec(prompt, selected_model)


class CodexSkill(Skill):
    @property
    def name(self) -> str:
        return "codex"

    @property
    def description(self) -> str:
        return "Use the local Codex subscription CLI for dense conversation or text compaction"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="codex_compact",
                description=(
                    "Use the local Codex subscription to compress long text or conversation "
                    "history into a dense working summary that preserves tasks, decisions, "
                    "errors, files, and next steps."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text or conversation history to compact",
                        },
                        "goal": {
                            "type": "string",
                            "description": (
                                "Optional focus for the summary, e.g. debugging state or open tasks"
                            ),
                        },
                        "max_words": {
                            "type": "integer",
                            "description": "Approximate word cap for the compacted summary",
                        },
                        "model": {
                            "type": "string",
                            "description": "Optional Codex model override",
                        },
                    },
                    "required": ["text"],
                },
            )
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "codex_compact":
            return f"Unknown tool: {tool_name}"
        return await self._compact(arguments)

    async def _compact(self, args: dict[str, Any]) -> str:
        text = str(args["text"]).strip()
        goal = str(args.get("goal", "")).strip()
        max_words = int(args.get("max_words", 220) or 220)
        model = str(args.get("model") or os.environ.get("TOWEL_CODEX_MODEL") or _DEFAULT_MODEL)
        return codex_compact_text(text, goal=goal, max_words=max_words, model=model)

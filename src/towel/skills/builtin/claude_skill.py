"""Claude skill — ask Claude for help, verification, or delegate complex tasks."""

from __future__ import annotations

import logging
from typing import Any

from towel.skills.base import Skill, ToolDefinition

log = logging.getLogger("towel.skills.claude")

_CLIENT = None


def _get_client() -> Any:
    """Lazy-init a shared Anthropic client using Claude Code's OAuth token."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    import uuid

    import anthropic

    from towel.agent.claude_runtime import _read_oauth_token

    token = _read_oauth_token()
    _CLIENT = anthropic.Anthropic(
        auth_token=token,
        default_headers={
            "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
            "x-app": "cli",
            "User-Agent": "claude-cli/2.1.89 (external, cli)",
            "X-Claude-Code-Session-Id": uuid.uuid4().hex,
        },
    )
    return _CLIENT


class ClaudeSkill(Skill):
    @property
    def name(self) -> str:
        return "claude"

    @property
    def description(self) -> str:
        return (
            "Ask Claude (Anthropic) for help — delegate complex "
            "reasoning, get a second opinion, or run Claude as a "
            "sub-agent for multi-step tasks"
        )

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="ask_claude",
                description=(
                    "Ask Claude a question and get a response. Use "
                    "this when you need a second opinion, want to "
                    "verify your reasoning, or need help with "
                    "something complex."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question or prompt for Claude",
                        },
                        "model": {
                            "type": "string",
                            "description": (
                                "Model to use: sonnet (default), opus, haiku"
                            ),
                        },
                    },
                    "required": ["question"],
                },
            ),
            ToolDefinition(
                name="claude_agent",
                description=(
                    "Run Claude as a sub-agent to complete a complex "
                    "multi-step task. Give it a goal and optional "
                    "context. Returns Claude's full response. Use for "
                    "tasks like code review, research, analysis, "
                    "writing, or debugging."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": (
                                "The task for Claude to complete"
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": (
                                "Additional context — code, file "
                                "contents, error messages, etc."
                            ),
                        },
                        "model": {
                            "type": "string",
                            "description": (
                                "Model to use: sonnet (default), opus, haiku"
                            ),
                        },
                    },
                    "required": ["task"],
                },
            ),
            ToolDefinition(
                name="claude_verify",
                description=(
                    "Ask Claude to verify or fact-check a claim, "
                    "solution, or piece of code. Returns Claude's "
                    "assessment of correctness."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": (
                                "The claim, solution, or code to verify"
                            ),
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context if needed",
                        },
                    },
                    "required": ["claim"],
                },
            ),
        ]

    async def execute(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        if tool_name == "ask_claude":
            return await self._ask(arguments)
        elif tool_name == "claude_agent":
            return await self._agent(arguments)
        elif tool_name == "claude_verify":
            return await self._verify(arguments)
        return f"Unknown tool: {tool_name}"

    async def _ask(self, args: dict[str, Any]) -> str:
        question = args["question"]
        model = self._resolve_model(args.get("model", "sonnet"))

        try:
            client = _get_client()
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                system=(
                    "x-anthropic-billing-header: "
                    "cc_version=2.1.89; cc_entrypoint=cli;\n\n"
                    "You are a helpful assistant. Give clear, "
                    "concise answers."
                ),
                messages=[{"role": "user", "content": question}],
            )
            return self._extract_text(resp)
        except Exception as e:
            log.error(f"ask_claude failed: {e}")
            return f"Error calling Claude: {e}"

    async def _agent(self, args: dict[str, Any]) -> str:
        task = args["task"]
        context = args.get("context", "")
        model = self._resolve_model(args.get("model", "sonnet"))

        prompt = task
        if context:
            prompt = f"{task}\n\n## Context\n\n{context}"

        try:
            client = _get_client()
            resp = client.messages.create(
                model=model,
                max_tokens=8192,
                system=(
                    "x-anthropic-billing-header: "
                    "cc_version=2.1.89; cc_entrypoint=cli;\n\n"
                    "You are an expert assistant completing a task. "
                    "Be thorough and detailed. Think step by step. "
                    "If the task involves code, include working code "
                    "in your response."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            return self._extract_text(resp)
        except Exception as e:
            log.error(f"claude_agent failed: {e}")
            return f"Error calling Claude: {e}"

    async def _verify(self, args: dict[str, Any]) -> str:
        claim = args["claim"]
        context = args.get("context", "")

        prompt = f"Please verify the following:\n\n{claim}"
        if context:
            prompt += f"\n\nAdditional context:\n{context}"

        try:
            client = _get_client()
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=(
                    "x-anthropic-billing-header: "
                    "cc_version=2.1.89; cc_entrypoint=cli;\n\n"
                    "You are a verification assistant. Carefully "
                    "analyze the claim, solution, or code provided. "
                    "State whether it is correct, partially correct, "
                    "or incorrect. Explain any issues found. Be "
                    "precise and cite specific problems."
                ),
                messages=[{"role": "user", "content": prompt}],
            )
            return self._extract_text(resp)
        except Exception as e:
            log.error(f"claude_verify failed: {e}")
            return f"Error calling Claude: {e}"

    @staticmethod
    def _resolve_model(alias: str) -> str:
        from towel.agent.claude_runtime import MODEL_ALIASES

        return MODEL_ALIASES.get(alias, alias)

    @staticmethod
    def _extract_text(resp: Any) -> str:
        parts = []
        for block in resp.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts) if parts else "(empty response)"

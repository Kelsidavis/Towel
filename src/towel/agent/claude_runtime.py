"""Claude runtime — direct Anthropic API via Max subscription OAuth tokens.

Reads the OAuth access token from Claude Code's macOS keychain entry,
then calls the Anthropic messages API directly using the Python SDK.
No subprocess, no Claude Code process — just lightweight HTTP streaming.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from towel.agent.context import estimate_output_reserve, maybe_compact_conversation
from towel.agent.conversation import Conversation, Message, Role
from towel.agent.events import AgentEvent
from towel.agent.instance_lock import acquire_runtime_lock
from towel.agent.runtime import format_tool_feedback, tool_result_is_error
from towel.agent.tool_parser import parse_tool_calls
from towel.config import TowelConfig
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.agent.claude")

MAX_TOOL_ITERATIONS = 999

# Model alias map — short names to full model IDs
MODEL_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

# Keychain service name used by Claude Code to store OAuth credentials
_KEYCHAIN_SERVICE = "Claude Code-credentials"


def _read_oauth_token() -> str:
    """Read the OAuth access token from Claude Code's macOS keychain entry.

    Claude Code stores credentials in the macOS keychain under the service
    name "Claude Code-credentials". The data is a JSON blob with structure:
        {"claudeAiOauth": {"accessToken": "sk-ant-oat01-...", ...}}
    """
    import os

    username = os.environ.get("USER") or os.getlogin()

    result = subprocess.run(
        ["security", "find-generic-password", "-a", username, "-w", "-s", _KEYCHAIN_SERVICE],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Fallback: try plaintext credentials file
        from pathlib import Path

        creds_path = Path.home() / ".claude" / ".credentials.json"
        if creds_path.exists():
            data = json.loads(creds_path.read_text())
        else:
            raise RuntimeError(
                "No Claude OAuth credentials found. Run `claude` and log in with "
                "your Max subscription first."
            )
    else:
        data = json.loads(result.stdout.strip())

    oauth = data.get("claudeAiOauth", {})
    token = oauth.get("accessToken")
    if not token:
        raise RuntimeError(
            "Claude OAuth token not found in credentials. Run `claude` and "
            "log in with your Max subscription."
        )

    scopes = oauth.get("scopes", [])
    if "user:inference" not in scopes:
        raise RuntimeError(
            "Claude OAuth token lacks 'user:inference' scope. "
            "Re-authenticate with `claude` to get inference access."
        )

    return token


def _resolve_model(model: str) -> str:
    """Resolve a model alias to a full model ID."""
    return MODEL_ALIASES.get(model, model)


@dataclass
class ClaudeGenerationResult:
    """Result of a single Claude API generation."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class ClaudeCodeRuntime:
    """Agent runtime that calls the Anthropic API directly via Max subscription.

    Reads OAuth tokens from Claude Code's keychain storage and uses the
    anthropic Python SDK to call the messages API. No subprocess overhead —
    just HTTP streaming.

    Usage:
        towel chat --backend claude                  # Sonnet (default)
        towel chat --backend claude --claude-model opus
    """

    def __init__(
        self,
        config: TowelConfig,
        skills: SkillRegistry | None = None,
        memory: Any | None = None,
        model: str = "sonnet",
    ) -> None:
        self.config = config
        self.skills = skills or SkillRegistry()
        self.memory = memory
        self.project_context: str | None = None  # Override from coordinator
        self.model = _resolve_model(model)
        self._client: Any = None  # anthropic.AsyncAnthropic
        self._loaded = False
        self._cancel_flag = False

    def cancel(self) -> None:
        """Signal the current generation to stop."""
        self._cancel_flag = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag

    async def load_model(self) -> None:
        """Initialize the Anthropic client with OAuth token from keychain."""
        if self._loaded:
            return

        acquire_runtime_lock()

        import anthropic

        token = _read_oauth_token()
        session_id = uuid.uuid4().hex
        self._client = anthropic.AsyncAnthropic(
            auth_token=token,
            default_headers={
                # These headers identify the request as Claude Code usage,
                # which bills against the Max subscription allowance rather
                # than as separate API usage.
                "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
                "x-app": "cli",
                "User-Agent": "claude-cli/2.1.89 (external, cli)",
                "X-Claude-Code-Session-Id": session_id,
            },
        )
        log.info(f"Claude API client ready (model: {self.model})")
        self._loaded = True

    def _build_system_prompt(self) -> str:
        """Build system prompt with towel's identity, context, and tool instructions."""
        # Billing attribution header — injected as a system prompt block,
        # not an HTTP header. This is how Claude Code tells the API to bill
        # the request against the Max subscription allowance.
        system = "x-anthropic-billing-header: cc_version=2.1.89; cc_entrypoint=cli;\n\n"

        system += self.config.identity + (
            "\n\nAfter using a tool, always answer the user's original question "
            "based on the tool result. Do not just acknowledge the tool output — "
            "use it to provide a direct, helpful answer. If you changed something "
            "or verified something, explicitly report that back to the user."
        )

        # Inject project context — use coordinator-provided override if set
        if self.project_context:
            system += self.project_context
        else:
            from towel.agent.project import load_project_context

            project_block = load_project_context()
            if project_block:
                system += project_block

        # Inject persistent memories
        if self.memory:
            memory_block = self.memory.to_prompt_block()
            if memory_block:
                system += memory_block

        # Tool instructions — Claude will emit <tool_call> tags that we parse
        tools = self.skills.tool_definitions()
        if tools:
            tool_lines = []
            for t in tools:
                params = t.get("parameters", {})
                props = params.get("properties", {})
                if props:
                    param_names = ", ".join(props.keys())
                    tool_lines.append(f"- {t['name']}({param_names}): {t['description']}")
                else:
                    tool_lines.append(f"- {t['name']}(): {t['description']}")

            tool_names = [t["name"] for t in tools]
            tool_name_list = ", ".join(tool_names)

            system += (
                "\n\n# Tools\n\n"
                "You may call one or more functions to assist with the user query.\n\n"
                "Available tools:\n" + "\n".join(tool_lines) + "\n\n"
                "For each function call, return a json object with function name and "
                "arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": <function-name>, "arguments": <args-json-object>}\n'
                "</tool_call>\n\n"
                f"The ONLY supported tool names are: {tool_name_list}\n\n"
                "IMPORTANT:\n"
                "- Only call functions from the list above. Do NOT invent or guess "
                "function names. If a tool you want is not listed, it does not exist.\n"
                "- Always use the exact <tool_call> format shown above.\n"
                "- When using a tool, prefer emitting just the tool call instead of "
                "narrating that you are about to check something.\n"
                "- After tool results arrive, either give the concrete answer or make "
                "one corrected retry. Do not repeat vague status updates.\n"
                "- If no tool is needed, respond directly without tool calls."
            )
        return system

    def _build_messages(self, conversation: Conversation) -> list[dict[str, str]]:
        """Convert towel conversation to Anthropic messages format."""
        existing_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in conversation.messages
        ]
        output_reserve = estimate_output_reserve(
            existing_messages,
            configured_max_tokens=self.config.model.max_tokens,
        )
        maybe_compact_conversation(
            conversation,
            system_content=self._build_system_prompt(),
            context_window=self.config.model.context_window,
            max_output_tokens=output_reserve,
        )
        messages: list[dict[str, str]] = []
        for msg in conversation.messages:
            if msg.role == Role.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == Role.TOOL:
                # Tool results go back as user messages for the next turn
                messages.append(
                    {"role": "user", "content": f"<tool_result>\n{msg.content}\n</tool_result>"}
                )
        return messages

    def build_inference_request(self, conversation: Conversation) -> dict[str, Any]:
        """Build a worker-safe Anthropic payload for this conversation."""
        return {
            "mode": "anthropic_messages",
            "system": self._build_system_prompt(),
            "messages": self._build_messages(conversation),
        }

    async def generate(self, conversation: Conversation) -> ClaudeGenerationResult:
        """Run a single generation pass (non-streaming)."""
        if not self._loaded:
            await self.load_model()

        return await self.generate_from_request(self.build_inference_request(conversation))

    async def generate_from_request(self, request: dict[str, Any]) -> ClaudeGenerationResult:
        """Generate from a prebuilt Anthropic messages payload."""
        if not self._loaded:
            await self.load_model()

        if request.get("mode") != "anthropic_messages":
            raise ValueError(f"Unsupported inference mode: {request.get('mode')}")

        response = await self._client.messages.create(
            model=self.model,
            max_tokens=self.config.model.max_tokens,
            system=request["system"],
            messages=request["messages"],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        return ClaudeGenerationResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def stream(self, conversation: Conversation) -> AsyncIterator[str]:
        """Stream generation token by token via the Anthropic streaming API."""
        if not self._loaded:
            await self.load_model()

        async for text in self.stream_from_request(self.build_inference_request(conversation)):
            yield text

    async def stream_from_request(self, request: dict[str, Any]) -> AsyncIterator[str]:
        """Stream generation from a prebuilt Anthropic messages payload."""
        if not self._loaded:
            await self.load_model()

        if request.get("mode") != "anthropic_messages":
            raise ValueError(f"Unsupported inference mode: {request.get('mode')}")

        async with self._client.messages.stream(
            model=self.model,
            max_tokens=self.config.model.max_tokens,
            system=request["system"],
            messages=request["messages"],
        ) as stream:
            async for text in stream.text_stream:
                if self._cancel_flag:
                    break
                yield text

    async def step(self, conversation: Conversation) -> Message:
        """Run one full agent step: generate -> maybe call tools -> return response."""
        for iteration in range(MAX_TOOL_ITERATIONS):
            result = await self.generate(conversation)
            tool_calls, remaining_text = parse_tool_calls(result.text)

            if not tool_calls:
                return Message(
                    role=Role.ASSISTANT,
                    content=result.text,
                    metadata={"backend": "claude", "model": self.model},
                )

            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            for tc in tool_calls:
                log.info(f"Tool call: {tc.name}({tc.arguments})")
                try:
                    tool_result = await self.skills.execute_tool(tc.name, tc.arguments)
                    result_str = (
                        str(tool_result) if not isinstance(tool_result, str) else tool_result
                    )
                    is_error = tool_result_is_error(result_str)
                except Exception as e:
                    result_str = f"Error executing {tc.name}: {e}"
                    is_error = True
                    log.error(result_str)

                conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )

        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        return Message(
            role=Role.ASSISTANT,
            content=remaining_text or "I've reached my tool execution limit for this turn.",
            metadata={"backend": "claude", "max_iterations": True},
        )

    async def step_streaming(self, conversation: Conversation) -> AsyncIterator[AgentEvent]:
        """Run a full agent step, yielding events as they happen."""
        self._cancel_flag = False

        for iteration in range(MAX_TOOL_ITERATIONS):
            full_text = ""
            async for chunk in self.stream(conversation):
                full_text += chunk
                yield AgentEvent.token(chunk)

            if self._cancel_flag:
                if full_text.strip():
                    conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.cancelled(
                    full_text,
                    metadata={"reason": "user_cancelled"},
                )
                self._cancel_flag = False
                return

            tool_calls, remaining_text = parse_tool_calls(full_text)

            if not tool_calls:
                conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.complete(
                    full_text,
                    metadata={"backend": "claude", "model": self.model},
                )
                return

            # Tool call loop
            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            for tc in tool_calls:
                if self._cancel_flag:
                    yield AgentEvent.cancelled(
                        remaining_text or "",
                        metadata={"reason": "user_cancelled"},
                    )
                    self._cancel_flag = False
                    return

                log.info(f"Tool call: {tc.name}({tc.arguments})")
                yield AgentEvent.tool_call(tc.name, tc.arguments)

                try:
                    tool_result = await self.skills.execute_tool(tc.name, tc.arguments)
                    result_str = (
                        str(tool_result) if not isinstance(tool_result, str) else tool_result
                    )
                    is_error = tool_result_is_error(result_str)
                except Exception as e:
                    result_str = f"Error executing {tc.name}: {e}"
                    is_error = True
                    log.error(result_str)

                yield AgentEvent.tool_result(tc.name, result_str)
                conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )

        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        yield AgentEvent.complete(
            remaining_text or "I've reached my tool execution limit for this turn.",
            metadata={"backend": "claude", "max_iterations": True},
        )

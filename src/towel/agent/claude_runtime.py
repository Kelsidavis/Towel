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

from towel.agent.context import (
    estimate_output_reserve,
    maybe_compact_conversation,
    select_context_window,
)
from towel.agent.conversation import Conversation, Message, Role
from towel.agent.events import AgentEvent
from towel.agent.instance_lock import acquire_runtime_lock
from towel.agent.runtime import (
    AUTONOMY_NUDGE,
    EMPTY_TEXT_FALLBACK,
    MAX_AUTONOMY_NUDGES,
    MAX_GOAL_NUDGES,
    TOOL_ERROR_NUDGE,
    TOOL_LOOP_REPEAT_LIMIT,
    _check_tool_loop,
    _tool_call_fingerprint,
    format_tool_feedback,
    looks_like_goal_incomplete,
    looks_like_unfulfilled_intent,
    summarize_tool_trace,
    tool_result_is_error,
)
from towel.agent.tool_parser import ToolCall, parse_tool_calls
from towel.agent.tools_payload import tools_as_anthropic
from towel.config import TowelConfig
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.agent.claude")

MAX_TOOL_ITERATIONS = 999


def _extract_anthropic_tool_calls(content_blocks: Any) -> list[ToolCall]:
    """Extract ``tool_use`` content blocks from an Anthropic Messages response.

    Each ``tool_use`` block carries ``id`` (the Anthropic ``tool_use_id`` we
    would need for a structured tool_result), ``name``, and ``input`` (an
    already-decoded dict of arguments).
    """
    calls: list[ToolCall] = []
    for block in content_blocks or []:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if block_type != "tool_use":
            continue
        name = getattr(block, "name", None) or (
            block.get("name") if isinstance(block, dict) else None
        )
        args = getattr(block, "input", None) or (
            block.get("input") if isinstance(block, dict) else None
        ) or {}
        if not isinstance(args, dict):
            args = {}
        if name:
            calls.append(ToolCall(name=name, arguments=args, raw=str(block)))
    return calls


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

    try:
        username = os.environ.get("USER") or os.getlogin()
    except (OSError, RuntimeError):
        username = "unknown"

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
    tool_calls: list[ToolCall] = field(default_factory=list)
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
        # Anthropic Messages API has first-class tool support; always enable it
        # when any tools are registered.
        self._native_tools_supported: bool = True
        self._last_stream_tool_calls: list[ToolCall] = []
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

    def _build_system_prompt(
        self,
        include_tools_section: bool = True,
        query: str | None = None,
        tools_available: bool = True,
    ) -> str:
        """Build system prompt with towel's identity, context, and tool instructions.

        When ``include_tools_section`` is False, the per-tool listing and call-format
        spec are omitted — used when the Anthropic ``tools=`` parameter is in play,
        in which case Claude already knows the tool schemas structurally.

        ``query`` is the current user turn used to rank persistent
        memories; ``None`` keeps the legacy full-dump behavior.
        """
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

        # Inject persistent memories — ranked when the caller supplied
        # the current user turn, full dump otherwise.
        if self.memory:
            memory_block = self.memory.to_prompt_block(query=query)
            if memory_block:
                system += memory_block

        tools = self.skills.tool_definitions() if tools_available else []
        if tools:
            if include_tools_section:
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
            else:
                system += (
                    "\n\n# Tool-use rules\n\n"
                    "You have access to a set of tools provided via the Anthropic "
                    "tools API. Use them when they help answer the user.\n\n"
                    "IMPORTANT:\n"
                    "- Only call the tools you have been given. Do NOT invent or "
                    "guess tool names.\n"
                    "- When using a tool, prefer just calling it instead of narrating "
                    "that you are about to check something.\n"
                    "- After tool results arrive, give the concrete answer or make "
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
        query = conversation.latest_user_query()
        from towel.agent.capture import run_capture_hooks
        run_capture_hooks(query, memory=self.memory, config=self.config, runtime=self)
        system_content = self._build_system_prompt(
            include_tools_section=not self._native_tools_supported,
            query=query,
        )
        effective_context_window = self.config.model.context_window
        if getattr(self.config.model, "auto_context", True):
            effective_context_window = select_context_window(
                system_content,
                existing_messages,
                configured_context_window=self.config.model.context_window,
                min_context_window=getattr(self.config.model, "min_context_window", 32768),
                max_output_tokens=output_reserve,
            )
        maybe_compact_conversation(
            conversation,
            system_content=system_content,
            context_window=effective_context_window,
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
        from towel.nodes.roles import classify_task_type, task_needs_tools

        use_native = self._native_tools_supported
        query = conversation.latest_user_query()
        from towel.agent.capture import run_capture_hooks
        run_capture_hooks(query, memory=self.memory, config=self.config, runtime=self)

        task_type = classify_task_type(query)
        wants_tools = task_needs_tools(task_type)

        request: dict[str, Any] = {
            "mode": "anthropic_messages",
            "system": self._build_system_prompt(
                include_tools_section=not use_native,
                query=query,
                tools_available=wants_tools,
            ),
            "messages": self._build_messages(conversation),
        }
        if use_native and wants_tools:
            native_tools = tools_as_anthropic(self.skills.tool_definitions())
            if native_tools:
                request["tools"] = native_tools
        return request

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

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.model.max_tokens,
            "system": request["system"],
            "messages": request["messages"],
        }
        if request.get("tools"):
            create_kwargs["tools"] = request["tools"]

        response = await self._client.messages.create(**create_kwargs)

        text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text += block.text

        return ClaudeGenerationResult(
            text=text,
            tool_calls=_extract_anthropic_tool_calls(response.content),
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

        stream_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.model.max_tokens,
            "system": request["system"],
            "messages": request["messages"],
        }
        if request.get("tools"):
            stream_kwargs["tools"] = request["tools"]

        self._last_stream_tool_calls = []
        self._last_stream_input_tokens = 0
        self._last_stream_output_tokens = 0
        async with self._client.messages.stream(**stream_kwargs) as stream:
            async for text in stream.text_stream:
                if self._cancel_flag:
                    break
                yield text
            try:
                final = await stream.get_final_message()
            except Exception as exc:
                log.debug("Anthropic stream.get_final_message failed: %s", exc)
                return
            self._last_stream_tool_calls = _extract_anthropic_tool_calls(final.content)
            if hasattr(final, "usage") and final.usage:
                self._last_stream_input_tokens = getattr(final.usage, "input_tokens", 0)
                self._last_stream_output_tokens = getattr(final.usage, "output_tokens", 0)

    async def step(self, conversation: Conversation) -> Message:
        """Run one full agent step: generate -> maybe call tools -> return response."""
        loop_fingerprints: list[str] = []
        stuck_call_name: str | None = None
        autonomy_nudges = 0
        goal_nudges = 0
        tool_trace: list[dict[str, Any]] = []
        remaining_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(MAX_TOOL_ITERATIONS):
            result = await self.generate(conversation)
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens
            if result.tool_calls:
                tool_calls = result.tool_calls
                remaining_text = result.text
            else:
                tool_calls, remaining_text = parse_tool_calls(result.text)

            if not tool_calls:
                if (
                    autonomy_nudges < MAX_AUTONOMY_NUDGES
                    and looks_like_unfulfilled_intent(result.text)
                ):
                    autonomy_nudges += 1
                    log.info("autonomy nudge %d: model narrated without acting",
                             autonomy_nudges)
                    conversation.add(Role.ASSISTANT, result.text)
                    conversation.add(Role.USER, AUTONOMY_NUDGE)
                    continue
                goal_nudge = looks_like_goal_incomplete(result.text, tool_trace)
                if goal_nudges < MAX_GOAL_NUDGES and goal_nudge:
                    goal_nudges += 1
                    log.info("goal-completion nudge %d: %s", goal_nudges,
                             "unaddressed errors" if goal_nudge is TOOL_ERROR_NUDGE
                             else "premature question")
                    conversation.add(Role.ASSISTANT, result.text)
                    conversation.add(Role.USER, goal_nudge)
                    continue
                content = result.text
                meta: dict[str, Any] = {
                    "backend": "claude",
                    "model": self.model,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                }
                if not content.strip() and tool_trace:
                    content = summarize_tool_trace(tool_trace)
                    meta["synthesized_summary"] = True
                elif not content.strip():
                    content = EMPTY_TEXT_FALLBACK
                    meta["empty_text_fallback"] = True
                return Message(
                    role=Role.ASSISTANT,
                    content=content,
                    metadata=meta,
                )

            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            if _check_tool_loop(
                loop_fingerprints, _tool_call_fingerprint(tool_calls)
            ):
                log.warning(
                    "Claude agent tool-loop detected (%r repeated %d times)",
                    tool_calls[0].name, TOOL_LOOP_REPEAT_LIMIT,
                )
                stuck_call_name = tool_calls[0].name
                break

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

                if (
                    not is_error and self.memory
                    and getattr(self.config, "auto_capture", True)
                ):
                    try:
                        from towel.memory.auto_capture import apply_tool_result
                        apply_tool_result(tc.name, result_str, self.memory)
                    except Exception as exc:
                        log.debug("Tool-result capture skipped: %s", exc)

                conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )
                tool_trace.append({
                    "tool": tc.name,
                    "status": "error" if is_error else "ok",
                })

        if stuck_call_name is not None:
            stuck_msg = (
                f"I got stuck calling {stuck_call_name!r} repeatedly. "
                "Stopping to avoid burning more time on this turn."
            )
            return Message(
                role=Role.ASSISTANT,
                content=(remaining_text + "\n\n" + stuck_msg) if remaining_text else stuck_msg,
                metadata={"backend": "claude", "loop_detected": True},
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
        loop_fingerprints: list[str] = []
        stuck_call_name: str | None = None
        autonomy_nudges = 0
        goal_nudges = 0
        tool_trace: list[dict[str, Any]] = []
        remaining_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(MAX_TOOL_ITERATIONS):
            full_text = ""
            async for chunk in self.stream(conversation):
                full_text += chunk
                yield AgentEvent.token(chunk)
            total_input_tokens += self._last_stream_input_tokens
            total_output_tokens += self._last_stream_output_tokens

            if self._cancel_flag:
                if full_text.strip():
                    conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.cancelled(
                    full_text,
                    metadata={"reason": "user_cancelled"},
                )
                self._cancel_flag = False
                return

            if self._last_stream_tool_calls:
                tool_calls = list(self._last_stream_tool_calls)
                self._last_stream_tool_calls = []
                remaining_text = full_text
            else:
                tool_calls, remaining_text = parse_tool_calls(full_text)

            if not tool_calls:
                if (
                    autonomy_nudges < MAX_AUTONOMY_NUDGES
                    and looks_like_unfulfilled_intent(full_text)
                ):
                    autonomy_nudges += 1
                    log.info("autonomy nudge %d: model narrated without acting",
                             autonomy_nudges)
                    conversation.add(Role.ASSISTANT, full_text)
                    conversation.add(Role.USER, AUTONOMY_NUDGE)
                    continue
                goal_nudge = looks_like_goal_incomplete(full_text, tool_trace)
                if goal_nudges < MAX_GOAL_NUDGES and goal_nudge:
                    goal_nudges += 1
                    log.info("goal-completion nudge %d: %s", goal_nudges,
                             "unaddressed errors" if goal_nudge is TOOL_ERROR_NUDGE
                             else "premature question")
                    conversation.add(Role.ASSISTANT, full_text)
                    conversation.add(Role.USER, goal_nudge)
                    continue
                text = full_text
                meta: dict[str, Any] = {
                    "backend": "claude",
                    "model": self.model,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                }
                if not text.strip() and tool_trace:
                    text = summarize_tool_trace(tool_trace)
                    meta["synthesized_summary"] = True
                elif not text.strip():
                    text = EMPTY_TEXT_FALLBACK
                    meta["empty_text_fallback"] = True
                conversation.add(Role.ASSISTANT, text)
                yield AgentEvent.complete(text, metadata=meta)
                return

            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            if _check_tool_loop(
                loop_fingerprints, _tool_call_fingerprint(tool_calls)
            ):
                log.warning(
                    "Claude agent (streaming) tool-loop detected (%r repeated %d times)",
                    tool_calls[0].name, TOOL_LOOP_REPEAT_LIMIT,
                )
                stuck_call_name = tool_calls[0].name
                break

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

                if (
                    not is_error and self.memory
                    and getattr(self.config, "auto_capture", True)
                ):
                    try:
                        from towel.memory.auto_capture import apply_tool_result
                        apply_tool_result(tc.name, result_str, self.memory)
                    except Exception as exc:
                        log.debug("Tool-result capture skipped: %s", exc)

                yield AgentEvent.tool_result(tc.name, result_str)
                conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )
                tool_trace.append({
                    "tool": tc.name,
                    "status": "error" if is_error else "ok",
                })

        if stuck_call_name is not None:
            stuck_msg = (
                f"I got stuck calling {stuck_call_name!r} repeatedly. "
                "Stopping to avoid burning more time on this turn."
            )
            conversation.add(Role.ASSISTANT, stuck_msg)
            yield AgentEvent.complete(
                (remaining_text + "\n\n" + stuck_msg) if remaining_text else stuck_msg,
                metadata={"backend": "claude", "loop_detected": True},
            )
            return
        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        yield AgentEvent.complete(
            remaining_text or "I've reached my tool execution limit for this turn.",
            metadata={"backend": "claude", "max_iterations": True},
        )

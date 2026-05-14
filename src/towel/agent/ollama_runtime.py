"""Ollama runtime — local inference via the Ollama HTTP API.

Works on any platform where Ollama is installed (Linux, macOS, Windows).
Uses the /api/chat endpoint so Ollama applies the correct chat template
for the running model. Reports the ``ollama_chat`` inference mode.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from towel.agent.context import estimate_output_reserve, maybe_compact_conversation
from towel.agent.conversation import Conversation, Message, Role
from towel.agent.events import AgentEvent
from towel.agent.runtime import format_tool_feedback, tool_result_is_error
from towel.agent.tool_parser import ToolCall, parse_tool_calls
from towel.agent.tools_payload import tools_as_openai_functions
from towel.config import TowelConfig
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.agent.ollama")

MAX_TOOL_ITERATIONS = 999
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _normalize_ollama_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert Ollama's structured ``message.tool_calls`` entries to ``ToolCall``s.

    Each Ollama entry looks like
    ``{"id": "...", "function": {"name": ..., "arguments": {...}}}``.
    Arguments are already parsed into a dict by Ollama — no JSON-decoding needed.
    """
    calls: list[ToolCall] = []
    for entry in raw_calls or []:
        fn = entry.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments") or {}
        if not isinstance(args, dict):
            # Defensive: some Ollama builds return a JSON string for arguments.
            try:
                args = json.loads(args) if isinstance(args, str) else {}
            except json.JSONDecodeError:
                args = {}
        if name:
            calls.append(ToolCall(name=name, arguments=args, raw=json.dumps(entry)))
    return calls


@dataclass
class OllamaGenerationResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OllamaRuntime:
    """Agent runtime that calls a local Ollama daemon via /api/chat.

    Uses the chat endpoint so Ollama applies the correct chat template
    for whichever model is loaded. No MLX dependency — works on Linux,
    macOS, and anywhere Ollama runs.

    Usage:
        towel worker --master ws://... --backend ollama
        towel worker --master ws://... --backend ollama --ollama-url http://gpu-box:11434
    """

    def __init__(
        self,
        config: TowelConfig,
        skills: SkillRegistry | None = None,
        memory: Any | None = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
    ) -> None:
        self.config = config
        self.skills = skills or SkillRegistry()
        self.memory = memory
        self.project_context: str | None = None  # Override from coordinator
        self.ollama_url = ollama_url.rstrip("/")
        self._loaded = False
        self._native_tools_supported: bool | None = None
        self._last_stream_tool_calls: list[ToolCall] = []
        self._cancel_flag = False

    def cancel(self) -> None:
        self._cancel_flag = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag

    async def load_model(self) -> None:
        """Verify the Ollama daemon is reachable and the model is available."""
        if self._loaded:
            return

        model = self.config.model.name
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                resp.raise_for_status()
                available = [m["name"] for m in resp.json().get("models", [])]
                if available and not any(
                    m == model or m.startswith(model.split(":")[0]) for m in available
                ):
                    log.warning(
                        f"Model '{model}' not found in Ollama. "
                        f"Available: {available}. "
                        f"Pull it with: ollama pull {model}"
                    )
            except httpx.ConnectError as exc:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.ollama_url}. "
                    "Is `ollama serve` running?"
                ) from exc

        self._native_tools_supported = await self._detect_native_tools_support(model)
        log.info(
            "Ollama runtime ready (model: %s, url: %s, native tools: %s)",
            model,
            self.ollama_url,
            "enabled" if self._native_tools_supported else "disabled (fallback to text)",
        )
        self._loaded = True

    async def _detect_native_tools_support(self, model: str) -> bool:
        """Probe Ollama's /api/show for the ``tools`` capability on this model."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.ollama_url}/api/show", json={"name": model})
                resp.raise_for_status()
                caps = resp.json().get("capabilities") or []
                return "tools" in caps
        except Exception as exc:
            log.debug("Tool capability probe failed: %s", exc)
            return False

    def _build_system_prompt(
        self,
        include_tools_section: bool = True,
        query: str | None = None,
    ) -> str:
        """Build system prompt with identity, context, and tool instructions.

        When ``include_tools_section`` is False the per-tool listing and call-format
        spec are dropped — used when Ollama's native ``tools`` field is in play and
        the chat template (Qwen3, Llama 3.1+, etc.) renders the tools itself.

        ``query`` is the current user turn used to rank persistent
        memories; ``None`` keeps the legacy full-dump behavior.
        """
        system = self.config.identity + (
            "\n\nAfter using a tool, always answer the user's original question "
            "based on the tool result. Do not just acknowledge the tool output — "
            "use it to provide a direct, helpful answer."
        )

        if self.project_context:
            system += self.project_context
        else:
            from towel.agent.project import load_project_context

            project_block = load_project_context()
            if project_block:
                system += project_block

        if self.memory:
            memory_block = self.memory.to_prompt_block(query=query)
            if memory_block:
                system += memory_block

        tools = self.skills.tool_definitions()
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
                system += (
                    "\n\n# Tools\n\n"
                    "You may call one or more functions to assist with the user query.\n\n"
                    "Available tools:\n" + "\n".join(tool_lines) + "\n\n"
                    "For each function call, return a json object within "
                    "<tool_call></tool_call> tags:\n"
                    "<tool_call>\n"
                    '{"name": <function-name>, "arguments": <args-json-object>}\n'
                    "</tool_call>\n\n"
                    f"The ONLY supported tool names are: {', '.join(tool_names)}\n\n"
                    "Only call listed tools. Do NOT invent function names."
                )
            else:
                system += (
                    "\n\n# Tool-use rules\n\n"
                    "The available tools are provided by the chat template's Tools "
                    "section below. Use them when they help answer the user.\n\n"
                    "IMPORTANT:\n"
                    "- Only call tools from the provided list. Do NOT invent or guess "
                    "tool names.\n"
                    "- When using a tool, prefer emitting just the tool call instead of "
                    "narrating that you are about to check something.\n"
                    "- After tool results arrive, give the concrete answer or make one "
                    "corrected retry. Do not repeat vague status updates.\n"
                    "- If no tool is needed, respond directly without tool calls."
                )
        return system

    def _build_messages(self, conversation: Conversation) -> list[dict[str, str]]:
        """Convert conversation to Ollama chat messages format."""
        query = conversation.latest_user_query()
        from towel.agent.capture import run_capture_hooks
        run_capture_hooks(query, memory=self.memory, config=self.config, runtime=self)
        system_content = self._build_system_prompt(
            include_tools_section=not bool(self._native_tools_supported),
            query=query,
        )
        existing_messages = [
            {"role": msg.role.value, "content": msg.content} for msg in conversation.messages
        ]
        output_reserve = estimate_output_reserve(
            existing_messages,
            configured_max_tokens=self.config.model.max_tokens,
        )
        maybe_compact_conversation(
            conversation,
            system_content=system_content,
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
                messages.append(
                    {"role": "user", "content": f"<tool_result>\n{msg.content}\n</tool_result>"}
                )
        return messages

    def build_inference_request(self, conversation: Conversation) -> dict[str, Any]:
        """Build a worker-safe Ollama chat payload for this conversation."""
        use_native = bool(self._native_tools_supported)
        query = conversation.latest_user_query()
        from towel.agent.capture import run_capture_hooks
        run_capture_hooks(query, memory=self.memory, config=self.config, runtime=self)
        request: dict[str, Any] = {
            "mode": "ollama_chat",
            "system": self._build_system_prompt(
                include_tools_section=not use_native,
                query=query,
            ),
            "messages": self._build_messages(conversation),
            "model": self.config.model.name,
        }
        if use_native:
            native_tools = tools_as_openai_functions(self.skills.tool_definitions())
            if native_tools:
                request["tools"] = native_tools
        return request

    async def generate(self, conversation: Conversation) -> OllamaGenerationResult:
        if not self._loaded:
            await self.load_model()
        return await self.generate_from_request(self.build_inference_request(conversation))

    async def generate_from_request(self, request: dict[str, Any]) -> OllamaGenerationResult:
        if not self._loaded:
            await self.load_model()

        mode = request.get("mode")
        if mode not in ("ollama_chat", "anthropic_messages"):
            raise ValueError(f"Unsupported inference mode for Ollama runtime: {mode}")

        model = request.get("model") or self.config.model.name
        system = request.get("system", "")
        raw_messages = request.get("messages", [])
        sys_msg = [{"role": "system", "content": system}]
        messages = sys_msg + raw_messages if system else raw_messages

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.model.temperature,
                "top_p": self.config.model.top_p,
                "num_predict": self.config.model.max_tokens,
            },
        }
        if request.get("tools"):
            payload["tools"] = request["tools"]

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{self.ollama_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        message = data.get("message", {}) or {}
        return OllamaGenerationResult(
            text=message.get("content", "") or "",
            tool_calls=_normalize_ollama_tool_calls(message.get("tool_calls") or []),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )

    async def stream(self, conversation: Conversation) -> AsyncIterator[str]:
        if not self._loaded:
            await self.load_model()
        async for token in self.stream_from_request(self.build_inference_request(conversation)):
            yield token

    async def stream_from_request(self, request: dict[str, Any]) -> AsyncIterator[str]:
        if not self._loaded:
            await self.load_model()

        mode = request.get("mode")
        if mode not in ("ollama_chat", "anthropic_messages"):
            raise ValueError(f"Unsupported inference mode for Ollama runtime: {mode}")

        model = request.get("model") or self.config.model.name
        system = request.get("system", "")
        raw_messages = request.get("messages", [])
        sys_msg = [{"role": "system", "content": system}]
        messages = sys_msg + raw_messages if system else raw_messages

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.model.temperature,
                "top_p": self.config.model.top_p,
                "num_predict": self.config.model.max_tokens,
            },
        }
        if request.get("tools"):
            payload["tools"] = request["tools"]

        self._last_stream_tool_calls = []
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"{self.ollama_url}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if self._cancel_flag:
                        break
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk_message = chunk.get("message") or {}
                    token = chunk_message.get("content", "") or ""
                    if token:
                        yield token
                    chunk_tcs = chunk_message.get("tool_calls") or []
                    if chunk_tcs:
                        # Ollama emits tool_calls in the final chunk (done=true).
                        # Stash them so step_streaming can pick them up after the
                        # stream finishes — the AsyncIterator[str] contract can't
                        # carry structured payloads directly.
                        self._last_stream_tool_calls.extend(
                            _normalize_ollama_tool_calls(chunk_tcs)
                        )
                    if chunk.get("done"):
                        break

    async def step(self, conversation: Conversation) -> Message:
        for _iteration in range(MAX_TOOL_ITERATIONS):
            result = await self.generate(conversation)
            # Native tool channel returns structured tool_calls; only fall back
            # to text parsing if the response was plain text.
            if result.tool_calls:
                tool_calls = result.tool_calls
                remaining_text = result.text
            else:
                tool_calls, remaining_text = parse_tool_calls(result.text)

            if not tool_calls:
                return Message(
                    role=Role.ASSISTANT,
                    content=result.text,
                    metadata={"backend": "ollama", "model": self.config.model.name},
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
            metadata={"backend": "ollama", "max_iterations": True},
        )

    async def step_streaming(self, conversation: Conversation) -> AsyncIterator[AgentEvent]:
        self._cancel_flag = False

        for _iteration in range(MAX_TOOL_ITERATIONS):
            full_text = ""
            async for chunk in self.stream(conversation):
                full_text += chunk
                yield AgentEvent.token(chunk)

            if self._cancel_flag:
                if full_text.strip():
                    conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.cancelled(full_text, metadata={"reason": "user_cancelled"})
                self._cancel_flag = False
                return

            # Prefer structured tool_calls captured during streaming; otherwise
            # fall back to parsing tool-call markers out of the accumulated text.
            if self._last_stream_tool_calls:
                tool_calls = list(self._last_stream_tool_calls)
                self._last_stream_tool_calls = []
                remaining_text = full_text
            else:
                tool_calls, remaining_text = parse_tool_calls(full_text)

            if not tool_calls:
                conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.complete(
                    full_text,
                    metadata={"backend": "ollama", "model": self.config.model.name},
                )
                return

            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            for tc in tool_calls:
                if self._cancel_flag:
                    yield AgentEvent.cancelled(
                        remaining_text or "", metadata={"reason": "user_cancelled"}
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
            metadata={"backend": "ollama", "max_iterations": True},
        )

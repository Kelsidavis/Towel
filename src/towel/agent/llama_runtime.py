"""llama.cpp runtime — local inference via llama-server's OpenAI-compatible API.

Connects to llama-server (llama.cpp's HTTP server) which exposes
/v1/chat/completions in OpenAI format. Supports streaming via SSE.

Usage:
    # Start llama-server:
    llama-server -m model.gguf -ngl 99 --port 8080

    # Then run Towel:
    towel chat --backend llama
    towel chat --backend llama --llama-url http://gpu-box:8080
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
from towel.agent.tool_parser import parse_tool_calls
from towel.config import TowelConfig
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.agent.llama")

MAX_TOOL_ITERATIONS = 999
DEFAULT_LLAMA_URL = "http://localhost:8080"


@dataclass
class LlamaGenerationResult:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LlamaRuntime:
    """Agent runtime that calls llama-server via /v1/chat/completions.

    Uses the OpenAI-compatible chat completions endpoint provided by
    llama.cpp's llama-server. No Ollama dependency — connects directly
    to the llama-server process.

    Usage:
        towel chat --backend llama
        towel chat --backend llama --llama-url http://gpu-box:8080
        towel worker --master ws://... --backend llama --llama-url http://gpu-box:8080
    """

    def __init__(
        self,
        config: TowelConfig,
        skills: SkillRegistry | None = None,
        memory: Any | None = None,
        llama_url: str = DEFAULT_LLAMA_URL,
        llama_model: str | None = None,
        auto_start: bool = True,
    ) -> None:
        self.config = config
        self.skills = skills or SkillRegistry()
        self.memory = memory
        self.llama_url = llama_url.rstrip("/")
        self.llama_model = llama_model
        self.auto_start = auto_start
        self._loaded = False
        self._cancel_flag = False
        self._managed_server: Any | None = None

    def cancel(self) -> None:
        self._cancel_flag = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_flag

    async def load_model(self) -> None:
        """Connect to llama-server, auto-starting it if needed."""
        if self._loaded:
            return

        # Try connecting to an existing llama-server first
        server_up = await self._check_health()
        if server_up:
            log.info(f"llama-server runtime ready (url: {self.llama_url})")
            self._loaded = True
            return

        # No server running — try auto-start if enabled
        if not self.auto_start:
            raise RuntimeError(
                f"Cannot connect to llama-server at {self.llama_url}. "
                "Start it with: llama-server -m model.gguf -ngl 99 --port 8080"
            )

        from towel.agent.discovery import ManagedLlamaServer, detect_system

        caps = detect_system()
        if not caps.has_llama_server:
            raise RuntimeError(
                "No llama-server binary found. Install llama.cpp or add it to PATH."
            )

        # Resolve model path
        model_path = self.llama_model
        if not model_path:
            best = caps.best_model
            if not best:
                raise RuntimeError(
                    "No GGUF models found. Place .gguf files in ~/models/, "
                    "~/.towel/models/, or use --llama-model to specify one."
                )
            model_path = str(best.path)
            log.info(f"Auto-selected model: {best.name} ({best.size_gb} GB)")

        # Extract port from llama_url
        port = 8080
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.llama_url)
            if parsed.port:
                port = parsed.port
        except Exception:
            pass

        server = ManagedLlamaServer(
            binary_path=caps.llama_server_path,
            model_path=model_path,
            port=port,
        )
        server.start()
        await server.wait_healthy()
        self._managed_server = server
        self._loaded = True
        log.info(f"llama-server auto-started on port {port} with {model_path}")

    async def _check_health(self) -> bool:
        """Return True if llama-server is healthy at self.llama_url."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.llama_url}/health")
                resp.raise_for_status()
                health = resp.json()
                status = health.get("status", "unknown")
                if status != "ok":
                    log.warning(f"llama-server health status: {status}")
                    return False
                return True
        except (httpx.ConnectError, httpx.ReadError):
            return False

    def shutdown(self) -> None:
        """Stop the managed llama-server if we started one."""
        if self._managed_server:
            self._managed_server.stop()
            self._managed_server = None

    def _build_system_prompt(self) -> str:
        """Build system prompt with identity, context, and tool instructions."""
        system = self.config.identity + (
            "\n\nAfter using a tool, always answer the user's original question "
            "based on the tool result. Do not just acknowledge the tool output — "
            "use it to provide a direct, helpful answer."
        )

        from towel.agent.project import load_project_context

        project_block = load_project_context()
        if project_block:
            system += project_block

        if self.memory:
            memory_block = self.memory.to_prompt_block()
            if memory_block:
                system += memory_block

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
        return system

    def _build_messages(self, conversation: Conversation) -> list[dict[str, str]]:
        """Convert conversation to OpenAI chat messages format."""
        system_content = self._build_system_prompt()
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
        """Build a worker-safe payload for this conversation."""
        return {
            "mode": "llama_chat",
            "system": self._build_system_prompt(),
            "messages": self._build_messages(conversation),
            "model": self.config.model.name,
        }

    async def generate(self, conversation: Conversation) -> LlamaGenerationResult:
        if not self._loaded:
            await self.load_model()
        return await self.generate_from_request(self.build_inference_request(conversation))

    async def generate_from_request(self, request: dict[str, Any]) -> LlamaGenerationResult:
        if not self._loaded:
            await self.load_model()

        mode = request.get("mode")
        if mode not in ("llama_chat", "anthropic_messages"):
            raise ValueError(f"Unsupported inference mode for llama runtime: {mode}")

        system = request.get("system", "")
        raw_messages = request.get("messages", [])

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(raw_messages)

        payload = {
            "messages": messages,
            "stream": False,
            "temperature": request.get("temperature", self.config.model.temperature),
            "top_p": request.get("top_p", self.config.model.top_p),
            "max_tokens": request.get("max_tokens", self.config.model.max_tokens),
        }
        if "reasoning_effort" in request:
            payload["reasoning_effort"] = request["reasoning_effort"]

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.llama_url}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LlamaGenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
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
        if mode not in ("llama_chat", "anthropic_messages"):
            raise ValueError(f"Unsupported inference mode for llama runtime: {mode}")

        system = request.get("system", "")
        raw_messages = request.get("messages", [])

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(raw_messages)

        payload = {
            "messages": messages,
            "stream": True,
            "temperature": request.get("temperature", self.config.model.temperature),
            "top_p": request.get("top_p", self.config.model.top_p),
            "max_tokens": request.get("max_tokens", self.config.model.max_tokens),
        }
        if "reasoning_effort" in request:
            payload["reasoning_effort"] = request["reasoning_effort"]

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{self.llama_url}/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if self._cancel_flag:
                        break
                    if not line:
                        continue
                    # SSE format: "data: {...}" or "data: [DONE]"
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token

    async def step(self, conversation: Conversation) -> Message:
        for _iteration in range(MAX_TOOL_ITERATIONS):
            result = await self.generate(conversation)
            tool_calls, remaining_text = parse_tool_calls(result.text)

            if not tool_calls:
                return Message(
                    role=Role.ASSISTANT,
                    content=result.text,
                    metadata={"backend": "llama", "model": self.config.model.name},
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
            metadata={"backend": "llama", "max_iterations": True},
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

            tool_calls, remaining_text = parse_tool_calls(full_text)

            if not tool_calls:
                conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.complete(
                    full_text,
                    metadata={"backend": "llama", "model": self.config.model.name},
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
            metadata={"backend": "llama", "max_iterations": True},
        )

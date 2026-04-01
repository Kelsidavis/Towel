"""Core agent runtime — manages MLX model loading, inference, and tool dispatch."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from towel.agent.context import estimate_output_reserve, fit_messages, maybe_compact_conversation
from towel.agent.conversation import Conversation, Message, Role
from towel.agent.events import AgentEvent
from towel.agent.instance_lock import acquire_runtime_lock
from towel.agent.tool_parser import parse_tool_calls
from towel.config import TowelConfig
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.agent")

MAX_TOOL_ITERATIONS = 10


def mlx_tokenizer_config() -> dict[str, Any]:
    """Return tokenizer config overrides for MLX loads.

    `fix_mistral_regex=True` suppresses the known incorrect-regex warning for
    affected converted tokenizers and applies the corrected tokenizer behavior.
    Transformers ignores this for unaffected tokenizers.
    """
    return {"fix_mistral_regex": False}


_TOOL_ERROR_PATTERNS = (
    re.compile(r"^Error executing\b", re.IGNORECASE),
    re.compile(r"^Unknown tool:\b", re.IGNORECASE),
    re.compile(r"^File not found:\b", re.IGNORECASE),
    re.compile(r"^Not a directory:\b", re.IGNORECASE),
    re.compile(r"^Invalid index:\b", re.IGNORECASE),
    re.compile(r"^File too large\b", re.IGNORECASE),
)


def tool_result_is_error(result: str) -> bool:
    """Heuristic for whether a tool result represents failure."""
    return any(pattern.search(result) for pattern in _TOOL_ERROR_PATTERNS)


def format_tool_feedback(tool_name: str, result: str, is_error: bool) -> str:
    """Format tool feedback so the next model step can recover reliably."""
    status = "error" if is_error else "ok"
    next_step = (
        "Retry with one corrected valid tool call, or answer directly with the limitation."
        if is_error
        else "Use this result to answer the user concretely. Do not stop at saying you will check."
    )
    return (
        f"[{tool_name}]\n"
        f"status: {status}\n"
        f"result:\n{result}\n\n"
        f"next:\n{next_step}"
    )


@dataclass
class GenerationResult:
    """Result of a single generation step."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_per_second: float = 0.0
    total_tokens: int = 0


class AgentRuntime:
    """The brain. Loads an MLX model, runs inference, dispatches tools.

    This is the core loop:
      1. Receive a message
      2. Build prompt from conversation + system identity + memory + available tools
      3. Run MLX inference
      4. If tool calls → execute them → feed results back → goto 3
      5. Return final response
    """

    def __init__(
        self,
        config: TowelConfig,
        skills: SkillRegistry | None = None,
        memory: Any | None = None,
    ) -> None:
        self.config = config
        self.skills = skills or SkillRegistry()
        self.memory = memory  # MemoryStore instance
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._cancel: asyncio.Event = asyncio.Event()
        # Single-thread executor to serialize all MLX Metal operations.
        # Metal command buffers are not thread-safe — concurrent access
        # from the default thread pool crashes the GPU driver.
        self._mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    def cancel(self) -> None:
        """Signal the current generation to stop."""
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    async def load_model(self) -> None:
        """Load the MLX model and tokenizer."""
        if self._loaded:
            return

        acquire_runtime_lock()

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        self._model, self._tokenizer = await loop.run_in_executor(
            self._mlx_executor, self._load_model_sync
        )
        self._loaded = True

    def _load_model_sync(self) -> tuple[Any, Any]:
        """Synchronous model loading via mlx_lm."""
        from mlx_lm import load

        model, tokenizer = load(
            self.config.model.name,
            tokenizer_config=mlx_tokenizer_config(),
        )
        return model, tokenizer

    async def generate(self, conversation: Conversation) -> GenerationResult:
        """Run a single generation pass."""
        if not self._loaded:
            await self.load_model()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._mlx_executor, self._generate_sync, conversation)
        return result

    def build_inference_request(self, conversation: Conversation) -> dict[str, Any]:
        """Build a worker-safe inference payload for this conversation."""
        return {"mode": "mlx_prompt", "prompt": self._build_prompt(conversation)}

    async def generate_from_request(self, request: dict[str, Any]) -> GenerationResult:
        """Generate from a prebuilt prompt payload."""
        if not self._loaded:
            await self.load_model()

        if request.get("mode") != "mlx_prompt":
            raise ValueError(f"Unsupported inference mode: {request.get('mode')}")

        loop = asyncio.get_event_loop()
        prompt = request["prompt"]
        max_tokens = request.get("max_tokens")
        return await loop.run_in_executor(
            self._mlx_executor, self._generate_prompt_sync, prompt, max_tokens
        )

    def _make_turboquant_cache(self) -> list | None:
        """Build a TurboQuant prompt cache if enabled, else None."""
        if not self.config.model.turboquant:
            return None
        from towel.agent.turboquant import make_turboquant_cache

        return make_turboquant_cache(
            self._model,
            kv_bits=self.config.model.turboquant_bits,
            qjl_ratio=self.config.model.turboquant_qjl_ratio,
        )

    def _generate_sync(self, conversation: Conversation) -> GenerationResult:
        """Synchronous generation via mlx_lm."""
        prompt = self._build_prompt(conversation)
        return self._generate_prompt_sync(prompt)

    def _generate_prompt_sync(self, prompt: str, max_tokens: int | None = None) -> GenerationResult:
        """Synchronous generation via mlx_lm from a prebuilt prompt."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=self.config.model.temperature,
            top_p=self.config.model.top_p,
        )
        extra_kwargs: dict[str, Any] = {}
        tq_cache = self._make_turboquant_cache()
        if tq_cache is not None:
            extra_kwargs["prompt_cache"] = tq_cache

        start = time.perf_counter()
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self.config.model.max_tokens,
            sampler=sampler,
            **extra_kwargs,
        )
        elapsed = time.perf_counter() - start

        # Rough token count from response length
        token_count = len(self._tokenizer.encode(response))
        tps = token_count / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            text=response,
            tokens_per_second=tps,
            total_tokens=token_count,
        )

    async def stream(self, conversation: Conversation) -> AsyncIterator[str]:
        """Stream generation token by token. Respects cancel signal."""
        if not self._loaded:
            await self.load_model()

        async for chunk in self.stream_from_request(self.build_inference_request(conversation)):
            yield chunk

    async def stream_from_request(self, request: dict[str, Any]) -> AsyncIterator[str]:
        """Stream generation from a prebuilt prompt payload."""
        if not self._loaded:
            await self.load_model()

        if request.get("mode") != "mlx_prompt":
            raise ValueError(f"Unsupported inference mode: {request.get('mode')}")

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancel_flag = self._cancel

        def _stream_sync() -> None:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            prompt = request["prompt"]
            sampler = make_sampler(
                temp=self.config.model.temperature,
                top_p=self.config.model.top_p,
            )
            extra_kwargs: dict[str, Any] = {}
            tq_cache = self._make_turboquant_cache()
            if tq_cache is not None:
                extra_kwargs["prompt_cache"] = tq_cache

            for chunk in stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.config.model.max_tokens,
                sampler=sampler,
                **extra_kwargs,
            ):
                if cancel_flag.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(self._mlx_executor, _stream_sync)

        while True:
            if cancel_flag.is_set():
                # Drain any remaining chunks
                while not queue.empty():
                    queue.get_nowait()
                break
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    async def step(self, conversation: Conversation) -> Message:
        """Run one full agent step: generate → maybe call tools → return response.

        If the model emits tool calls, we execute them, inject results as
        TOOL messages, and re-generate — up to MAX_TOOL_ITERATIONS times.
        """
        total_tokens = 0
        last_tps = 0.0

        for iteration in range(MAX_TOOL_ITERATIONS):
            result = await self.generate(conversation)
            total_tokens += result.total_tokens
            last_tps = result.tokens_per_second

            tool_calls, remaining_text = parse_tool_calls(result.text)

            if not tool_calls:
                # No tool calls — return the final text response
                return Message(
                    role=Role.ASSISTANT,
                    content=result.text,
                    metadata={"tps": last_tps, "tokens": total_tokens},
                )

            # Add the assistant's message (with tool calls stripped) to conversation
            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            # Execute each tool call and add results
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

        # Hit max iterations — return what we have
        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        return Message(
            role=Role.ASSISTANT,
            content=remaining_text or "I've reached my tool execution limit for this turn.",
            metadata={"tps": last_tps, "tokens": total_tokens, "max_iterations": True},
        )

    async def step_streaming(self, conversation: Conversation) -> AsyncIterator[AgentEvent]:
        """Run a full agent step, yielding events as they happen.

        Streams tokens during generation, emits tool call/result events,
        and ends with a response_complete or cancelled event.
        """
        # Reset cancel flag for this generation
        self._cancel.clear()
        total_tokens = 0

        for iteration in range(MAX_TOOL_ITERATIONS):
            # Stream tokens and accumulate the full response
            full_text = ""
            start = time.perf_counter()
            async for chunk in self.stream(conversation):
                full_text += chunk
                yield AgentEvent.token(chunk)
            elapsed = time.perf_counter() - start

            # Check if generation was cancelled
            if self._cancel.is_set():
                if full_text.strip():
                    conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.cancelled(
                    full_text,
                    metadata={"tokens": total_tokens, "reason": "user_cancelled"},
                )
                self._cancel.clear()
                return

            # Estimate token count from accumulated text
            if self._tokenizer:
                token_count = len(self._tokenizer.encode(full_text))
            else:
                token_count = len(full_text.split())
            total_tokens += token_count
            tps = token_count / elapsed if elapsed > 0 else 0.0

            # Check for tool calls
            tool_calls, remaining_text = parse_tool_calls(full_text)

            if not tool_calls:
                conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.complete(
                    full_text,
                    metadata={"tps": tps, "tokens": total_tokens},
                )
                return

            # Tool call loop
            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            for tc in tool_calls:
                if self._cancel.is_set():
                    yield AgentEvent.cancelled(
                        remaining_text or "",
                        metadata={"tokens": total_tokens, "reason": "user_cancelled"},
                    )
                    self._cancel.clear()
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

        # Hit max iterations
        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        yield AgentEvent.complete(
            remaining_text or "I've reached my tool execution limit for this turn.",
            metadata={"tps": 0, "tokens": total_tokens, "max_iterations": True},
        )

    def _build_system_content(self) -> str:
        """Build the system prompt including project context, memory, and tool definitions."""
        system = self.config.identity + (
            "\n\nAfter using a tool, always answer the user's original question "
            "based on the tool result. Do not just acknowledge the tool output — "
            "use it to provide a direct, helpful answer. If you changed something "
            "or verified something, explicitly report that back to the user."
        )

        # Inject project context from .towel.md files
        from towel.agent.project import load_project_context

        project_block = load_project_context()
        if project_block:
            system += project_block

        # Inject persistent memories
        if self.memory:
            memory_block = self.memory.to_prompt_block()
            if memory_block:
                system += memory_block
        tools = self.skills.tool_definitions()
        if tools:
            # Compact format: name + description only. Full parameter schemas
            # bloat the prompt (~330 tools) and slow inference significantly.
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

    def _token_count(self, text: str) -> int:
        """Count tokens using the loaded tokenizer, or estimate."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // 4)

    def _build_prompt(self, conversation: Conversation) -> str:
        """Build a chat prompt string from the conversation history.

        Uses the context window manager to fit messages within the
        model's token budget, dropping oldest messages first.
        """
        system_content = self._build_system_content()
        all_messages = conversation.to_chat_messages()
        output_reserve = estimate_output_reserve(
            all_messages,
            configured_max_tokens=self.config.model.max_tokens,
            token_counter=self._token_count,
        )
        maybe_compact_conversation(
            conversation,
            system_content=system_content,
            context_window=self.config.model.context_window,
            max_output_tokens=output_reserve,
            token_counter=self._token_count,
        )
        all_messages = conversation.to_chat_messages()

        # Collect indices of pinned messages so they survive context eviction
        pinned_indices = {i for i, msg in enumerate(conversation.messages) if msg.pinned}

        fitted_messages, budget = fit_messages(
            system_content=system_content,
            messages=all_messages,
            context_window=self.config.model.context_window,
            max_output_tokens=output_reserve,
            token_counter=self._token_count,
            pinned_indices=pinned_indices if pinned_indices else None,
        )

        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "system", "content": system_content}]
            # Convert tool messages to the format the chat template expects.
            # Qwen3 expects tool role messages with content in the standard
            # chat template — the tokenizer handles wrapping them.
            for msg in fitted_messages:
                if msg["role"] == "tool":
                    # Strip the [tool_name] prefix and pass as tool role
                    content = msg["content"]
                    # Extract tool name from "[tool_name] ..." format
                    if content.startswith("[") and "]" in content:
                        _name, _, result = content.partition("]")
                        tool_name = _name.lstrip("[")
                        messages.append(
                            {
                                "role": "tool",
                                "content": result.lstrip(),
                                "name": tool_name,
                            }
                        )
                    else:
                        messages.append(msg)
                else:
                    messages.append(msg)
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback if template doesn't support tool role
                fallback_messages = [{"role": "system", "content": system_content}]
                for msg in fitted_messages:
                    if msg["role"] == "tool":
                        fallback_messages.append(
                            {
                                "role": "user",
                                "content": f"<tool_response>\n{msg['content']}\n</tool_response>",
                            }
                        )
                    else:
                        fallback_messages.append(msg)
                return self._tokenizer.apply_chat_template(
                    fallback_messages, tokenize=False, add_generation_prompt=True
                )

        # Fallback: simple concatenation
        parts = [f"System: {system_content}\n"]
        for msg in fitted_messages:
            parts.append(f"{msg['role'].capitalize()}: {msg['content']}\n")
        parts.append("Assistant: ")
        return "".join(parts)

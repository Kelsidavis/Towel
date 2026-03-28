"""Core agent runtime — manages MLX model loading, inference, and tool dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from towel.config import TowelConfig
from towel.agent.context import fit_messages, ContextBudget
from towel.agent.conversation import Conversation, Message, Role
from towel.agent.events import AgentEvent
from towel.agent.tool_parser import parse_tool_calls
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.agent")

MAX_TOOL_ITERATIONS = 10


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

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        self._model, self._tokenizer = await loop.run_in_executor(
            None, self._load_model_sync
        )
        self._loaded = True

    def _load_model_sync(self) -> tuple[Any, Any]:
        """Synchronous model loading via mlx_lm."""
        from mlx_lm import load

        model, tokenizer = load(self.config.model.name)
        return model, tokenizer

    async def generate(self, conversation: Conversation) -> GenerationResult:
        """Run a single generation pass."""
        if not self._loaded:
            await self.load_model()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._generate_sync, conversation
        )
        return result

    def _generate_sync(self, conversation: Conversation) -> GenerationResult:
        """Synchronous generation via mlx_lm."""
        from mlx_lm import generate

        prompt = self._build_prompt(conversation)
        start = time.perf_counter()
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.config.model.max_tokens,
            temp=self.config.model.temperature,
            top_p=self.config.model.top_p,
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

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancel_flag = self._cancel

        def _stream_sync() -> None:
            from mlx_lm import stream_generate

            prompt = self._build_prompt(conversation)
            for chunk in stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.config.model.max_tokens,
                temp=self.config.model.temperature,
                top_p=self.config.model.top_p,
            ):
                if cancel_flag.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(None, _stream_sync)

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
                    result_str = str(tool_result) if not isinstance(tool_result, str) else tool_result
                except Exception as e:
                    result_str = f"Error executing {tc.name}: {e}"
                    log.error(result_str)

                conversation.add(
                    Role.TOOL,
                    f"[{tc.name}] {result_str}",
                    tool_name=tc.name,
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
                    result_str = str(tool_result) if not isinstance(tool_result, str) else tool_result
                except Exception as e:
                    result_str = f"Error executing {tc.name}: {e}"
                    log.error(result_str)

                yield AgentEvent.tool_result(tc.name, result_str)
                conversation.add(Role.TOOL, f"[{tc.name}] {result_str}", tool_name=tc.name)

        # Hit max iterations
        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        yield AgentEvent.complete(
            remaining_text or "I've reached my tool execution limit for this turn.",
            metadata={"tps": 0, "tokens": total_tokens, "max_iterations": True},
        )

    def _build_system_content(self) -> str:
        """Build the system prompt including project context, memory, and tool definitions."""
        system = self.config.identity

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
            tool_lines = []
            for t in tools:
                params = t.get("parameters", {})
                param_desc = ", ".join(
                    f'{k}: {v.get("type", "any")}'
                    for k, v in params.get("properties", {}).items()
                ) if params else ""
                tool_lines.append(f"- {t['name']}({param_desc}): {t['description']}")

            system += (
                "\n\nYou have access to the following tools. To use a tool, "
                "emit a JSON block like this:\n"
                '```json\n{"tool": "tool_name", "arguments": {"arg": "value"}}\n```\n\n'
                "Available tools:\n" + "\n".join(tool_lines)
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

        fitted_messages, budget = fit_messages(
            system_content=system_content,
            messages=all_messages,
            context_window=self.config.model.context_window,
            max_output_tokens=self.config.model.max_tokens,
            token_counter=self._token_count,
        )

        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "system", "content": system_content}]
            messages.extend(fitted_messages)
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Fallback: simple concatenation
        parts = [f"System: {system_content}\n"]
        for msg in fitted_messages:
            parts.append(f"{msg['role'].capitalize()}: {msg['content']}\n")
        parts.append("Assistant: ")
        return "".join(parts)

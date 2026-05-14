"""Shared per-turn memory-capture hooks for every agent runtime.

Each runtime (MLX, llama-server, Ollama, Claude) calls
:func:`run_capture_hooks` with the current user-turn text right before
it builds the system prompt. Centralizing this lets all four
runtimes share the same regex + optional-LLM extract behavior
without copying code between them — important because the four
runtimes diverge in tooling but should agree on what memory
captures look like.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Protocol

log = logging.getLogger("towel.agent.capture")


class _StepLike(Protocol):
    async def step(self, conversation: Any) -> Any: ...


def run_capture_hooks(
    query: str | None,
    *,
    memory: Any,
    config: Any,
    runtime: _StepLike,
) -> None:
    """Run regex auto-capture, then schedule LLM extract on miss.

    ``runtime`` is the calling AgentRuntime / LlamaRuntime / etc.,
    used only for its ``step`` method when the LLM-extract path
    fires. Everything else (memory store, config flags) is read off
    the explicit arguments so this stays trivially mockable in
    unit tests.
    """
    if not query or memory is None:
        return
    regex_captures: list = []
    if getattr(config, "auto_capture", True):
        try:
            from towel.memory.auto_capture import apply as _ac_apply
            regex_captures = _ac_apply(query, memory)
        except Exception as exc:
            log.debug("auto-capture failed: %s", exc)
    if regex_captures or not getattr(config, "auto_llm_extract", False):
        return
    # Background LLM extraction — fire-and-forget. The same backend
    # serializes the work behind the actual turn generation, so it
    # runs when the model is idle.
    try:
        from towel.agent.conversation import Conversation, Role
        from towel.memory.llm_extract import schedule_background_extraction

        async def _step(prompt: str) -> str:
            conv = Conversation()
            conv.add(Role.USER, prompt)
            msg = await runtime.step(conv)
            return getattr(msg, "content", "") or ""

        schedule_background_extraction(
            query, _step, memory,
            scope=getattr(memory, "default_scope", "") or None,
        )
    except Exception as exc:
        log.debug("auto-llm-extract scheduling failed: %s", exc)

"""Agent hooks — let plugins react to lifecycle events.

Hooks fire at key moments in the agent lifecycle:
  on_message_received  — user sends a message
  on_before_generate   — about to run inference
  on_after_generate    — inference complete
  on_tool_call         — tool is being called
  on_tool_result       — tool returned a result
  on_error             — an error occurred
  on_conversation_save — conversation persisted

Skills or middleware can register hooks to log, modify, or
trigger side effects at any of these points.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Awaitable

log = logging.getLogger("towel.agent.hooks")

HookFn = Callable[..., Awaitable[None]]


class HookRegistry:
    """Registry for agent lifecycle hooks."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[tuple[str, HookFn]]] = defaultdict(list)

    def on(self, event: str, name: str, fn: HookFn) -> None:
        """Register a hook for an event."""
        self._hooks[event].append((name, fn))
        log.debug(f"Hook registered: {name} -> {event}")

    def off(self, event: str, name: str) -> None:
        """Remove a named hook from an event."""
        self._hooks[event] = [(n, f) for n, f in self._hooks[event] if n != name]

    async def emit(self, event: str, **kwargs: Any) -> None:
        """Fire all hooks for an event."""
        for name, fn in self._hooks.get(event, []):
            try:
                await fn(**kwargs)
            except Exception as e:
                log.warning(f"Hook {name} error on {event}: {e}")

    def list_hooks(self) -> dict[str, list[str]]:
        """Return {event: [hook_names]}."""
        return {event: [n for n, _ in hooks] for event, hooks in self._hooks.items() if hooks}

    @property
    def count(self) -> int:
        return sum(len(hooks) for hooks in self._hooks.values())


# Singleton
hooks = HookRegistry()


# ── Built-in hook helpers ──

async def log_hook(**kwargs: Any) -> None:
    """Simple logging hook — logs all events."""
    event = kwargs.pop("_event", "unknown")
    details = ", ".join(f"{k}={str(v)[:50]}" for k, v in kwargs.items())
    log.info(f"[hook] {event}: {details}")


def register_builtin_hooks(registry: HookRegistry) -> None:
    """Register default hooks."""
    # Logging is off by default — enable with /hooks enable logging
    pass

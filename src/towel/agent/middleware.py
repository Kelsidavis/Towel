"""Agent middleware — hooks that run before/after generation.

Middleware can inspect, modify, or block messages flowing through
the agent. Use cases: content filtering, rate limiting, logging,
auto-tagging, cost tracking.

Each middleware is a callable:
  async def my_middleware(ctx: MiddlewareContext) -> MiddlewareContext

Chain them with MiddlewareStack.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("towel.agent.middleware")


@dataclass
class MiddlewareContext:
    """Data flowing through the middleware chain."""

    user_message: str
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False
    block_reason: str = ""

    # Timing
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0


MiddlewareFn = Callable[[MiddlewareContext], Awaitable[MiddlewareContext]]


class MiddlewareStack:
    """Ordered chain of middleware functions."""

    def __init__(self) -> None:
        self._pre: list[tuple[str, MiddlewareFn]] = []
        self._post: list[tuple[str, MiddlewareFn]] = []

    def add_pre(self, name: str, fn: MiddlewareFn) -> None:
        """Add middleware that runs BEFORE generation."""
        self._pre.append((name, fn))
        log.info(f"Pre-middleware registered: {name}")

    def add_post(self, name: str, fn: MiddlewareFn) -> None:
        """Add middleware that runs AFTER generation."""
        self._post.append((name, fn))
        log.info(f"Post-middleware registered: {name}")

    async def run_pre(self, ctx: MiddlewareContext) -> MiddlewareContext:
        for name, fn in self._pre:
            try:
                ctx = await fn(ctx)
                if ctx.blocked:
                    log.info(f"Blocked by {name}: {ctx.block_reason}")
                    return ctx
            except Exception as e:
                log.warning(f"Pre-middleware {name} error: {e}")
        return ctx

    async def run_post(self, ctx: MiddlewareContext) -> MiddlewareContext:
        for name, fn in self._post:
            try:
                ctx = await fn(ctx)
            except Exception as e:
                log.warning(f"Post-middleware {name} error: {e}")
        return ctx

    def list_middleware(self) -> dict[str, list[str]]:
        return {
            "pre": [n for n, _ in self._pre],
            "post": [n for n, _ in self._post],
        }


# ── Built-in middleware ──


async def rate_limiter(ctx: MiddlewareContext) -> MiddlewareContext:
    """Block if too many requests in a short window."""
    _rate_limiter_state.setdefault("times", [])
    now = time.time()
    window = 60  # 1 minute
    max_requests = 30

    times = _rate_limiter_state["times"]
    times[:] = [t for t in times if now - t < window]

    if len(times) >= max_requests:
        ctx.blocked = True
        ctx.block_reason = f"Rate limit: {max_requests} requests per minute"
        return ctx

    times.append(now)
    return ctx


_rate_limiter_state: dict[str, Any] = {}


async def content_logger(ctx: MiddlewareContext) -> MiddlewareContext:
    """Log request/response for debugging."""
    if ctx.response:
        # Post-generation
        log.info(
            f"Response: {len(ctx.response)} chars, "
            f"{ctx.elapsed:.2f}s, "
            f"tokens={ctx.metadata.get('tokens', '?')}"
        )
    else:
        # Pre-generation
        log.info(f"Request: {len(ctx.user_message)} chars")
    return ctx


async def cost_tracker(ctx: MiddlewareContext) -> MiddlewareContext:
    """Track cumulative token usage for cost estimation."""
    if ctx.response:
        tokens = ctx.metadata.get("tokens", 0)
        _cost_state["total_tokens"] = _cost_state.get("total_tokens", 0) + tokens
        _cost_state["total_requests"] = _cost_state.get("total_requests", 0) + 1
        ctx.metadata["cumulative_tokens"] = _cost_state["total_tokens"]
        ctx.metadata["cumulative_requests"] = _cost_state["total_requests"]
    return ctx


_cost_state: dict[str, int] = {}

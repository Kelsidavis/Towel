"""Tests for agent middleware system."""

import pytest
from towel.agent.middleware import (
    MiddlewareContext, MiddlewareStack,
    rate_limiter, content_logger, cost_tracker,
    _rate_limiter_state, _cost_state,
)


class TestMiddlewareContext:
    def test_defaults(self):
        ctx = MiddlewareContext(user_message="hello")
        assert ctx.user_message == "hello"
        assert ctx.response == ""
        assert not ctx.blocked

    def test_elapsed(self):
        ctx = MiddlewareContext(user_message="x", started_at=100.0, completed_at=102.5)
        assert ctx.elapsed == 2.5


class TestMiddlewareStack:
    @pytest.mark.asyncio
    async def test_pre_runs_in_order(self):
        order = []
        async def a(ctx):
            order.append("a"); return ctx
        async def b(ctx):
            order.append("b"); return ctx

        stack = MiddlewareStack()
        stack.add_pre("a", a)
        stack.add_pre("b", b)

        ctx = MiddlewareContext(user_message="test")
        await stack.run_pre(ctx)
        assert order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_pre_blocking(self):
        async def blocker(ctx):
            ctx.blocked = True
            ctx.block_reason = "nope"
            return ctx
        async def should_not_run(ctx):
            raise RuntimeError("should not reach")

        stack = MiddlewareStack()
        stack.add_pre("blocker", blocker)
        stack.add_pre("unreachable", should_not_run)

        ctx = MiddlewareContext(user_message="test")
        result = await stack.run_pre(ctx)
        assert result.blocked
        assert result.block_reason == "nope"

    @pytest.mark.asyncio
    async def test_post_runs(self):
        ran = []
        async def logger(ctx):
            ran.append(True); return ctx

        stack = MiddlewareStack()
        stack.add_post("logger", logger)

        ctx = MiddlewareContext(user_message="x", response="y")
        await stack.run_post(ctx)
        assert len(ran) == 1

    def test_list_middleware(self):
        stack = MiddlewareStack()
        async def noop(ctx): return ctx
        stack.add_pre("a", noop)
        stack.add_post("b", noop)
        listing = stack.list_middleware()
        assert listing == {"pre": ["a"], "post": ["b"]}


class TestBuiltinMiddleware:
    @pytest.mark.asyncio
    async def test_rate_limiter_passes(self):
        _rate_limiter_state.clear()
        ctx = MiddlewareContext(user_message="hi")
        result = await rate_limiter(ctx)
        assert not result.blocked

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks(self):
        import time
        _rate_limiter_state.clear()
        _rate_limiter_state["times"] = [time.time()] * 30
        ctx = MiddlewareContext(user_message="hi")
        result = await rate_limiter(ctx)
        assert result.blocked
        assert "Rate limit" in result.block_reason

    @pytest.mark.asyncio
    async def test_cost_tracker(self):
        _cost_state.clear()
        ctx = MiddlewareContext(user_message="q", response="a", metadata={"tokens": 100})
        result = await cost_tracker(ctx)
        assert result.metadata["cumulative_tokens"] == 100

        ctx2 = MiddlewareContext(user_message="q", response="a", metadata={"tokens": 50})
        result2 = await cost_tracker(ctx2)
        assert result2.metadata["cumulative_tokens"] == 150

    @pytest.mark.asyncio
    async def test_content_logger(self):
        # Just verify it doesn't crash
        ctx = MiddlewareContext(user_message="hello")
        await content_logger(ctx)
        ctx.response = "world"
        await content_logger(ctx)

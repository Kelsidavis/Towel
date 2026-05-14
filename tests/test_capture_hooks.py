"""Integration tests for the shared per-turn capture hooks.

Covers the path through run_capture_hooks that previously only had
unit tests on the individual helpers (apply / schedule_background_
extraction). The runtime hook itself is exercised end-to-end with
a stub runtime so the orchestration logic (regex first, then LLM
fallback gated by config, then no-op on success) is covered.
"""

from __future__ import annotations

import asyncio

import pytest

from towel.agent.capture import run_capture_hooks
from towel.agent.conversation import Conversation, Role
from towel.config import TowelConfig
from towel.memory.llm_extract import _inflight
from towel.memory.store import MemoryStore


class _StubRuntime:
    """Minimal runtime that returns a canned LLM-extract response."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.step_calls: list[str] = []

    async def step(self, conversation):
        # Capture the prompt the LLM-extract helper sent us so the
        # test can assert the full path executed.
        msg = conversation.messages[-1]
        self.step_calls.append(msg.content)

        class _Reply:
            def __init__(self, c: str) -> None:
                self.content = c

        return _Reply(self._response)


@pytest.fixture(autouse=True)
def reset_inflight():
    _inflight.clear()
    yield


@pytest.fixture
def store(tmp_path):
    return MemoryStore(store_dir=tmp_path)


class TestRegexPath:
    def test_regex_match_captures_and_skips_llm(self, store):
        # "I'm a backend engineer" hits the role regex, so the LLM
        # extract path must NOT fire even when auto_llm_extract is on.
        config = TowelConfig(auto_capture=True, auto_llm_extract=True)
        runtime = _StubRuntime(response="[]")

        async def main() -> None:
            run_capture_hooks(
                "I'm a backend engineer.",
                memory=store, config=config, runtime=runtime,
            )
            # Give any (incorrectly) scheduled task a chance to run.
            await asyncio.sleep(0.05)

        asyncio.run(main())
        assert store.recall("role") is not None
        assert store.recall("role").content == "backend engineer"
        # LLM extract NOT called.
        assert runtime.step_calls == []


class TestLLMExtractPath:
    def test_regex_miss_with_flag_on_fires_llm(self, store):
        # Pick text the regex set won't match: no first-person cues,
        # no explicit remember, no preference phrasing. The LLM
        # extract path should kick in.
        config = TowelConfig(auto_capture=True, auto_llm_extract=True)
        canned = '[{"key": "stack", "content": "rust + tokio", "type": "fact"}]'
        runtime = _StubRuntime(response=canned)

        async def main() -> None:
            run_capture_hooks(
                "the deploy looked clean and the pipeline went green",
                memory=store, config=config, runtime=runtime,
            )
            # Let the background task complete.
            await asyncio.sleep(0.1)

        asyncio.run(main())
        assert runtime.step_calls, "stub runtime.step was not awaited"
        # The capture landed under the auto-source tag.
        e = store.recall("stack")
        assert e is not None
        assert e.source == "llm_extract:auto"
        assert e.content == "rust + tokio"

    def test_regex_miss_without_flag_does_nothing(self, store):
        config = TowelConfig(auto_capture=True, auto_llm_extract=False)
        runtime = _StubRuntime(response='[{"key": "x", "content": "y", "type": "fact"}]')

        async def main() -> None:
            run_capture_hooks(
                "the deploy looked clean and the pipeline went green",
                memory=store, config=config, runtime=runtime,
            )
            await asyncio.sleep(0.05)

        asyncio.run(main())
        assert runtime.step_calls == []
        assert store.count == 0


class TestGracefulFailure:
    def test_no_memory_is_noop(self):
        config = TowelConfig(auto_capture=True, auto_llm_extract=True)
        runtime = _StubRuntime(response="[]")
        # memory=None mustn't crash.
        run_capture_hooks(
            "I'm a senior engineer", memory=None, config=config, runtime=runtime,
        )
        assert runtime.step_calls == []

    def test_empty_query_is_noop(self, store):
        config = TowelConfig(auto_capture=True, auto_llm_extract=True)
        runtime = _StubRuntime(response="[]")
        run_capture_hooks(
            "", memory=store, config=config, runtime=runtime,
        )
        assert store.count == 0


class TestRuntimeWiringRemainsConnected:
    """Smoke-test that the four runtimes still call the shared helper
    rather than diverging back to inline implementations."""

    def test_runtime_calls_helper(self, store, monkeypatch):
        # Patch run_capture_hooks to a sentinel and verify AgentRuntime
        # routes through it. We don't run inference — we just call the
        # extracted method directly.
        from towel.agent.runtime import AgentRuntime

        seen: list[str] = []

        def fake_hooks(query, *, memory, config, runtime):
            seen.append(query)

        monkeypatch.setattr("towel.agent.capture.run_capture_hooks", fake_hooks)
        config = TowelConfig(identity="x")
        rt = AgentRuntime(config, memory=store)
        rt._run_capture_hooks("hello world")
        assert seen == ["hello world"]

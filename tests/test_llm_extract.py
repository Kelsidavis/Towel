"""Tests for the LLM-based memory extractor.

These cover the prompt → response parsing layer; the actual model
invocation is mocked since CI doesn't load a backend. Real-model
quality is observed via `towel memory extract` on the operator's
own backend.
"""

from __future__ import annotations

import asyncio

import pytest

from towel.memory.llm_extract import (
    LLMCapture,
    extract_via_llm,
    parse_response,
)

# ── parser ────────────────────────────────────────────────────────────


class TestParseResponse:
    def test_plain_json_array(self):
        raw = '[{"key": "role", "content": "engineer", "type": "user"}]'
        out = parse_response(raw)
        assert out == [LLMCapture("role", "engineer", "user")]

    def test_empty_array(self):
        assert parse_response("[]") == []

    def test_strips_markdown_fences(self):
        raw = '```json\n[{"key": "k", "content": "v", "type": "fact"}]\n```'
        out = parse_response(raw)
        assert out == [LLMCapture("k", "v", "fact")]

    def test_tolerates_preamble(self):
        # The model often prefixes with a sentence despite the prompt.
        raw = 'Here are the extracted facts:\n[{"key": "k", "content": "v", "type": "fact"}]'
        out = parse_response(raw)
        assert len(out) == 1
        assert out[0].key == "k"

    def test_invalid_type_defaults_to_fact(self):
        raw = '[{"key": "k", "content": "v", "type": "bogus"}]'
        out = parse_response(raw)
        assert out[0].memory_type == "fact"

    def test_drops_items_missing_required_fields(self):
        raw = '[{"key": "ok", "content": "v", "type": "fact"}, {"key": "bad"}, {"content": "x"}]'
        out = parse_response(raw)
        assert [c.key for c in out] == ["ok"]

    def test_drops_empty_string_fields(self):
        raw = '[{"key": "", "content": "v", "type": "fact"}, {"key": "k", "content": "", "type": "fact"}]'
        assert parse_response(raw) == []

    def test_malformed_json_returns_empty(self):
        assert parse_response("not json at all") == []
        assert parse_response("[broken") == []

    def test_empty_input_returns_empty(self):
        assert parse_response("") == []
        assert parse_response("   ") == []

    def test_non_list_top_level_returns_empty(self):
        # An object instead of an array — drop it.
        assert parse_response('{"key": "k", "content": "v", "type": "fact"}') == []


# ── extract_via_llm with mocked step ──────────────────────────────────


class TestExtractViaLLM:
    def test_passes_text_into_prompt(self):
        captured: list[str] = []

        async def step(prompt: str) -> str:
            captured.append(prompt)
            return "[]"

        asyncio.run(extract_via_llm("I am a senior engineer.", step))
        assert captured
        assert "I am a senior engineer." in captured[0]

    def test_returns_parsed_captures(self):
        async def step(prompt: str) -> str:
            return '[{"key": "role", "content": "engineer", "type": "user"}]'

        out = asyncio.run(extract_via_llm("some text", step))
        assert out == [LLMCapture("role", "engineer", "user")]

    def test_empty_text_skips_call(self):
        called = False

        async def step(prompt: str) -> str:
            nonlocal called
            called = True
            return "[]"

        out = asyncio.run(extract_via_llm("", step))
        assert out == []
        assert not called

    def test_step_exception_returns_empty(self):
        async def step(prompt: str) -> str:
            raise RuntimeError("backend down")

        out = asyncio.run(extract_via_llm("text", step))
        assert out == []


class TestScheduleBackgroundExtraction:
    """The fire-and-forget path. Tests run the loop briefly to let
    the scheduled task complete, then inspect the store."""

    def test_schedules_and_writes_captures(self, tmp_path):
        from towel.memory.llm_extract import _inflight, schedule_background_extraction
        from towel.memory.store import MemoryStore

        _inflight.clear()  # isolate from other tests
        store = MemoryStore(store_dir=tmp_path)

        async def step(prompt: str) -> str:
            return '[{"key": "role", "content": "engineer", "type": "user"}]'

        async def run() -> None:
            scheduled = schedule_background_extraction(
                "I am a senior engineer", step, store,
            )
            assert scheduled is True
            # Yield to let the scheduled task run to completion.
            await asyncio.sleep(0.05)

        asyncio.run(run())
        assert store.recall("role") is not None
        assert store.recall("role").source == "llm_extract:auto"

    def test_deduplicates_inflight_text(self, tmp_path):
        from towel.memory.llm_extract import _inflight, schedule_background_extraction
        from towel.memory.store import MemoryStore

        _inflight.clear()
        store = MemoryStore(store_dir=tmp_path)
        calls = []

        async def step(prompt: str) -> str:
            calls.append(prompt)
            await asyncio.sleep(0.01)
            return "[]"

        async def run() -> None:
            a = schedule_background_extraction("identical text", step, store)
            b = schedule_background_extraction("identical text", step, store)
            assert a is True
            # Second call dedup-rejects while the first is in flight.
            assert b is False
            await asyncio.sleep(0.05)

        asyncio.run(run())
        # Only one inference call despite two schedule attempts.
        assert len(calls) == 1

    def test_no_loop_returns_false(self, tmp_path):
        # Calling outside an asyncio loop must just refuse to
        # schedule rather than crash the synchronous caller.
        from towel.memory.llm_extract import _inflight, schedule_background_extraction
        from towel.memory.store import MemoryStore

        _inflight.clear()
        store = MemoryStore(store_dir=tmp_path)

        async def step(prompt: str) -> str:
            return "[]"

        # No asyncio.run wrapping → no running loop.
        result = schedule_background_extraction("x", step, store)
        assert result is False

    def test_empty_text_skips_schedule(self, tmp_path):
        from towel.memory.llm_extract import _inflight, schedule_background_extraction
        from towel.memory.store import MemoryStore

        _inflight.clear()
        store = MemoryStore(store_dir=tmp_path)

        async def step(prompt: str) -> str:
            return "[]"

        async def run() -> bool:
            return schedule_background_extraction("   ", step, store)

        assert asyncio.run(run()) is False

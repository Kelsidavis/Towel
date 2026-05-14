"""Tests for the message classification heuristics.

Lives at the boundary between intent ("chat" / "tool" / "task") and
TaskType (the finer-grained label that picks worker requirements).
Both are cheap regex paths — the LLM fallback is exercised only when
both return None / a non-matching value.

This file specifically guards the trivial-question short-circuit
added after benchmarking showed `what's 2+2?` routing to a 27B model
for 22s instead of the 2B model that could answer in 1s.
"""

from __future__ import annotations

from towel.nodes.roles import (
    TaskType,
    classify_message_intent,
    classify_task_type,
)


class TestGreetings:
    """The pre-existing greeting heuristic must still work."""

    def test_simple_greetings_are_chat(self):
        for msg in ["hi", "hello!", "thanks", "yo there", "ok cool"]:
            assert classify_task_type(msg) == TaskType.CHAT
            assert classify_message_intent(msg) == "chat"

    def test_greeting_inside_long_text_doesnt_match(self):
        # "hi" inside a long message is not a greeting.
        long = "hi there, can you walk me through the entire architecture of the dispatcher subsystem"
        assert classify_task_type(long) != TaskType.CHAT


class TestTrivialQuestions:
    """The new short-question short-circuit."""

    def test_short_what_questions_are_chat(self):
        for msg in [
            "what's 2+2?",
            "what's the time?",
            "what is python?",
            "whats the date today",
        ]:
            assert classify_task_type(msg) == TaskType.CHAT, msg
            assert classify_message_intent(msg) == "chat", msg

    def test_short_yes_no_questions_are_chat(self):
        for msg in [
            "is it raining?",
            "is python typed?",   # bare "is X" — the benchmark miss
            "are you sure",
            "do you sleep",
            "can you help",
            "does this work?",
            "did the test pass?",
        ]:
            assert classify_task_type(msg) == TaskType.CHAT, msg
            assert classify_message_intent(msg) == "chat", msg

    def test_arithmetic_is_chat(self):
        # The benchmark case that prompted this feature.
        assert classify_task_type("what's 2+2?") == TaskType.CHAT
        assert classify_task_type("3 * 7 = ?") == TaskType.CHAT
        assert classify_task_type("5 plus 4") == TaskType.CHAT

    def test_long_explainers_still_go_to_explain(self):
        # Boundary case: starts with "what is" but is long → should
        # remain an EXPLAIN task, not collapse to CHAT.
        long_explain = (
            "what is the relationship between RRF fusion, cosine "
            "similarity, and BM25, and how do they interact in a "
            "memory retrieval system"
        )
        assert classify_task_type(long_explain) == TaskType.EXPLAIN

    def test_long_yes_no_still_falls_through(self):
        long_q = (
            "is it true that the dispatcher applies preemption hooks "
            "before falling through to the capability-fallback path? "
            "and if so, when does it skip the role match?"
        )
        # Length > 60, so the trivial-question short-circuit doesn't
        # fire. It's not a greeting either, so this falls through
        # the table to a more specific match or None.
        assert classify_task_type(long_q) != TaskType.CHAT


class TestExistingTaskTypes:
    """Make sure the trivial-question addition didn't shadow other
    task-type matches."""

    def test_fetch_url_still_matches(self):
        assert classify_task_type("can you fetch https://example.com") in {
            TaskType.FETCH,
        }
        assert classify_message_intent("can you fetch https://example.com") == "tool"

    def test_explain_still_matches_long_questions(self):
        # "what does X mean" is an EXPLAIN signal, length > 60.
        msg = "what does the dispatcher actually do when no worker matches the requested task type"
        assert classify_task_type(msg) == TaskType.EXPLAIN

    def test_test_run_keyword(self):
        assert classify_task_type("run the test suite please") == TaskType.TEST_RUN

    def test_git_keyword(self):
        assert classify_task_type("git commit and push the changes") == TaskType.GIT_OPS


class TestFastTaskRouting:
    """prefer_fast tasks (CHAT, TRIAGE, LINT) must pick the smallest
    worker available — not whichever happens to be first in the dict."""

    def test_chat_picks_smaller_vram_worker(self):
        from towel.nodes.roles import TaskType, best_node_for_task

        small = {
            "id": "fast-w", "enabled": True, "busy": False,
            "assigned_tasks": [TaskType.CHAT],
            "capabilities": {"total_vram_mb": 4096, "context_window": 8192, "backend": "llama"},
        }
        big = {
            "id": "slow-w", "enabled": True, "busy": False,
            "assigned_tasks": [TaskType.CHAT],
            "capabilities": {"total_vram_mb": 24000, "context_window": 8192, "backend": "llama"},
        }
        # Order shouldn't matter for the choice — try both.
        for nodes in ([small, big], [big, small]):
            chosen = best_node_for_task(TaskType.CHAT, nodes)
            assert chosen is small, f"got {chosen['id']}"

    def test_chat_role_prefers_classifier_path(self):
        # intent=chat → NodeRole.CLASSIFIER per _role_for_intent.
        # Verify the dispatcher fallback doesn't accidentally pick
        # the heaviest worker for chat-class requests.
        from towel.gateway.dispatcher import _role_for_intent
        from towel.nodes.roles import NodeRole

        assert _role_for_intent("chat") == NodeRole.CLASSIFIER
        assert _role_for_intent("task") == NodeRole.INFERENCE
        assert _role_for_intent("tool") == NodeRole.TOOL_WORKER


class TestIntentClassification:
    def test_url_is_tool(self):
        assert classify_message_intent("check https://google.com") == "tool"

    def test_fetch_is_tool(self):
        assert classify_message_intent("fetch the latest data") == "tool"

    def test_neutral_returns_none(self):
        # A medium-length task-y prompt without obvious cues falls to
        # the LLM classifier (returns None).
        assert classify_message_intent("refactor this code base") is None

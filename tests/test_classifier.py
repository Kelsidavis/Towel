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

    def test_polite_action_requests_do_not_route_as_chat(self):
        cases = {
            "can you run the tests?": TaskType.TEST_RUN,
            "could you add tests for the parser": TaskType.TEST_GEN,
            "would you fix the dispatcher bug": TaskType.REFACTOR,
            "please update the README": TaskType.GENERATE,
            "can you create a file for this": TaskType.GENERATE,
        }
        for msg, expected in cases.items():
            assert classify_task_type(msg) == expected, msg
            assert classify_message_intent(msg) != "chat", msg

    def test_polite_help_request_stays_chat(self):
        assert classify_task_type("can you help") == TaskType.CHAT
        assert classify_message_intent("can you help") == "chat"

    def test_git_keyword(self):
        assert classify_task_type("git commit and push the changes") == TaskType.GIT_OPS

    def test_build_keyword_only_matches_compile_context(self):
        """Bare "build a X" was silently classified as TaskType.BUILD
        (prefer_fast), which routed code-generation orchestrations to
        the smaller worker. Live observation:
        "Build a tic-tac-toe game" landed on Gemma-4-E2B 4GB instead
        of SparklesMint 27B. Only the explicit compile/package
        phrases should match BUILD now."""
        # Compile-context phrases still match.
        assert classify_task_type("compile the kernel") == TaskType.BUILD
        assert classify_task_type("npm run build") == TaskType.BUILD
        assert classify_task_type("pip install towel") == TaskType.BUILD
        assert classify_task_type("build the project") == TaskType.BUILD
        assert classify_task_type("build the docker image") == TaskType.BUILD
        # Generation-context phrases must NOT match BUILD — they
        # should fall through to GENERATE (or whichever later rule
        # catches them), which is prefer_quality and lands on the
        # larger worker.
        assert classify_task_type("build a tic-tac-toe game") != TaskType.BUILD
        assert classify_task_type("Build a 3D pygame demo") != TaskType.BUILD
        assert classify_task_type("build a snake clone in python") != TaskType.BUILD


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


class TestQualityTaskRouting:
    """prefer_quality tasks (EXPLAIN, TRANSLATE, ANALYZE, …) must pick
    the largest available worker. Live observation in 2026-05: an
    "explain X" prompt routed to the 4GB small worker (where every
    other prefer_quality task already favors the big one), the small
    worker timed out at 5 min, and the user waited the full
    worker_inference_timeout for nothing. EXPLAIN and TRANSLATE were
    falling through to default least-loaded sort because they alone
    of the INFERENCE-role tasks had no prefer_* flag set — a config
    oversight, not a deliberate choice."""

    def _stub_workers(self, task):
        small = {
            "id": "small-w", "enabled": True, "busy": False,
            "assigned_tasks": [task],
            "capabilities": {
                "total_vram_mb": 4096, "context_window": 32768,
                "backend": "llama",
            },
        }
        big = {
            "id": "big-w", "enabled": True, "busy": False,
            "assigned_tasks": [task],
            "capabilities": {
                "total_vram_mb": 24000, "context_window": 65536,
                "backend": "llama",
            },
        }
        return small, big

    def test_explain_routes_to_larger_worker(self):
        from towel.nodes.roles import TaskType, best_node_for_task
        small, big = self._stub_workers(TaskType.EXPLAIN)
        for nodes in ([small, big], [big, small]):
            chosen = best_node_for_task(TaskType.EXPLAIN, nodes)
            assert chosen is big, f"got {chosen['id']}"

    def test_translate_routes_to_larger_worker(self):
        from towel.nodes.roles import TaskType, best_node_for_task
        small, big = self._stub_workers(TaskType.TRANSLATE)
        for nodes in ([small, big], [big, small]):
            chosen = best_node_for_task(TaskType.TRANSLATE, nodes)
            assert chosen is big, f"got {chosen['id']}"

    def test_every_inference_task_has_a_routing_preference(self):
        """A regression guard for the EXPLAIN/TRANSLATE oversight:
        every INFERENCE-role task should have either prefer_quality
        or prefer_fast set explicitly. Falling through to the default
        least-loaded sort means an idle fleet with two equally-loaded
        workers picks arbitrarily — which is exactly how this bug
        slipped in: live an "explain" landed on the small worker and
        timed out. New INFERENCE tasks added in the future will trip
        this test if they forget to specify a preference."""
        from towel.nodes.roles import TASK_REQUIREMENTS, NodeRole
        missing = []
        for task, reqs in TASK_REQUIREMENTS.items():
            roles = reqs.get("roles", [])
            if NodeRole.INFERENCE not in roles:
                continue
            if not (reqs.get("prefer_quality") or reqs.get("prefer_fast")):
                missing.append(task)
        assert not missing, (
            f"INFERENCE-role tasks without prefer_quality/prefer_fast: "
            f"{[t.value for t in missing]}"
        )


class TestIntentClassification:
    def test_url_is_tool(self):
        assert classify_message_intent("check https://google.com") == "tool"

    def test_fetch_is_tool(self):
        assert classify_message_intent("fetch the latest data") == "tool"

    def test_neutral_returns_none(self):
        # A medium-length task-y prompt without obvious cues falls to
        # the LLM classifier (returns None).
        assert classify_message_intent("refactor this code base") is None

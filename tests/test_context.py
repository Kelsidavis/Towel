"""Tests for context window management."""

from towel.agent.context import (
    ContextBudget,
    count_tokens_fallback,
    fit_messages,
    maybe_compact_conversation,
)
from towel.agent.conversation import Conversation, Role


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


# Use a deterministic counter: 1 token per word
def word_counter(text: str) -> int:
    return len(text.split()) if text.strip() else 0


class TestContextBudget:
    def test_input_budget(self):
        b = ContextBudget(context_window=8192, max_output_tokens=4096)
        assert b.input_budget == 4096

    def test_remaining(self):
        b = ContextBudget(
            context_window=8192,
            max_output_tokens=4096,
            system_tokens=500,
            message_tokens=1000,
        )
        assert b.remaining == 4096 - 500 - 1000


class TestFitMessages:
    def test_all_messages_fit(self):
        messages = [
            _msg("user", "hello"),
            _msg("assistant", "hi there"),
            _msg("user", "how are you"),
        ]
        fitted, budget = fit_messages(
            system_content="You are helpful.",
            messages=messages,
            context_window=8192,
            max_output_tokens=4096,
            token_counter=word_counter,
        )
        assert len(fitted) == 3
        assert budget.messages_dropped == 0

    def test_drops_oldest_messages(self):
        # Create many short messages that exceed the budget
        messages = [_msg("user", f"message {i}") for i in range(50)]
        fitted, budget = fit_messages(
            system_content="System prompt.",
            messages=messages,
            context_window=100,
            max_output_tokens=30,
            token_counter=word_counter,
        )
        assert budget.messages_dropped > 0
        assert budget.messages_included < 50
        # Last message should always be present
        assert fitted[-1]["content"] == messages[-1]["content"]

    def test_always_includes_last_message(self):
        messages = [
            _msg("user", "old message " * 100),
            _msg("assistant", "old response " * 100),
            _msg("user", "current question"),
        ]
        fitted, budget = fit_messages(
            system_content="System.",
            messages=messages,
            context_window=100,
            max_output_tokens=30,
            token_counter=word_counter,
        )
        assert len(fitted) >= 1
        assert fitted[-1]["content"] == "current question"

    def test_preserves_latest_user_even_if_last_message_is_assistant(self):
        messages = [
            _msg("user", "older context " * 40),
            _msg("user", "real question"),
            _msg("assistant", "very long answer " * 40),
        ]
        fitted, budget = fit_messages(
            system_content="System.",
            messages=messages,
            context_window=120,
            max_output_tokens=40,
            token_counter=word_counter,
        )
        contents = [m["content"] for m in fitted]
        assert "real question" in contents
        assert fitted[-1]["role"] == "assistant"

    def test_truncates_huge_latest_message_to_budget(self):
        messages = [
            _msg("user", "question"),
            _msg("assistant", "very long answer " * 200),
        ]
        fitted, budget = fit_messages(
            system_content="System.",
            messages=messages,
            context_window=90,
            max_output_tokens=30,
            token_counter=word_counter,
        )
        assert fitted[-1]["content"].startswith("...")
        assert budget.message_tokens <= budget.input_budget - budget.system_tokens

    def test_empty_messages(self):
        fitted, budget = fit_messages(
            system_content="System.",
            messages=[],
            context_window=8192,
            max_output_tokens=4096,
            token_counter=word_counter,
        )
        assert fitted == []
        assert budget.messages_included == 0

    def test_preserves_chronological_order(self):
        messages = [_msg("user", f"msg {i}") for i in range(5)]
        fitted, budget = fit_messages(
            system_content="Sys.",
            messages=messages,
            context_window=8192,
            max_output_tokens=4096,
            token_counter=word_counter,
        )
        for i in range(len(fitted) - 1):
            # Messages should be in order
            idx_a = next(j for j, m in enumerate(messages) if m["content"] == fitted[i]["content"])
            idx_b = next(
                j for j, m in enumerate(messages) if m["content"] == fitted[i + 1]["content"]
            )
            assert idx_a < idx_b

    def test_budget_stats_correct(self):
        messages = [_msg("user", "hello world")]
        fitted, budget = fit_messages(
            system_content="System prompt here.",
            messages=messages,
            context_window=1000,
            max_output_tokens=200,
            token_counter=word_counter,
        )
        assert budget.context_window == 1000
        assert budget.max_output_tokens == 200
        assert budget.system_tokens > 0
        assert budget.message_tokens > 0
        assert budget.messages_included == 1
        assert budget.messages_dropped == 0

    def test_huge_system_prompt(self):
        fitted, budget = fit_messages(
            system_content="word " * 10000,
            messages=[_msg("user", "hi")],
            context_window=100,
            max_output_tokens=50,
            token_counter=word_counter,
        )
        # System prompt exceeds budget — should return empty
        assert fitted == []

    def test_with_tool_messages(self):
        messages = [
            _msg("user", "read my file"),
            _msg("assistant", "Let me read that."),
            _msg("tool", "[read_file] contents of the file here " * 50),
            _msg("assistant", "Here's what the file says."),
            _msg("user", "now summarize it"),
        ]
        fitted, budget = fit_messages(
            system_content="System.",
            messages=messages,
            context_window=500,
            max_output_tokens=100,
            token_counter=word_counter,
        )
        # Should include at least the last user message
        assert fitted[-1]["content"] == "now summarize it"
        assert budget.messages_included <= 5

    def test_adds_compacted_summary_for_dropped_messages(self):
        messages = []
        for i in range(8):
            messages.append(_msg("user", f"Question {i} with extra context"))
            messages.append(_msg("assistant", f"Answer {i} with extra detail"))

        fitted, budget = fit_messages(
            system_content="System.",
            messages=messages,
            context_window=120,
            max_output_tokens=40,
            token_counter=word_counter,
        )

        assert fitted[0]["role"] == "system"
        assert "Compacted summary" in fitted[0]["content"]
        assert any("Question 0" in fitted[0]["content"] or "Answer 0" in fitted[0]["content"] for _ in [0])

    def test_skips_large_old_tool_message_when_fitting_recent_context(self):
        messages = [
            _msg("user", "first question"),
            _msg("tool", "[read_file] " + ("blob " * 200)),
            _msg("assistant", "short answer"),
            _msg("user", "follow-up"),
        ]
        fitted, budget = fit_messages(
            system_content="System.",
            messages=messages,
            context_window=90,
            max_output_tokens=30,
            token_counter=word_counter,
        )

        contents = [m["content"] for m in fitted]
        assert "follow-up" in contents
        assert "short answer" in contents


class TestFallbackTokenCounter:
    def test_basic(self):
        assert count_tokens_fallback("hello world!!") == 3  # 13 chars / 4
        assert count_tokens_fallback("") == 1  # minimum 1

    def test_long_text(self):
        text = "a" * 4000
        assert count_tokens_fallback(text) == 1000


class TestPersistentCompaction:
    def test_compacts_over_budget_conversation(self):
        conv = Conversation()
        for i in range(12):
            conv.add(Role.USER, f"Question {i} " * 8)
            conv.add(Role.ASSISTANT, f"Answer {i} " * 8)

        changed = maybe_compact_conversation(
            conv,
            system_content="System.",
            context_window=140,
            max_output_tokens=40,
            token_counter=word_counter,
            keep_recent=6,
            max_summary_tokens=40,
        )

        assert changed is True
        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[0].metadata["compacted"] is True
        assert "Compacted summary" in conv.messages[0].content
        assert len(conv.messages) <= 7

    def test_keeps_recent_and_pinned_messages(self):
        conv = Conversation()
        conv.add(Role.USER, "old pinned question " * 6)
        conv.messages[0].pinned = True
        for i in range(10):
            conv.add(Role.USER, f"Q{i} " * 6)
            conv.add(Role.ASSISTANT, f"A{i} " * 6)

        changed = maybe_compact_conversation(
            conv,
            system_content="System.",
            context_window=140,
            max_output_tokens=40,
            token_counter=word_counter,
            keep_recent=4,
            max_summary_tokens=40,
        )

        assert changed is True
        contents = [m.content for m in conv.messages]
        assert any("old pinned question" in c for c in contents)
        assert any("Q9" in c for c in contents)

    def test_noop_when_conversation_fits(self):
        conv = Conversation()
        conv.add(Role.USER, "short question")
        conv.add(Role.ASSISTANT, "short answer")

        changed = maybe_compact_conversation(
            conv,
            system_content="System.",
            context_window=500,
            max_output_tokens=100,
            token_counter=word_counter,
        )

        assert changed is False
        assert len(conv.messages) == 2

    def test_prefers_codex_external_compaction_when_available(self, monkeypatch):
        conv = Conversation()
        for i in range(12):
            conv.add(Role.USER, f"Question {i} " * 8)
            conv.add(Role.ASSISTANT, f"Answer {i} " * 8)

        monkeypatch.setattr(
            "towel.agent.context._maybe_external_compact_summary",
            lambda text, max_words: "Dense external summary",
        )

        changed = maybe_compact_conversation(
            conv,
            system_content="System.",
            context_window=140,
            max_output_tokens=40,
            token_counter=word_counter,
            keep_recent=6,
            max_summary_tokens=40,
        )

        assert changed is True
        assert "via Codex" in conv.messages[0].content
        assert "Dense external summary" in conv.messages[0].content

    def test_falls_back_to_local_summary_when_external_unavailable(self, monkeypatch):
        conv = Conversation()
        for i in range(12):
            conv.add(Role.USER, f"Question {i} " * 8)
            conv.add(Role.ASSISTANT, f"Answer {i} " * 8)

        monkeypatch.setattr(
            "towel.agent.context._maybe_external_compact_summary",
            lambda text, max_words: None,
        )

        changed = maybe_compact_conversation(
            conv,
            system_content="System.",
            context_window=140,
            max_output_tokens=40,
            token_counter=word_counter,
            keep_recent=6,
            max_summary_tokens=40,
        )

        assert changed is True
        assert "Compacted summary" in conv.messages[0].content
        assert "via Codex" not in conv.messages[0].content

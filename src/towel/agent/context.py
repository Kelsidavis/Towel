"""Context window manager — fits conversations into token budgets.

Strategy: sliding window with priority preservation.
  1. System prompt + tool definitions are always included (non-negotiable)
  2. The most recent user message is always included
  3. Fill remaining budget with the most recent messages, working backwards
  4. If a single message exceeds the budget, truncate it from the front

This keeps the agent grounded in the current task while maintaining
as much recent context as the model can handle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("towel.agent.context")


@dataclass
class ContextBudget:
    """Token budget breakdown."""

    context_window: int
    max_output_tokens: int
    system_tokens: int = 0
    message_tokens: int = 0
    messages_included: int = 0
    messages_dropped: int = 0

    @property
    def input_budget(self) -> int:
        """Tokens available for input (context - reserved output)."""
        return self.context_window - self.max_output_tokens

    @property
    def remaining(self) -> int:
        return self.input_budget - self.system_tokens - self.message_tokens


def count_tokens_fallback(text: str) -> int:
    """Rough token estimate when no tokenizer is available (~4 chars per token)."""
    return max(1, len(text) // 4)


def fit_messages(
    system_content: str,
    messages: list[dict[str, str]],
    context_window: int,
    max_output_tokens: int,
    token_counter: Callable[[str], int] | None = None,
) -> tuple[list[dict[str, str]], ContextBudget]:
    """Select messages that fit within the token budget.

    Args:
        system_content: The system prompt (always included).
        messages: Chat messages in chronological order [{role, content}, ...].
        context_window: Total context window size in tokens.
        max_output_tokens: Tokens reserved for generation output.
        token_counter: Function that counts tokens in a string.
            Falls back to char-based estimate if None.

    Returns:
        (fitted_messages, budget) — the messages that fit, plus budget stats.
    """
    count = token_counter or count_tokens_fallback
    budget = ContextBudget(
        context_window=context_window,
        max_output_tokens=max_output_tokens,
    )

    # System prompt is non-negotiable
    # Add overhead for chat template formatting (~4 tokens per message for role tags etc.)
    template_overhead_per_msg = 4
    budget.system_tokens = count(system_content) + template_overhead_per_msg

    if budget.remaining <= 0:
        log.warning("System prompt alone exceeds context budget")
        return [], budget

    if not messages:
        return [], budget

    # Count tokens for each message
    msg_tokens: list[int] = []
    for msg in messages:
        tokens = count(msg["content"]) + template_overhead_per_msg
        msg_tokens.append(tokens)

    # Always include the last message (current user turn)
    # Then fill backwards with as many recent messages as fit
    selected_indices: list[int] = []
    tokens_used = 0

    for i in range(len(messages) - 1, -1, -1):
        cost = msg_tokens[i]

        if tokens_used + cost > budget.remaining:
            # If this is the very last message (current turn), truncate to fit
            if i == len(messages) - 1 and not selected_indices:
                selected_indices.append(i)
                tokens_used += min(cost, budget.remaining)
                log.debug(f"Truncating current message to fit ({cost} -> {budget.remaining} tokens)")
            break

        selected_indices.append(i)
        tokens_used += cost

    # Reverse to get chronological order
    selected_indices.reverse()

    available_before = budget.remaining
    budget.message_tokens = tokens_used
    budget.messages_included = len(selected_indices)
    budget.messages_dropped = len(messages) - len(selected_indices)

    if budget.messages_dropped > 0:
        log.info(
            f"Context window: kept {budget.messages_included}/{len(messages)} messages "
            f"({budget.message_tokens} tokens), dropped {budget.messages_dropped} oldest"
        )

    fitted = [messages[i] for i in selected_indices]

    # If the last message was truncated to fit, trim its content
    if fitted and msg_tokens[selected_indices[-1]] > available_before:
        max_chars = available_before * 4  # reverse the ~4 chars/token estimate
        fitted[-1] = {
            **fitted[-1],
            "content": "..." + fitted[-1]["content"][-max_chars:],
        }

    return fitted, budget

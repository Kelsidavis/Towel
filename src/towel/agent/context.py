"""Context window manager — fits conversations into token budgets.

Strategy: sliding window with priority preservation and lightweight compaction.
  1. System prompt + tool definitions are always included (non-negotiable)
  2. The latest message and latest user message are anchored
  3. Fill remaining budget with the most recent messages, working backwards
  4. If older messages are dropped, replace them with a compact summary when possible
  5. If a single anchored message exceeds the budget, truncate it from the front

This keeps the agent grounded in the current task while preserving more of the
conversation's intent than a pure drop-oldest strategy.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from towel.agent.conversation import Conversation, Message, Role

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


def _latest_user_index(messages: list[dict[str, str]]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            return i
    return None


def _compact_message_line(msg: dict[str, str]) -> str | None:
    role = msg["role"]
    content = msg["content"].strip()
    if not content:
        return None

    if role == "tool":
        if content.startswith("[") and "]" in content:
            tool_name = content[1 : content.index("]")]
            status_match = re.search(r"status:\s+(\w+)", content)
            status = f" ({status_match.group(1)})" if status_match else ""
            return f"- Tool: {tool_name}{status}"
        return "- Tool result"

    if role == "system":
        return None

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None

    first_line = lines[0]
    if len(first_line) > 120:
        first_line = first_line[:120] + "..."

    label = "Q" if role == "user" else "A"
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
    if code_blocks:
        snippet = code_blocks[0].strip()
        if len(snippet) > 180:
            snippet = snippet[:180] + "\n..."
        return f"- {label}: {first_line}\n  ```\n  {snippet}\n  ```"
    return f"- {label}: {first_line}"


def _build_compact_summary(messages: list[dict[str, str]]) -> str:
    lines = [f"[Compacted summary of {len(messages)} earlier messages]"]
    for msg in messages:
        line = _compact_message_line(msg)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _build_compact_summary_from_messages(messages: list[Message]) -> str:
    lines = [f"[Compacted summary of {len(messages)} earlier messages]"]
    for msg in messages:
        if msg.role == Role.SYSTEM and msg.metadata.get("compacted"):
            lines.append(msg.content.strip())
            continue
        line = _compact_message_line(msg.to_chat_dict())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n..."


def _truncate_from_front(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def _maybe_external_compact_summary(text: str, max_words: int) -> str | None:
    """Prefer subscription-backed Codex compaction when available."""
    try:
        from towel.skills.builtin.codex_skill import codex_compact_available, codex_compact_text
    except Exception:
        return None

    if not codex_compact_available():
        return None

    result = codex_compact_text(
        text,
        goal="compact prior conversation state for future turns",
        max_words=max_words,
    ).strip()
    if not result or result.startswith("Error calling Codex compactor:"):
        return None
    return result


def maybe_compact_conversation(
    conversation: Conversation,
    system_content: str,
    context_window: int,
    max_output_tokens: int,
    token_counter: Callable[[str], int] | None = None,
    keep_recent: int = 8,
    max_summary_tokens: int = 256,
) -> bool:
    """Persistently compact older conversation history when over budget.

    Replaces older non-pinned messages with a compact system summary so future
    turns do not keep paying to re-scan the same stale history.
    """
    count = token_counter or count_tokens_fallback
    template_overhead_per_msg = 4
    input_budget = context_window - max_output_tokens
    system_tokens = count(system_content) + template_overhead_per_msg
    message_tokens = sum(count(m.content) + template_overhead_per_msg for m in conversation.messages)

    if system_tokens + message_tokens <= input_budget:
        return False
    if len(conversation.messages) <= keep_recent + 1:
        return False

    recent_start = max(0, len(conversation.messages) - keep_recent)
    keep_indices = {i for i in range(recent_start, len(conversation.messages))}
    keep_indices.update(i for i, msg in enumerate(conversation.messages) if msg.pinned)

    compressible = [m for i, m in enumerate(conversation.messages) if i not in keep_indices]
    if not compressible:
        return False

    local_summary_text = _build_compact_summary_from_messages(compressible)
    summary_token_budget = min(
        max_summary_tokens,
        max(32, input_budget - system_tokens - (keep_recent * template_overhead_per_msg)),
    )
    summary_text = _maybe_external_compact_summary(local_summary_text, max_words=summary_token_budget)
    if summary_text:
        summary_text = "[Compacted summary of earlier messages via Codex]\n" + summary_text
    else:
        summary_text = local_summary_text

    summary_text = _truncate_text_to_tokens(summary_text, summary_token_budget)
    if not summary_text:
        return False

    summary_msg = Message(
        role=Role.SYSTEM,
        content=summary_text,
        metadata={"compacted": True, "original_count": len(compressible)},
    )
    kept_messages = [m for i, m in enumerate(conversation.messages) if i in sorted(keep_indices)]
    conversation.messages = [summary_msg] + kept_messages
    log.info(
        "Compacted conversation history: %s older messages summarized, %s kept",
        len(compressible),
        len(kept_messages),
    )
    return True


def fit_messages(
    system_content: str,
    messages: list[dict[str, str]],
    context_window: int,
    max_output_tokens: int,
    token_counter: Callable[[str], int] | None = None,
    pinned_indices: set[int] | None = None,
) -> tuple[list[dict[str, str]], ContextBudget]:
    """Select messages that fit within the token budget.

    Args:
        system_content: The system prompt (always included).
        messages: Chat messages in chronological order [{role, content}, ...].
        context_window: Total context window size in tokens.
        max_output_tokens: Tokens reserved for generation output.
        token_counter: Function that counts tokens in a string.
            Falls back to char-based estimate if None.
        pinned_indices: Set of message indices that must always be included
            (even when older messages are dropped for space).

    Returns:
        (fitted_messages, budget) — the messages that fit, plus budget stats.
    """
    count = token_counter or count_tokens_fallback
    pinned = set(pinned_indices or set())
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

    # Reserve space for pinned messages and anchored recency first
    latest_idx = len(messages) - 1
    latest_user_idx = _latest_user_index(messages)
    anchors = {latest_idx}
    if latest_user_idx is not None:
        anchors.add(latest_user_idx)
    pinned.update(anchors)

    selected_indices: list[int] = []
    tokens_used = 0

    pinned_cost = 0
    for i in sorted(pinned):
        if 0 <= i < len(messages):
            pinned_cost += msg_tokens[i]

    # Fill backwards with recent messages, skipping pinned (added separately)
    for i in range(len(messages) - 1, -1, -1):
        if i in pinned:
            continue  # pinned messages are added regardless
        cost = msg_tokens[i]

        if tokens_used + cost + pinned_cost > budget.remaining:
            # Older tool outputs are often verbose and low-value once we have newer turns.
            if messages[i]["role"] == "tool":
                continue
            break

        selected_indices.append(i)
        tokens_used += cost

    # Add pinned/anchored messages
    for i in sorted(pinned):
        if 0 <= i < len(messages) and i not in selected_indices:
            selected_indices.append(i)
            tokens_used += msg_tokens[i]

    # Sort to get chronological order
    selected_indices.sort()
    fitted = [messages[i] for i in selected_indices]

    input_available = budget.input_budget - budget.system_tokens
    if tokens_used > input_available and fitted:
        last_idx = selected_indices[-1]
        excess = tokens_used - input_available
        keep_tokens = max(1, msg_tokens[last_idx] - excess - template_overhead_per_msg)
        fitted[-1] = {
            **fitted[-1],
            "content": _truncate_from_front(fitted[-1]["content"], keep_tokens),
        }
        tokens_used = input_available

    budget.message_tokens = tokens_used
    budget.messages_included = len(selected_indices)
    budget.messages_dropped = len(messages) - len(selected_indices)

    if budget.messages_dropped > 0:
        log.info(
            f"Context window: kept {budget.messages_included}/{len(messages)} messages "
            f"({budget.message_tokens} tokens), dropped {budget.messages_dropped} oldest"
        )

    dropped_indices = [i for i in range(len(messages)) if i not in selected_indices]
    if dropped_indices:
        summary_source = [messages[i] for i in dropped_indices if messages[i]["role"] != "system"]
        if summary_source:
            summary_text = _build_compact_summary(summary_source)
            summary_tokens = count(summary_text) + template_overhead_per_msg
            remaining_after_selection = budget.remaining
            if summary_tokens > remaining_after_selection:
                summary_text = _truncate_text_to_tokens(
                    summary_text,
                    max(1, remaining_after_selection - template_overhead_per_msg),
                )
                summary_tokens = count(summary_text) + template_overhead_per_msg
            if summary_text and summary_tokens <= remaining_after_selection:
                fitted.insert(0, {"role": "system", "content": summary_text})
                budget.message_tokens += summary_tokens
                budget.messages_included += 1

    return fitted, budget

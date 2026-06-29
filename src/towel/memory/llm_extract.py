"""LLM-based memory extraction — catches what regex misses.

The regex extractor in ``auto_capture`` is fast and predictable but
narrow by construction. Cases regex can't reach:

* multi-sentence context ("I tried postgres last year. It didn't
  scale. So now we run cockroach." → preference=cockroach)
* indirect mention ("our infra still runs on the old vendor")
* paraphrase that doesn't match anchor cues

This module pairs a small extraction prompt with a strict JSON-only
output contract and a parser that drops malformed lines. Conservative
on purpose: the LLM is told to return nothing rather than guess.

Sync usage:
    captures = await extract_via_llm(text, agent_step_async)
    for cap in captures:
        store.remember(cap.key, cap.content, memory_type=cap.memory_type,
                       source="llm_extract")

Where ``agent_step_async`` is any awaitable callable accepting a
prompt string and returning the model's reply as plain text. Lets
this module stay decoupled from the runtime layer.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("towel.memory.llm_extract")


# Use a literal placeholder + .replace() instead of .format() so the
# JSON example with curly braces doesn't trip the format-string parser.
_PROMPT_TEMPLATE = (
    """\
You are an extraction assistant. Read the user's text below and \
return memorable facts as a strict JSON array. Each item must be:

  {"key": "<short-snake-case-id>", "content": "<the fact>", """
    """"type": "<user|preference|project|fact>"}

Rules:
- Output ONLY the JSON array, no preamble, no markdown fences, no commentary.
- If nothing memorable is in the text, output exactly: []
- Prefer fewer high-quality captures over many uncertain ones.
- Extract only facts the user stated about themselves or their work.
- Never record judgments, safety assessments, or characterizations of the
  user's intent. If the text is a request or opinion, output [].
- "user" = stable facts about the user (role, employer, location).
- "preference" = how they like things done.
- "project" = current work, deadlines, ongoing initiatives.
- "fact" = anything else worth remembering.

USER TEXT:
__TEXT__

JSON ARRAY:"""
)


@dataclass(frozen=True)
class LLMCapture:
    """One LLM-proposed memory; parallels auto_capture.Capture."""

    key: str
    content: str
    memory_type: str


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_VALID_TYPES = {"user", "preference", "project", "fact"}


def parse_response(raw: str) -> list[LLMCapture]:
    """Tolerantly parse an LLM extraction response into LLMCaptures.

    Strips markdown fences if the model added them despite the prompt,
    locates the first '[' and last ']' to extract just the JSON array,
    drops items that don't have all three required string fields or
    whose type isn't in the allowed set. Returns ``[]`` on any failure
    — better silent than spurious.
    """
    if not raw:
        return []
    text = _FENCE_RE.sub("", raw).strip()
    if not text:
        return []
    # Find the JSON array bounds — the model often prefixes a few
    # words despite being told not to.
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end < start:
        return []
    blob = text[start : end + 1]
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        log.debug("LLM extract response failed JSON parse: %s", exc)
        return []
    if not isinstance(parsed, list):
        return []
    out: list[LLMCapture] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        content = item.get("content")
        mtype = item.get("type", "fact")
        if not (isinstance(key, str) and isinstance(content, str)):
            continue
        if not key.strip() or not content.strip():
            continue
        if mtype not in _VALID_TYPES:
            mtype = "fact"
        out.append(
            LLMCapture(
                key=key.strip(),
                content=content.strip(),
                memory_type=mtype,
            )
        )
    return out


# Track in-flight queries so a chatty user doesn't pile up parallel
# extractions for the same text. The runtime fires once and drops
# duplicate requests until the first one finishes.
_inflight: set[str] = set()


def schedule_background_extraction(
    text: str,
    step: Callable[[str], Awaitable[str]],
    store: Any,
    *,
    scope: str | None = None,
) -> bool:
    """Fire-and-forget LLM extraction in the current asyncio loop.

    Returns True if a task was actually scheduled. Returns False
    when:

    * the text is empty,
    * we're not in an asyncio loop (sync caller — runtime always
      is async, so this is just a guard),
    * an extraction for this exact text is already in flight.

    Failures inside the task are swallowed at debug level — the
    user's response must never be blocked or fail because of
    background extraction. Captures land with source set to
    ``llm_extract:auto`` so they're easy to audit and tidy
    independently of operator-driven extract calls.
    """
    import asyncio

    text = (text or "").strip()
    if not text:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    if text in _inflight:
        return False
    _inflight.add(text)

    async def _run() -> None:
        try:
            from towel.memory.guard import reject_reason

            captures = await extract_via_llm(text, step)
            for cap in captures:
                if reject_reason(cap.key, cap.content) is not None:
                    log.debug("Guard refused llm-extract capture: %s", cap.key)
                    continue
                if store.recall(cap.key) is not None:
                    continue
                try:
                    store.remember(
                        cap.key, cap.content,
                        memory_type=cap.memory_type,
                        source="llm_extract:auto",
                        scope=scope,
                    )
                except Exception as exc:
                    log.debug("auto-llm-extract store write failed: %s", exc)
        except Exception as exc:
            log.debug("auto-llm-extract task failed: %s", exc)
        finally:
            _inflight.discard(text)

    loop.create_task(_run())
    return True


async def extract_via_llm(
    text: str, step: Callable[[str], Awaitable[str]]
) -> list[LLMCapture]:
    """Run the extraction prompt against ``step`` and parse the reply.

    ``step`` is any awaitable that maps prompt → response. Decoupling
    from concrete runtime classes keeps this module unit-testable
    without spinning up a backend.
    """
    if not text or not text.strip():
        return []
    prompt = _PROMPT_TEMPLATE.replace("__TEXT__", text.strip())
    try:
        reply = await step(prompt)
    except Exception as exc:
        log.debug("LLM extract step failed: %s", exc)
        return []
    return parse_response(reply or "")

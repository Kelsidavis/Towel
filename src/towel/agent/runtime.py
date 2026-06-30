"""Core agent runtime — manages MLX model loading, inference, and tool dispatch."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from towel.agent.context import (
    estimate_output_reserve,
    fit_messages,
    maybe_compact_conversation,
    select_context_window,
)
from towel.agent.conversation import Conversation, Message, Role
from towel.agent.events import AgentEvent
from towel.agent.instance_lock import acquire_runtime_lock
from towel.agent.tool_parser import parse_tool_calls
from towel.agent.tools_payload import tools_as_openai_functions
from towel.config import TowelConfig
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.agent")

MAX_TOOL_ITERATIONS = 999
EMPTY_TEXT_FALLBACK = (
    "I wasn't able to put together a response that turn — try rephrasing "
    "or asking again."
)

# Loop-detection threshold for the agent's tool-call loop. When the
# same (name, args) fingerprint appears in this many consecutive
# iterations, the model is stuck — break out before burning the
# rest of MAX_TOOL_ITERATIONS (~5h on a 20s/call worker).
TOOL_LOOP_REPEAT_LIMIT = 3


def _tool_call_fingerprint(tool_calls: Any) -> str:
    """Stable fingerprint of one iteration's tool calls for loop detection."""
    import json as _json

    return _json.dumps(
        [(tc.name, tc.arguments) for tc in tool_calls],
        sort_keys=True, default=str,
    )


def _check_tool_loop(history: list[str], fingerprint: str) -> bool:
    """Append `fingerprint`; return True iff the last
    ``TOOL_LOOP_REPEAT_LIMIT`` entries are all identical.

    Mutates ``history`` in place — keeps at most
    ``TOOL_LOOP_REPEAT_LIMIT`` entries.
    """
    history.append(fingerprint)
    if len(history) > TOOL_LOOP_REPEAT_LIMIT:
        history.pop(0)
    return (
        len(history) == TOOL_LOOP_REPEAT_LIMIT
        and len(set(history)) == 1
    )


def mlx_tokenizer_config() -> dict[str, Any]:
    """Return tokenizer config overrides for MLX loads.

    `fix_mistral_regex` was previously passed here, but newer transformers
    versions handle it internally and error if it's forwarded as a kwarg
    (duplicate positional + keyword argument in _patch_mistral_regex).
    """
    return {}


# The trailing-colon patterns previously used ``\b`` (e.g. ``^Unknown tool:\b``)
# which never fires — ``\b`` after ``:`` requires a following word character,
# but the matching strings always have a space there. Use the colon as the
# boundary instead.
#
# The leading ``^Error\b`` catch-all replaces the older
# ``^Error executing\b`` / ``^Error calling\b`` pair. 41 places across
# src/towel/skills/ return ``f"Error: {e}"`` or ``f"Error reading X: {e}"``
# / ``f"Error creating X: {e}"`` / ``f"Error checking X: ..."`` etc. —
# none of which matched the two narrow prefixes, so every one of those
# tool failures was getting classified as a successful result. The
# agent then told the model "Use this result to answer the user
# concretely" on top of a literal "Error: file not found" string.
# ``\b`` after Error keeps "Errors" (e.g. "Errors per page") from
# being mislabeled while catching everything that actually starts
# with the canonical Error word.
_TOOL_ERROR_PATTERNS = (
    re.compile(r"^Error\b", re.IGNORECASE),
    re.compile(r"^Unknown tool:", re.IGNORECASE),
    re.compile(r"^File not found:", re.IGNORECASE),
    re.compile(r"^Not a directory:", re.IGNORECASE),
    re.compile(r"^Invalid index:", re.IGNORECASE),
    re.compile(r"^File too large\b", re.IGNORECASE),
    re.compile(r"^HTTP error:", re.IGNORECASE),
    re.compile(r"^\[4\d\d\]"),  # HTTP 4xx client errors
    re.compile(r"^\[5\d\d\]"),  # HTTP 5xx server errors
    re.compile(r"^Permission denied\b", re.IGNORECASE),
    re.compile(r"^No module named\b", re.IGNORECASE),
)


def tool_result_is_error(result: str) -> bool:
    """Heuristic for whether a tool result represents failure."""
    return any(pattern.search(result) for pattern in _TOOL_ERROR_PATTERNS)


_RETRYABLE_ERROR_HINTS = (
    "Did you mean:",  # registry's close-match suggestion for typo'd tool names
    "Unknown tool:",  # bare unknown-tool errors are usually correctable
)


def _is_retryable_error(result: str) -> bool:
    """Decide whether a tool error is the kind the model should retry once.

    Typo'd tool names and unknown-tool errors are usually fixable by re-emitting
    the call with a corrected name. File-not-found / 4xx / permission errors,
    by contrast, won't change on retry — so the default error policy stays
    conservative and only retryable cases get the encouraging guidance.
    """
    return any(hint in result for hint in _RETRYABLE_ERROR_HINTS)


def format_tool_feedback(tool_name: str, result: str, is_error: bool) -> str:
    """Format tool feedback so the next model step can recover reliably."""
    status = "error" if is_error else "ok"
    if not is_error:
        next_step = (
            "Use this result to answer the user concretely. Do not stop at "
            "saying you will check."
        )
    elif _is_retryable_error(result):
        next_step = (
            "The tool name was wrong but a close match was suggested. Retry "
            "ONCE with the corrected tool name (or stop if nothing applies)."
        )
    else:
        next_step = (
            "The tool failed. Do NOT retry the same tool or try alternative "
            "tools to work around this. Answer the user directly using what "
            "you already know, and mention the limitation briefly."
        )
    return (
        f"[{tool_name}]\n"
        f"status: {status}\n"
        f"result:\n{result}\n\n"
        f"next:\n{next_step}"
    )


# How many times in one turn we'll nudge a model that narrates an action
# ("I'll run the build now…") without actually emitting the tool call, before
# giving up and returning its prose. Small models do this often; one or two
# nudges usually converts the narration into a real tool call, while the cap
# keeps a chronically-narrating model from looping forever.
MAX_AUTONOMY_NUDGES = 2

# Phrases that signal the model described work it *intends* to do rather than
# work it *did*. Anchored to first-person future intent + an action verb (or an
# explicit "stand by"/"please wait") so ordinary final answers — "let me know
# if…", "you can run X" — don't trip it.
_UNFULFILLED_INTENT_RE = re.compile(
    r"\b(?:"
    r"i['’]?ll\s+(?:now\s+|then\s+|go\s+ahead\s+and\s+)?"
    r"(?:run|start|begin|create|write|build|check|fetch|download|install|"
    r"generate|search|look|update|add|make|set\s+up|configure|implement|"
    r"execute|continue|proceed|do)"
    r"|i\s+will\s+(?:now\s+)?(?:run|start|begin|create|write|build|check|"
    r"fetch|download|install|generate|search|update|add|make|configure|"
    r"implement|execute|continue|proceed|do)"
    r"|i['’]?m\s+going\s+to\b|i\s+am\s+going\s+to\b|i\s+am\s+now\b"
    r"|let\s+me\s+(?:now\s+)?(?:run|start|begin|create|write|build|check|"
    r"fetch|download|install|generate|search|update|add|make|configure|"
    r"implement|execute|go\s+ahead|continue|proceed)"
    r"|stand\s+by\b|please\s+wait\b|one\s+moment\b|proceeding\s+to\b"
    r"|next,?\s+i['’]?ll\b"
    r")",
    re.IGNORECASE,
)

# Injected as a user turn when narration-without-action is detected, to convert
# "I'll do X" into an actual tool call on the next generation.
AUTONOMY_NUDGE = (
    "You described an action but did not call any tool, so nothing actually "
    "happened. If the task is not finished, emit the tool call now to do it — "
    "do not say you will. If it is already finished, give the final result."
)


def looks_like_unfulfilled_intent(text: str) -> bool:
    """True when a no-tool-call response reads as narrated-but-unperformed work.

    Lets a tool-capable turn distinguish "I'll run the build now, stand by"
    (narration — keep going) from a genuine final answer (done). Used to decide
    whether to nudge the model to act instead of ending the turn mid-goal.
    """
    if not text or not text.strip():
        return False
    return bool(_UNFULFILLED_INTENT_RE.search(text))


MAX_GOAL_NUDGES = 2

_STOP_TO_ASK_RE = re.compile(
    r"\b(?:"
    r"(?:shall|should)\s+i\s+(?:proceed|continue|go\s+ahead|start|begin)"
    r"|do\s+you\s+want\s+me\s+to\s+(?:proceed|continue|go\s+ahead|start)"
    r"|would\s+you\s+like\s+me\s+to\s+(?:proceed|continue|go\s+ahead|start)"
    r"|before\s+i\s+(?:proceed|continue|start|begin|go)"
    r"|please\s+(?:confirm|specify|clarify)\b"
    r"|can\s+you\s+(?:confirm|specify|clarify|tell\s+me)\b"
    r"|i\s+need\s+(?:you\s+to|more\s+information|to\s+know)\b"
    r")",
    re.IGNORECASE,
)

_COMPLETION_MARKERS_RE = re.compile(
    r"\b(?:"
    r"(?:i(?:'ve| have)?|we(?:'ve)?) (?:completed|finished|created|written|updated|fixed|installed|built|configured)"
    r"|(?:here (?:are|is)|the (?:result|output|answer) (?:is|was))"
    r"|successfully"
    r"|succeeded"
    r"|passed"
    r"|exit code 0"
    r"|done[.!\s]"
    r")",
    re.IGNORECASE,
)

GOAL_COMPLETION_NUDGE = (
    "You stopped to ask the user a question, but the task is not finished and "
    "you have enough context to proceed. Make a reasonable decision and keep "
    "working toward the goal — only ask the user if you truly cannot determine "
    "the answer yourself."
)

TOOL_ERROR_NUDGE = (
    "One or more tools failed during this turn but your response did not address "
    "the errors. Try a different approach, work around the issue, or explain "
    "clearly what went wrong and why you cannot proceed."
)


def _has_unaddressed_tool_errors(text: str, tool_trace: list[dict[str, Any]]) -> bool:
    errors = [t for t in tool_trace if t.get("status") == "error"]
    if not errors:
        return False
    return not bool(re.search(
        r"\b(?:error|fail|issue|problem|couldn't|unable|didn't work)\b",
        text, re.IGNORECASE,
    ))


def looks_like_goal_incomplete(
    text: str,
    tool_trace: list[dict[str, Any]] | None = None,
) -> str | None:
    """If the model stopped prematurely — asking permission to continue, or
    ignoring tool errors — return the appropriate nudge text. Otherwise None.
    """
    if not text or not text.strip():
        return None
    if tool_trace and _has_unaddressed_tool_errors(text, tool_trace):
        return TOOL_ERROR_NUDGE
    if _STOP_TO_ASK_RE.search(text):
        if _COMPLETION_MARKERS_RE.search(text):
            return None
        return GOAL_COMPLETION_NUDGE
    return None


def summarize_tool_trace(tool_trace: list[dict[str, Any]]) -> str:
    """One-line recap of the tools run this turn, for when the model finishes
    work but returns no concluding text — so the user sees what happened rather
    than a blank reply. e.g. "Ran 3 tools: run_command (ok), write_file (ok)…".
    """
    if not tool_trace:
        return ""
    n = len(tool_trace)
    parts = [f"{t.get('tool', '?')} ({t.get('status', '?')})" for t in tool_trace[:5]]
    more = "" if n <= 5 else f", +{n - 5} more"
    noun = "tool" if n == 1 else "tools"
    return f"Ran {n} {noun}: " + ", ".join(parts) + more + "."


@dataclass
class GenerationResult:
    """Result of a single generation step."""

    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens_per_second: float = 0.0
    total_tokens: int = 0


class AgentRuntime:
    """The brain. Loads an MLX model, runs inference, dispatches tools.

    This is the core loop:
      1. Receive a message
      2. Build prompt from conversation + system identity + memory + available tools
      3. Run MLX inference
      4. If tool calls → execute them → feed results back → goto 3
      5. Return final response
    """

    def __init__(
        self,
        config: TowelConfig,
        skills: SkillRegistry | None = None,
        memory: Any | None = None,
    ) -> None:
        self.config = config
        self.skills = skills or SkillRegistry()
        self.memory = memory  # MemoryStore instance
        self.project_context: str | None = None  # Override from coordinator
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._native_tools_supported: bool | None = None
        self._cancel: asyncio.Event = asyncio.Event()
        # Single-thread executor to serialize all MLX Metal operations.
        # Metal command buffers are not thread-safe — concurrent access
        # from the default thread pool crashes the GPU driver.
        self._mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    def cancel(self) -> None:
        """Signal the current generation to stop."""
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    async def load_model(self) -> None:
        """Load the MLX model and tokenizer."""
        if self._loaded:
            return

        acquire_runtime_lock()

        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        self._model, self._tokenizer = await loop.run_in_executor(
            self._mlx_executor, self._load_model_sync
        )
        self._native_tools_supported = self._detect_native_tools_support()
        log.info(
            "Native tools channel: %s",
            "enabled" if self._native_tools_supported else "disabled (fallback to text)",
        )
        self._loaded = True

    def _load_model_sync(self) -> tuple[Any, Any]:
        """Synchronous model loading via mlx_lm."""
        from mlx_lm import load

        model, tokenizer = load(
            self.config.model.name,
            tokenizer_config=mlx_tokenizer_config(),
        )
        return model, tokenizer

    async def generate(
        self,
        conversation: Conversation,
        *,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Run a single generation pass.

        ``temperature``: optional override for the sampler. Defaults
        to ``config.model.temperature``. Pass a low value (e.g. 0.2)
        for tasks where deterministic-ish output matters more than
        creativity — ensemble synthesis uses this to keep arbitration
        consistent across runs of the same input.
        """
        if not self._loaded:
            await self.load_model()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._mlx_executor, self._generate_sync, conversation, temperature,
        )
        return result

    def build_inference_request(self, conversation: Conversation) -> dict[str, Any]:
        """Build a worker-safe inference payload for this conversation."""
        return {"mode": "mlx_prompt", "prompt": self._build_prompt(conversation)}

    async def generate_from_request(self, request: dict[str, Any]) -> GenerationResult:
        """Generate from a prebuilt prompt payload."""
        if not self._loaded:
            await self.load_model()

        if request.get("mode") != "mlx_prompt":
            raise ValueError(f"Unsupported inference mode: {request.get('mode')}")

        loop = asyncio.get_event_loop()
        prompt = request["prompt"]
        max_tokens = request.get("max_tokens")
        return await loop.run_in_executor(
            self._mlx_executor, self._generate_prompt_sync, prompt, max_tokens
        )

    def _make_turboquant_cache(self) -> list | None:
        """Build a TurboQuant prompt cache if enabled, else None."""
        if not self.config.model.turboquant:
            return None
        from towel.agent.turboquant import make_turboquant_cache

        return make_turboquant_cache(
            self._model,
            kv_bits=self.config.model.turboquant_bits,
            qjl_ratio=self.config.model.turboquant_qjl_ratio,
        )

    def _generate_sync(
        self,
        conversation: Conversation,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Synchronous generation via mlx_lm."""
        prompt = self._build_prompt(conversation)
        return self._generate_prompt_sync(prompt, temperature=temperature)

    def _generate_prompt_sync(
        self,
        prompt: str,
        max_tokens: int | None = None,
        *,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Synchronous generation via mlx_lm from a prebuilt prompt."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        effective_temp = (
            temperature
            if temperature is not None
            else self.config.model.temperature
        )
        sampler = make_sampler(
            temp=effective_temp,
            top_p=self.config.model.top_p,
        )
        extra_kwargs: dict[str, Any] = {}
        tq_cache = self._make_turboquant_cache()
        if tq_cache is not None:
            extra_kwargs["prompt_cache"] = tq_cache

        start = time.perf_counter()
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self.config.model.max_tokens,
            sampler=sampler,
            **extra_kwargs,
        )
        elapsed = time.perf_counter() - start

        # Rough token count from response length
        token_count = len(self._tokenizer.encode(response))
        tps = token_count / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            text=response,
            tokens_per_second=tps,
            total_tokens=token_count,
        )

    async def stream(self, conversation: Conversation) -> AsyncIterator[str]:
        """Stream generation token by token. Respects cancel signal."""
        if not self._loaded:
            await self.load_model()

        async for chunk in self.stream_from_request(self.build_inference_request(conversation)):
            yield chunk

    async def stream_from_request(self, request: dict[str, Any]) -> AsyncIterator[str]:
        """Stream generation from a prebuilt prompt payload."""
        if not self._loaded:
            await self.load_model()

        if request.get("mode") != "mlx_prompt":
            raise ValueError(f"Unsupported inference mode: {request.get('mode')}")

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        cancel_flag = self._cancel

        def _stream_sync() -> None:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            prompt = request["prompt"]
            sampler = make_sampler(
                temp=self.config.model.temperature,
                top_p=self.config.model.top_p,
            )
            extra_kwargs: dict[str, Any] = {}
            tq_cache = self._make_turboquant_cache()
            if tq_cache is not None:
                extra_kwargs["prompt_cache"] = tq_cache

            for chunk in stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.config.model.max_tokens,
                sampler=sampler,
                **extra_kwargs,
            ):
                if cancel_flag.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.get_event_loop().run_in_executor(self._mlx_executor, _stream_sync)

        while True:
            if cancel_flag.is_set():
                # Drain any remaining chunks
                while not queue.empty():
                    queue.get_nowait()
                break
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    async def step(self, conversation: Conversation) -> Message:
        """Run one full agent step: generate → maybe call tools → return response.

        If the model emits tool calls, we execute them, inject results as
        TOOL messages, and re-generate — up to MAX_TOOL_ITERATIONS times.
        """
        total_tokens = 0
        last_tps = 0.0
        loop_fingerprints: list[str] = []
        stuck_call_name: str | None = None
        autonomy_nudges = 0
        goal_nudges = 0
        tool_trace: list[dict[str, Any]] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            result = await self.generate(conversation)
            total_tokens += result.total_tokens
            last_tps = result.tokens_per_second

            tool_calls, remaining_text = parse_tool_calls(result.text)

            if not tool_calls:
                if (
                    autonomy_nudges < MAX_AUTONOMY_NUDGES
                    and looks_like_unfulfilled_intent(result.text)
                ):
                    autonomy_nudges += 1
                    log.info("autonomy nudge %d: model narrated without acting",
                             autonomy_nudges)
                    conversation.add(Role.ASSISTANT, result.text)
                    conversation.add(Role.USER, AUTONOMY_NUDGE)
                    continue
                goal_nudge = looks_like_goal_incomplete(result.text, tool_trace)
                if goal_nudges < MAX_GOAL_NUDGES and goal_nudge:
                    goal_nudges += 1
                    log.info("goal-completion nudge %d: %s", goal_nudges,
                             "unaddressed errors" if goal_nudge is TOOL_ERROR_NUDGE
                             else "premature question")
                    conversation.add(Role.ASSISTANT, result.text)
                    conversation.add(Role.USER, goal_nudge)
                    continue
                text = result.text
                metadata: dict[str, Any] = {"tps": last_tps, "tokens": total_tokens}
                if not text.strip():
                    text = EMPTY_TEXT_FALLBACK
                    metadata["empty_text_fallback"] = True
                return Message(
                    role=Role.ASSISTANT,
                    content=text,
                    metadata=metadata,
                )

            # Add the assistant's message (with tool calls stripped) to conversation
            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            # Loop-detection: same fingerprint TOOL_LOOP_REPEAT_LIMIT
            # times in a row = stuck, break out before burning the rest
            # of MAX_TOOL_ITERATIONS.
            if _check_tool_loop(
                loop_fingerprints, _tool_call_fingerprint(tool_calls)
            ):
                log.warning(
                    "Local agent tool-loop detected (%r repeated %d times)",
                    tool_calls[0].name, TOOL_LOOP_REPEAT_LIMIT,
                )
                stuck_call_name = tool_calls[0].name
                break

            # Execute each tool call and add results
            for tc in tool_calls:
                log.info(f"Tool call: {tc.name}({tc.arguments})")
                try:
                    tool_result = await self.skills.execute_tool(tc.name, tc.arguments)
                    result_str = (
                        str(tool_result) if not isinstance(tool_result, str) else tool_result
                    )
                    is_error = tool_result_is_error(result_str)
                except Exception as e:
                    result_str = f"Error executing {tc.name}: {e}"
                    is_error = True
                    log.error(result_str)

                # Conservative tool-result auto-capture: only fires
                # on the explicit-remember pattern, only on success.
                if (
                    not is_error and self.memory
                    and getattr(self.config, "auto_capture", True)
                ):
                    try:
                        from towel.memory.auto_capture import apply_tool_result
                        apply_tool_result(tc.name, result_str, self.memory)
                    except Exception as exc:
                        log.debug("Tool-result capture skipped: %s", exc)

                conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )
                tool_trace.append({
                    "tool": tc.name,
                    "status": "error" if is_error else "ok",
                })

        # Either hit max iterations OR loop detection broke us out.
        if stuck_call_name is not None:
            stuck_msg = (
                f"I got stuck calling {stuck_call_name!r} repeatedly. "
                "Stopping to avoid burning more time on this turn."
            )
            return Message(
                role=Role.ASSISTANT,
                content=(remaining_text + "\n\n" + stuck_msg) if remaining_text else stuck_msg,
                metadata={
                    "tps": last_tps,
                    "tokens": total_tokens,
                    "loop_detected": True,
                },
            )
        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        return Message(
            role=Role.ASSISTANT,
            content=remaining_text or "I've reached my tool execution limit for this turn.",
            metadata={"tps": last_tps, "tokens": total_tokens, "max_iterations": True},
        )

    async def step_streaming(self, conversation: Conversation) -> AsyncIterator[AgentEvent]:
        """Run a full agent step, yielding events as they happen.

        Streams tokens during generation, emits tool call/result events,
        and ends with a response_complete or cancelled event.
        """
        # Reset cancel flag for this generation
        self._cancel.clear()
        total_tokens = 0
        loop_fingerprints: list[str] = []
        stuck_call_name: str | None = None
        autonomy_nudges = 0
        goal_nudges = 0
        tool_trace: list[dict[str, Any]] = []

        for iteration in range(MAX_TOOL_ITERATIONS):
            # Stream tokens and accumulate the full response
            full_text = ""
            start = time.perf_counter()
            async for chunk in self.stream(conversation):
                full_text += chunk
                yield AgentEvent.token(chunk)
            elapsed = time.perf_counter() - start

            # Check if generation was cancelled
            if self._cancel.is_set():
                if full_text.strip():
                    conversation.add(Role.ASSISTANT, full_text)
                yield AgentEvent.cancelled(
                    full_text,
                    metadata={"tokens": total_tokens, "reason": "user_cancelled"},
                )
                self._cancel.clear()
                return

            # Estimate token count from accumulated text
            if self._tokenizer:
                token_count = len(self._tokenizer.encode(full_text))
            else:
                token_count = len(full_text.split())
            total_tokens += token_count
            tps = token_count / elapsed if elapsed > 0 else 0.0

            # Check for tool calls
            tool_calls, remaining_text = parse_tool_calls(full_text)

            if not tool_calls:
                if (
                    autonomy_nudges < MAX_AUTONOMY_NUDGES
                    and looks_like_unfulfilled_intent(full_text)
                ):
                    autonomy_nudges += 1
                    log.info("autonomy nudge %d: model narrated without acting",
                             autonomy_nudges)
                    conversation.add(Role.ASSISTANT, full_text)
                    conversation.add(Role.USER, AUTONOMY_NUDGE)
                    continue
                goal_nudge = looks_like_goal_incomplete(full_text, tool_trace)
                if goal_nudges < MAX_GOAL_NUDGES and goal_nudge:
                    goal_nudges += 1
                    log.info("goal-completion nudge %d: %s", goal_nudges,
                             "unaddressed errors" if goal_nudge is TOOL_ERROR_NUDGE
                             else "premature question")
                    conversation.add(Role.ASSISTANT, full_text)
                    conversation.add(Role.USER, goal_nudge)
                    continue
                text = full_text
                metadata: dict[str, Any] = {"tps": tps, "tokens": total_tokens}
                if not text.strip():
                    text = EMPTY_TEXT_FALLBACK
                    metadata["empty_text_fallback"] = True
                conversation.add(Role.ASSISTANT, text)
                yield AgentEvent.complete(
                    text,
                    metadata=metadata,
                )
                return

            # Tool call loop
            if remaining_text:
                conversation.add(Role.ASSISTANT, remaining_text)

            # Loop-detection (see step()).
            if _check_tool_loop(
                loop_fingerprints, _tool_call_fingerprint(tool_calls)
            ):
                log.warning(
                    "Local agent (streaming) tool-loop detected (%r repeated %d times)",
                    tool_calls[0].name, TOOL_LOOP_REPEAT_LIMIT,
                )
                stuck_call_name = tool_calls[0].name
                break

            for tc in tool_calls:
                if self._cancel.is_set():
                    yield AgentEvent.cancelled(
                        remaining_text or "",
                        metadata={"tokens": total_tokens, "reason": "user_cancelled"},
                    )
                    self._cancel.clear()
                    return

                log.info(f"Tool call: {tc.name}({tc.arguments})")
                yield AgentEvent.tool_call(tc.name, tc.arguments)

                try:
                    tool_result = await self.skills.execute_tool(tc.name, tc.arguments)
                    result_str = (
                        str(tool_result) if not isinstance(tool_result, str) else tool_result
                    )
                    is_error = tool_result_is_error(result_str)
                except Exception as e:
                    result_str = f"Error executing {tc.name}: {e}"
                    is_error = True
                    log.error(result_str)

                if (
                    not is_error and self.memory
                    and getattr(self.config, "auto_capture", True)
                ):
                    try:
                        from towel.memory.auto_capture import apply_tool_result
                        apply_tool_result(tc.name, result_str, self.memory)
                    except Exception as exc:
                        log.debug("Tool-result capture skipped: %s", exc)

                yield AgentEvent.tool_result(tc.name, result_str)
                conversation.add(
                    Role.TOOL,
                    format_tool_feedback(tc.name, result_str, is_error),
                    tool_name=tc.name,
                    status="error" if is_error else "ok",
                )
                tool_trace.append({
                    "tool": tc.name,
                    "status": "error" if is_error else "ok",
                })

        # Either hit max iterations OR loop detection broke us out.
        # Persist the terminal message into the conversation so callers
        # that just forward the event stream (e.g. the WS handler and
        # the OpenAI-compat SSE path) still leave the same text on
        # disk that the live client received. Same asymmetry that bit
        # _stream_remote_inference in 803d1b4: the non-streaming step()
        # returns a Message for the caller to append; the streaming
        # variant has no return value, so it must mutate the
        # conversation here or the assistant turn disappears from
        # replay.
        if stuck_call_name is not None:
            stuck_msg = (
                f"I got stuck calling {stuck_call_name!r} repeatedly. "
                "Stopping to avoid burning more time on this turn."
            )
            # remaining_text was already appended earlier in this
            # iteration's tool-call branch (line "if remaining_text"),
            # so only stuck_msg needs adding here.
            conversation.add(Role.ASSISTANT, stuck_msg)
            yield AgentEvent.complete(
                (remaining_text + "\n\n" + stuck_msg) if remaining_text else stuck_msg,
                metadata={
                    "tps": 0,
                    "tokens": total_tokens,
                    "loop_detected": True,
                },
            )
            return
        log.warning(f"Hit max tool iterations ({MAX_TOOL_ITERATIONS})")
        max_iter_msg = "I've reached my tool execution limit for this turn."
        if not remaining_text:
            conversation.add(Role.ASSISTANT, max_iter_msg)
        yield AgentEvent.complete(
            remaining_text or max_iter_msg,
            metadata={"tps": 0, "tokens": total_tokens, "max_iterations": True},
        )

    def _run_capture_hooks(self, query: str | None) -> None:
        """Thin shim around towel.agent.capture.run_capture_hooks.

        Kept as an instance method so tests can patch it per-instance,
        and so subclasses could override capture behavior without
        intercepting the entire step() body.
        """
        from towel.agent.capture import run_capture_hooks

        run_capture_hooks(
            query, memory=self.memory, config=self.config, runtime=self,
        )

    def _build_system_content(
        self,
        include_tools_section: bool = True,
        query: str | None = None,
        tools_available: bool = True,
    ) -> str:
        """Build the system prompt including project context, memory, and tool definitions.

        When ``include_tools_section`` is False, the per-tool listing and call-format
        spec are omitted — used when the tokenizer's chat template natively renders
        the tool list via the ``tools=`` kwarg (e.g. Qwen3, Llama 3.1+, Hermes).
        Behavioral guardrails (no inventing tools, terse calls, single retry) remain.

        ``query`` is the current user turn; when set, the memory block is
        ranked by relevance and trimmed to the top few rather than
        dumping the entire memory corpus. ``None`` keeps the legacy
        full-dump behavior for non-conversation callers.
        """
        system = self.config.identity + (
            "\n\nAfter using a tool, always answer the user's original question "
            "based on the tool result. Do not just acknowledge the tool output — "
            "use it to provide a direct, helpful answer. If you changed something "
            "or verified something, explicitly report that back to the user."
        )

        # Inject project context — use coordinator-provided override if set,
        # otherwise discover from local .towel.md files
        if self.project_context:
            system += self.project_context
        else:
            from towel.agent.project import load_project_context

            project_block = load_project_context()
            if project_block:
                system += project_block

        # Inject persistent memories — ranked by relevance to the
        # current user turn when one is available, dumped in full
        # otherwise.
        if self.memory:
            memory_block = self.memory.to_prompt_block(query=query)
            if memory_block:
                system += memory_block
        tools = self.skills.tool_definitions() if tools_available else []
        if tools:
            if include_tools_section:
                # Compact format: name + description only. Full parameter schemas
                # bloat the prompt (~330 tools) and slow inference significantly.
                tool_lines = []
                for t in tools:
                    params = t.get("parameters", {})
                    props = params.get("properties", {})
                    if props:
                        param_names = ", ".join(props.keys())
                        tool_lines.append(f"- {t['name']}({param_names}): {t['description']}")
                    else:
                        tool_lines.append(f"- {t['name']}(): {t['description']}")

                tool_names = [t["name"] for t in tools]
                tool_name_list = ", ".join(tool_names)

                system += (
                    "\n\n# Tools\n\n"
                    "You may call one or more functions to assist with the user query.\n\n"
                    "Available tools:\n" + "\n".join(tool_lines) + "\n\n"
                    "For each function call, return a json object with function name and "
                    "arguments within <tool_call></tool_call> XML tags:\n"
                    "<tool_call>\n"
                    '{"name": <function-name>, "arguments": <args-json-object>}\n'
                    "</tool_call>\n\n"
                    f"The ONLY supported tool names are: {tool_name_list}\n\n"
                    "IMPORTANT:\n"
                    "- Only call functions from the list above. Do NOT invent or guess "
                    "function names. If a tool you want is not listed, it does not exist.\n"
                    "- Always use the exact <tool_call> format shown above.\n"
                    "- When using a tool, prefer emitting just the tool call instead of "
                    "narrating that you are about to check something.\n"
                    "- After tool results arrive, either give the concrete answer or make "
                    "one corrected retry. Do not repeat vague status updates.\n"
                    "- If no tool is needed, respond directly without tool calls."
                )
            else:
                # Tools are rendered by the chat template via tools= kwarg.
                # Keep only behavioral guidance here, under a distinct header
                # so it doesn't collide with the template's own "# Tools" block.
                system += (
                    "\n\n# Tool-use rules\n\n"
                    "The available tools are listed under the chat template's "
                    "Tools section below. Use them when they help answer the user.\n\n"
                    "IMPORTANT:\n"
                    "- Only call tools from the provided list. Do NOT invent or guess "
                    "tool names. If a tool you want is not listed, it does not exist.\n"
                    "- When using a tool, prefer emitting just the tool call instead of "
                    "narrating that you are about to check something.\n"
                    "- After tool results arrive, either give the concrete answer or make "
                    "one corrected retry. Do not repeat vague status updates.\n"
                    "- If no tool is needed, respond directly without tool calls."
                )
        return system

    def _tools_for_chat_template(self) -> list[dict[str, Any]]:
        """OpenAI-function dicts for ``apply_chat_template(tools=...)``."""
        return tools_as_openai_functions(self.skills.tool_definitions())

    def _detect_native_tools_support(self) -> bool:
        """Probe whether the loaded tokenizer's chat template consumes the ``tools=`` kwarg.

        Modern templates (Qwen3, Llama 3.1+, Hermes) render the tools list themselves
        and emit model-native call markers; older templates silently ignore the kwarg.
        We detect by comparing template output with and without a probe tool — if the
        probe name appears, the template is rendering tools.
        """
        tokenizer = self._tokenizer
        if not tokenizer or not hasattr(tokenizer, "apply_chat_template"):
            return False
        probe = [
            {
                "type": "function",
                "function": {
                    "name": "__towel_probe__",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        sample = [{"role": "user", "content": "hi"}]
        try:
            with_tools = tokenizer.apply_chat_template(
                sample, tools=probe, tokenize=False, add_generation_prompt=False
            )
        except Exception as exc:
            log.debug("apply_chat_template(tools=...) raised: %s", exc)
            return False
        return "__towel_probe__" in (with_tools or "")

    def _token_count(self, text: str) -> int:
        """Count tokens using the loaded tokenizer, or estimate."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return max(1, len(text) // 4)

    def _build_prompt(self, conversation: Conversation) -> str:
        """Build a chat prompt string from the conversation history.

        Uses the context window manager to fit messages within the
        model's token budget, dropping oldest messages first.
        """
        from towel.nodes.roles import classify_task_type, task_needs_tools

        use_native_tools = bool(self._native_tools_supported)
        query = conversation.latest_user_query()
        self._run_capture_hooks(query)

        task_type = classify_task_type(query)
        wants_tools = task_needs_tools(task_type)

        system_content = self._build_system_content(
            include_tools_section=not use_native_tools,
            query=query,
            tools_available=wants_tools,
        )
        all_messages = conversation.to_chat_messages()
        output_reserve = estimate_output_reserve(
            all_messages,
            configured_max_tokens=self.config.model.max_tokens,
            token_counter=self._token_count,
        )
        effective_context_window = self.config.model.context_window
        if getattr(self.config.model, "auto_context", True):
            effective_context_window = select_context_window(
                system_content,
                all_messages,
                configured_context_window=self.config.model.context_window,
                min_context_window=getattr(self.config.model, "min_context_window", 32768),
                max_output_tokens=output_reserve,
                token_counter=self._token_count,
            )
        maybe_compact_conversation(
            conversation,
            system_content=system_content,
            context_window=effective_context_window,
            max_output_tokens=output_reserve,
            token_counter=self._token_count,
        )
        all_messages = conversation.to_chat_messages()

        # Collect indices of pinned messages so they survive context eviction
        pinned_indices = {i for i, msg in enumerate(conversation.messages) if msg.pinned}

        fitted_messages, budget = fit_messages(
            system_content=system_content,
            messages=all_messages,
            context_window=effective_context_window,
            max_output_tokens=output_reserve,
            token_counter=self._token_count,
            pinned_indices=pinned_indices if pinned_indices else None,
        )

        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "system", "content": system_content}]
            # Convert tool messages to the format the chat template expects.
            # Qwen3 expects tool role messages with content in the standard
            # chat template — the tokenizer handles wrapping them.
            for msg in fitted_messages:
                if msg["role"] == "tool":
                    # Strip the [tool_name] prefix and pass as tool role
                    content = msg["content"]
                    # Extract tool name from "[tool_name] ..." format
                    if content.startswith("[") and "]" in content:
                        _name, _, result = content.partition("]")
                        tool_name = _name.lstrip("[")
                        messages.append(
                            {
                                "role": "tool",
                                "content": result.lstrip(),
                                "name": tool_name,
                            }
                        )
                    else:
                        messages.append(msg)
                else:
                    messages.append(msg)
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
                # Qwen3-thinking variants render a long `<|channel>thought\n …`
                # chain-of-thought block before the visible answer when this
                # kwarg defaults to True (the model card default). With small
                # max_tokens caps the visible response is just the channel
                # marker; with large caps the model wedges in the reasoning
                # block without producing a final answer in time. Most chat
                # templates ignore unknown kwargs, so explicitly opting OUT
                # here is safe across models that don't support thinking.
                # Callers that genuinely want reasoning output can flip
                # `enable_thinking` back to True at the runtime level — they
                # know they want it; the default user-facing chat path does
                # not.
                "enable_thinking": False,
            }
            if use_native_tools and wants_tools:
                native_tools = self._tools_for_chat_template()
                if native_tools:
                    template_kwargs["tools"] = native_tools
            try:
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as exc:
                # Native tools may not actually be supported despite our probe.
                # Disable for the rest of this process and fall through to the
                # tool-role fallback (which uses text-injected tool listings).
                if use_native_tools:
                    log.warning(
                        "Native tools channel failed at render time (%s); "
                        "falling back to text-injected tool list.",
                        exc,
                    )
                    self._native_tools_supported = False
                    return self._build_prompt(conversation)
                # Fallback if template doesn't support tool role
                fallback_messages = [{"role": "system", "content": system_content}]
                for msg in fitted_messages:
                    if msg["role"] == "tool":
                        fallback_messages.append(
                            {
                                "role": "user",
                                "content": f"<tool_response>\n{msg['content']}\n</tool_response>",
                            }
                        )
                    else:
                        fallback_messages.append(msg)
                return self._tokenizer.apply_chat_template(
                    fallback_messages, tokenize=False, add_generation_prompt=True
                )

        # Fallback: simple concatenation
        parts = [f"System: {system_content}\n"]
        for msg in fitted_messages:
            parts.append(f"{msg['role'].capitalize()}: {msg['content']}\n")
        parts.append("Assistant: ")
        return "".join(parts)

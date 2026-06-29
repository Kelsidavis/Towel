"""Tool call parser — extracts structured tool calls from model output.

Supports multiple formats that LLMs commonly emit:
  1. JSON block:  ```json\n{"tool": "name", "arguments": {...}}\n```
  2. XML-style:   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
  3. Function-style: <|tool_call|>name(arg1="val1", arg2="val2")<|/tool_call|>
  4. Bare JSON object with "tool"/"name" + "arguments"/"parameters" keys
  5. Qwen Hermes-style: ✿FUNCTION✿name\n✿ARGS✿{...}\n✿RESULT✿ (or end of text)
  6. Qwen ChatML tool_calls array inside structured output

The parser is intentionally lenient — models are messy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    """A parsed tool invocation."""

    name: str
    arguments: dict[str, Any]
    raw: str  # the original matched text

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


# Patterns ordered from most specific to least
_PATTERNS = [
    # ```json ... ``` blocks containing tool calls
    re.compile(
        r"```(?:json)?\s*(\{[^`]*?\"(?:tool|name)\"\s*:.*?\})\s*```",
        re.DOTALL,
    ),
    # <tool_call>...</tool_call> XML-style
    re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        re.DOTALL,
    ),
    # <|tool_call|>...<|/tool_call|> special token style
    re.compile(
        r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>",
        re.DOTALL,
    ),
    # Bare JSON object on its own line with tool/name key
    re.compile(
        r"^(\{[^\n]*?\"(?:tool|name)\"\s*:[^\n]*?\})$",
        re.MULTILINE,
    ),
]

# For function-call style: <|tool_call|>func_name(args)<|/tool_call|>
_FUNC_CALL_PATTERN = re.compile(
    r"<\|tool_call\|>\s*(\w+)\(([^)]*)\)\s*<\|/tool_call\|>",
    re.DOTALL,
)

# Lenient special-token style emitted by some small/abliterated models, e.g.
#   <|tool_call>call:shell
#   echo hello
# The closing pipe before ``>`` is often dropped, the ``call:`` prefix is
# optional, and the closing ``<|/tool_call|>`` may be missing entirely. Args
# follow as a paren group, a JSON object, or a raw trailing line (terminated by
# the next tool-call token or end of text). Bare identifiers only — blocks that
# open with ``{`` are JSON envelopes handled by _PATTERNS, so the leading
# ``[A-Za-z_]`` name class deliberately won't match them.
_SPECIAL_TOKEN_CALL_PATTERN = re.compile(
    r"<\|tool_call\|?>\s*"               # opening token, closing pipe optional
    r"(?:call:)?\s*"                     # optional `call:` prefix
    r"([A-Za-z_]\w*)"                    # tool name
    r"(?:\(([^)]*)\)|[:\n]\s*(.*?))?"    # (paren args) | : / newline raw args
    r"\s*(?=<\|/?tool_call\|?>|<\||$)",  # stop at next token or end of text
    re.DOTALL,
)

# Qwen Hermes-style: ✿FUNCTION✿name\n✿ARGS✿{...} (terminated by ✿RESULT✿ or end)
_QWEN_HERMES_PATTERN = re.compile(
    r"✿FUNCTION✿\s*(\w+)\s*\n✿ARGS✿\s*(\{.*?\})\s*(?:✿RESULT✿|✿|$)",
    re.DOTALL,
)

# Qwen ChatML tool_calls array: "tool_calls": [{"function": {"name": ..., "arguments": ...}}]
_QWEN_CHATML_TOOL_CALLS = re.compile(
    r'"tool_calls"\s*:\s*(\[.*?\])',
    re.DOTALL,
)


def parse_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Parse tool calls from model output text.

    Returns:
        A tuple of (tool_calls, remaining_text) where remaining_text
        is the model output with tool call blocks stripped out.

    Defensive: a buggy backend can pass None even though the type
    says str. Coerce to "" at the boundary so callers don't have to
    sprinkle their own isinstance guards. (Catches the same class
    of crash _synthesize_ensemble had to guard against externally.)
    """
    if not isinstance(text, str):
        text = ""
    calls: list[ToolCall] = []
    remaining = text

    # Try Qwen Hermes-style first (✿FUNCTION✿ / ✿ARGS✿)
    for match in _QWEN_HERMES_PATTERN.finditer(text):
        func_name = match.group(1)
        raw_json = match.group(2)
        try:
            args = json.loads(raw_json)
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(name=func_name, arguments=args, raw=match.group(0)))
            remaining = remaining.replace(match.group(0), "")
        except (json.JSONDecodeError, TypeError):
            continue

    if calls:
        return calls, remaining.strip()

    # Try Qwen ChatML tool_calls array style
    chatml_match = _QWEN_CHATML_TOOL_CALLS.search(text)
    if chatml_match:
        try:
            tool_calls_arr = json.loads(chatml_match.group(1))
            if isinstance(tool_calls_arr, list):
                for tc in tool_calls_arr:
                    parsed = _normalize_chatml_tool_call(tc)
                    if parsed:
                        calls.append(
                            ToolCall(
                                name=parsed["name"],
                                arguments=parsed["arguments"],
                                raw=chatml_match.group(0),
                            )
                        )
                if calls:
                    remaining = remaining.replace(chatml_match.group(0), "")
                    return calls, remaining.strip()
        except (json.JSONDecodeError, TypeError):
            pass

    # Try function-call style
    for match in _FUNC_CALL_PATTERN.finditer(text):
        func_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = _parse_func_args(args_str)
            calls.append(ToolCall(name=func_name, arguments=args, raw=match.group(0)))
            remaining = remaining.replace(match.group(0), "")
        except (ValueError, SyntaxError):
            continue

    if calls:
        return calls, remaining.strip()

    # Try the lenient special-token style (<|tool_call>call:name\nargs)
    for match in _SPECIAL_TOKEN_CALL_PATTERN.finditer(text):
        name = match.group(1)
        paren_args = match.group(2)
        raw_args = match.group(3)
        if paren_args is not None:
            try:
                args = _parse_func_args(paren_args.strip())
            except (ValueError, SyntaxError):
                args = {}
        elif raw_args is not None and raw_args.strip():
            body = raw_args.strip()
            # The trailing text may itself be a JSON object of arguments;
            # otherwise fall back to the same {"input": ...} envelope the
            # JSON normalizer uses for unkeyed string arguments.
            try:
                parsed_body = json.loads(body)
                args = parsed_body if isinstance(parsed_body, dict) else {"input": body}
            except (json.JSONDecodeError, TypeError):
                args = {"input": body}
        else:
            args = {}
        calls.append(ToolCall(name=name, arguments=args, raw=match.group(0)))
        remaining = remaining.replace(match.group(0), "")

    if calls:
        return calls, remaining.strip()

    # Try JSON-based patterns
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            raw_json = match.group(1)
            try:
                parsed = _normalize_tool_json(json.loads(raw_json))
                if parsed:
                    calls.append(
                        ToolCall(
                            name=parsed["name"],
                            arguments=parsed["arguments"],
                            raw=match.group(0),
                        )
                    )
                    remaining = remaining.replace(match.group(0), "")
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        if calls:
            break  # Use the first pattern that matched

    return calls, remaining.strip()


def _normalize_tool_json(obj: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize various JSON tool call formats to {name, arguments}."""
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not name or not isinstance(name, str):
        return None

    arguments = (
        obj.get("arguments") or obj.get("parameters") or obj.get("params") or obj.get("args") or {}
    )
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}

    return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}


def _normalize_chatml_tool_call(tc: Any) -> dict[str, Any] | None:
    """Normalize a Qwen ChatML tool_calls array entry.

    Handles both:
      {"function": {"name": "...", "arguments": "..."}}
      {"name": "...", "arguments": {...}}
    """
    if not isinstance(tc, dict):
        return None

    # Qwen ChatML nests under "function" key
    func = tc.get("function", tc)
    if not isinstance(func, dict):
        return None

    name = func.get("name")
    if not name or not isinstance(name, str):
        return None

    arguments = func.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"input": arguments}

    return {"name": name, "arguments": arguments if isinstance(arguments, dict) else {}}


def _parse_func_args(args_str: str) -> dict[str, Any]:
    """Parse function-style arguments like: arg1="val1", arg2=42"""
    if not args_str:
        return {}

    # Try as JSON object first (without braces)
    try:
        return json.loads("{" + args_str + "}")
    except json.JSONDecodeError:
        pass

    # Parse key=value pairs
    result: dict[str, Any] = {}
    for pair in re.split(r",\s*", args_str):
        if "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        # Try to parse as JSON value
        try:
            result[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            result[key] = value

    return result

"""Tests for the tool call parser."""

from towel.agent.tool_parser import ToolCall, parse_tool_calls


def test_json_block_tool_call():
    text = """Here's what I'll do:
```json
{"tool": "read_file", "arguments": {"path": "/tmp/test.txt"}}
```
Let me read that for you."""
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/tmp/test.txt"}
    assert "```" not in remaining
    assert "Let me read that" in remaining


def test_xml_style_tool_call():
    text = '<tool_call>{"name": "run_command", "arguments": {"command": "ls -la"}}</tool_call>'
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments["command"] == "ls -la"


def test_special_token_tool_call():
    text = '<|tool_call|>{"name": "fetch_url", "arguments": {"url": "https://example.com"}}<|/tool_call|>'
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "fetch_url"


def test_function_style_tool_call():
    text = '<|tool_call|>read_file(path="/tmp/test.txt")<|/tool_call|>'
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments["path"] == "/tmp/test.txt"


def test_no_tool_calls():
    text = "Just a regular response with no tool calls."
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 0
    assert remaining == text


def test_multiple_tool_calls():
    text = """I'll check both:
```json
{"tool": "read_file", "arguments": {"path": "a.txt"}}
```
and
```json
{"tool": "read_file", "arguments": {"path": "b.txt"}}
```"""
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].arguments["path"] == "a.txt"
    assert calls[1].arguments["path"] == "b.txt"


def test_parameters_alias():
    text = '```json\n{"name": "run_command", "parameters": {"command": "echo hi"}}\n```'
    calls, _ = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments["command"] == "echo hi"


def test_qwen_hermes_single():
    text = 'I\'ll read that file.\n✿FUNCTION✿read_file\n✿ARGS✿{"path": "/tmp/test.txt"}\n✿RESULT✿'
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/tmp/test.txt"}
    assert "I'll read that file." in remaining
    assert "✿FUNCTION✿" not in remaining


def test_qwen_hermes_multiple():
    text = (
        "Let me check both.\n"
        '✿FUNCTION✿read_file\n✿ARGS✿{"path": "a.txt"}\n✿RESULT✿\n'
        '✿FUNCTION✿read_file\n✿ARGS✿{"path": "b.txt"}\n✿RESULT✿'
    )
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].arguments["path"] == "a.txt"
    assert calls[1].arguments["path"] == "b.txt"


def test_qwen_hermes_no_result_terminator():
    """Qwen sometimes omits ✿RESULT✿ at end of output."""
    text = '✿FUNCTION✿run_command\n✿ARGS✿{"command": "ls"}'
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments["command"] == "ls"


def test_qwen_chatml_tool_calls_array():
    text = (
        '{"tool_calls": [{"function": {"name": "read_file",'
        ' "arguments": "{\\"path\\": \\"/tmp/test.txt\\"}"}}]}'
    )
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/tmp/test.txt"}


def test_qwen_chatml_multiple_tool_calls():
    text = (
        '{"tool_calls": [{"function": {"name": "read_file",'
        ' "arguments": "{\\"path\\": \\"a.txt\\"}"}}, {"function":'
        ' {"name": "run_command",'
        ' "arguments": "{\\"command\\": \\"ls\\"}"}}]}'
    )
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "read_file"
    assert calls[1].name == "run_command"


def test_qwen_chatml_dict_arguments():
    """Arguments as dict instead of JSON string."""
    text = '{"tool_calls": [{"function": {"name": "search", "arguments": {"query": "hello"}}}]}'
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": "hello"}


def test_tool_call_to_dict():
    tc = ToolCall(name="test", arguments={"a": 1}, raw="raw")
    assert tc.to_dict() == {"name": "test", "arguments": {"a": 1}}


def test_special_token_call_dropped_closing_pipe():
    """Small/abliterated models emit <|tool_call>call:name\\nargs — the
    closing pipe before ``>`` and the trailing <|/tool_call|> are dropped,
    and args arrive as a raw line. Previously this leaked verbatim as the
    assistant's text instead of being parsed."""
    calls, remaining = parse_tool_calls("<|tool_call>call:shell\necho hello-from-tool")
    assert len(calls) == 1
    assert calls[0].name == "shell"
    assert calls[0].arguments == {"input": "echo hello-from-tool"}
    assert remaining == ""


def test_special_token_call_with_keyed_body():
    """A small model emitting `call:run_command\\ncommand: <cmd>` must parse the
    `command:` prefix as the argument key — not leave it inside the value (which
    previously ran `command: cd ...` and failed with `command:: not found`)."""
    calls, _ = parse_tool_calls(
        "<|tool_call>call:run_command\ncommand: cd /tmp && ls -la"
    )
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments == {"command": "cd /tmp && ls -la"}


def test_special_token_keyed_body_ignores_url_colon():
    """A colon with no following space (a URL) is NOT a key:value split."""
    calls, _ = parse_tool_calls("<|tool_call>call:fetch_url\nhttps://example.com/x")
    assert len(calls) == 1
    assert calls[0].arguments == {"input": "https://example.com/x"}


def test_special_token_call_with_json_args():
    """The raw trailing body is parsed as JSON arguments when it is a JSON
    object, not wrapped in the {"input": ...} fallback."""
    calls, _ = parse_tool_calls(
        '<|tool_call|>call:write_file\n{"path": "a.txt", "content": "hi"}'
    )
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments == {"path": "a.txt", "content": "hi"}


def test_special_token_json_envelope_still_uses_json_path():
    """A <|tool_call|>{...}<|/tool_call|> JSON envelope must keep routing
    through the JSON normalizer (name/arguments extracted) and must NOT be
    swept into the lenient {"input": ...} fallback."""
    calls, _ = parse_tool_calls(
        '<|tool_call|>{"name": "read_file", "arguments": {"path": "x"}}<|/tool_call|>'
    )
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "x"}


def test_special_token_pattern_no_false_positive_on_prose():
    """Plain prose without any tool-call token yields no calls."""
    calls, remaining = parse_tool_calls("Sure, I can help — no tools needed here.")
    assert calls == []
    assert remaining == "Sure, I can help — no tools needed here."


def test_parse_tool_calls_handles_none_defensively():
    """A backend that returns GenerationResult(text=None) used to
    crash callers inside re.finditer. The parser now coerces non-
    str input to "" so every caller naturally falls through to the
    "no tool calls, empty remaining" path."""
    calls, remaining = parse_tool_calls(None)  # type: ignore[arg-type]
    assert calls == []
    assert remaining == ""


def test_parse_tool_calls_handles_non_string_defensively():
    """Same guard applies to any non-string input, not just None."""
    for bad in (42, [], {}, object()):
        calls, remaining = parse_tool_calls(bad)  # type: ignore[arg-type]
        assert calls == []
        assert remaining == ""

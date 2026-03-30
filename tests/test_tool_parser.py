"""Tests for the tool call parser."""

from towel.agent.tool_parser import parse_tool_calls, ToolCall


def test_json_block_tool_call():
    text = '''Here's what I'll do:
```json
{"tool": "read_file", "arguments": {"path": "/tmp/test.txt"}}
```
Let me read that for you.'''
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
    text = '''I'll check both:
```json
{"tool": "read_file", "arguments": {"path": "a.txt"}}
```
and
```json
{"tool": "read_file", "arguments": {"path": "b.txt"}}
```'''
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
    text = "I'll read that file.\n✿FUNCTION✿read_file\n✿ARGS✿{\"path\": \"/tmp/test.txt\"}\n✿RESULT✿"
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/tmp/test.txt"}
    assert "I'll read that file." in remaining
    assert "✿FUNCTION✿" not in remaining


def test_qwen_hermes_multiple():
    text = (
        "Let me check both.\n"
        "✿FUNCTION✿read_file\n✿ARGS✿{\"path\": \"a.txt\"}\n✿RESULT✿\n"
        "✿FUNCTION✿read_file\n✿ARGS✿{\"path\": \"b.txt\"}\n✿RESULT✿"
    )
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].arguments["path"] == "a.txt"
    assert calls[1].arguments["path"] == "b.txt"


def test_qwen_hermes_no_result_terminator():
    """Qwen sometimes omits ✿RESULT✿ at end of output."""
    text = "✿FUNCTION✿run_command\n✿ARGS✿{\"command\": \"ls\"}"
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "run_command"
    assert calls[0].arguments["command"] == "ls"


def test_qwen_chatml_tool_calls_array():
    text = '{"tool_calls": [{"function": {"name": "read_file", "arguments": "{\\\"path\\\": \\\"/tmp/test.txt\\\"}"}}]}'
    calls, remaining = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "/tmp/test.txt"}


def test_qwen_chatml_multiple_tool_calls():
    text = '{"tool_calls": [{"function": {"name": "read_file", "arguments": "{\\\"path\\\": \\\"a.txt\\\"}"}}, {"function": {"name": "run_command", "arguments": "{\\\"command\\\": \\\"ls\\\"}"}}]}'
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

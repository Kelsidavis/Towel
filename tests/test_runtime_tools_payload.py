"""Cross-runtime tests for the native tools channel.

Verifies that the Ollama, llama-server, and Claude runtimes:
  - render their system prompt without the inline tool listing when native
    tools are supported (the chat template / API renders the tools instead);
  - convert each backend's structured tool-call format back into ``ToolCall``
    objects the agent step-loop can consume;
  - include the tools list in ``build_inference_request`` when native tools are
    supported, and omit it otherwise.
"""

from __future__ import annotations

from typing import Any

from towel.agent.claude_runtime import ClaudeCodeRuntime, _extract_anthropic_tool_calls
from towel.agent.llama_runtime import LlamaRuntime, _normalize_openai_tool_calls
from towel.agent.ollama_runtime import OllamaRuntime, _normalize_ollama_tool_calls
from towel.agent.tools_payload import tools_as_anthropic, tools_as_openai_functions
from towel.config import TowelConfig
from towel.skills.base import Skill, ToolDefinition
from towel.skills.registry import SkillRegistry


class _ReadFileSkill(Skill):
    @property
    def name(self) -> str:
        return "fs"

    @property
    def description(self) -> str:
        return "fs"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="read_file",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]

    async def execute(self, tool_name: str, arguments: dict):
        return "ok"


def _skills_with_one_tool() -> SkillRegistry:
    skills = SkillRegistry()
    skills.register(_ReadFileSkill())
    return skills


# --------------------------------------------------------------------------- #
# Shared payload helpers                                                      #
# --------------------------------------------------------------------------- #


class TestToolPayloadHelpers:
    def test_openai_functions_format(self):
        out = tools_as_openai_functions(_skills_with_one_tool().tool_definitions())
        assert out == [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

    def test_anthropic_format_uses_input_schema(self):
        out = tools_as_anthropic(_skills_with_one_tool().tool_definitions())
        assert out == [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]

    def test_empty_parameters_default_to_empty_object_schema(self):
        out = tools_as_openai_functions([{"name": "ping", "description": "ping"}])
        assert out[0]["function"]["parameters"] == {"type": "object", "properties": {}}


# --------------------------------------------------------------------------- #
# Ollama                                                                      #
# --------------------------------------------------------------------------- #


class TestOllamaNativeTools:
    def _runtime(self) -> OllamaRuntime:
        config = TowelConfig(identity="You are Towel.")
        return OllamaRuntime(config, skills=_skills_with_one_tool())

    def test_system_prompt_drops_inline_listing_when_native(self):
        rt = self._runtime()
        native = rt._build_system_prompt(include_tools_section=False)
        text_mode = rt._build_system_prompt(include_tools_section=True)
        assert "- read_file" not in native
        assert "<tool_call>" not in native
        assert "- read_file" in text_mode
        assert "<tool_call>" in text_mode

    def test_build_inference_request_includes_tools_when_native(self):
        rt = self._runtime()
        rt._native_tools_supported = True
        req = rt.build_inference_request(_empty_conversation())
        assert "tools" in req
        assert req["tools"][0]["function"]["name"] == "read_file"

    def test_build_inference_request_omits_tools_when_text_only(self):
        rt = self._runtime()
        rt._native_tools_supported = False
        req = rt.build_inference_request(_empty_conversation())
        assert "tools" not in req

    def test_normalizer_handles_dict_arguments(self):
        calls = _normalize_ollama_tool_calls(
            [{"id": "x", "function": {"name": "read_file", "arguments": {"path": "/etc/hosts"}}}]
        )
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "/etc/hosts"}

    def test_normalizer_handles_string_arguments(self):
        # Defensive: some Ollama builds return arguments as a JSON-encoded string.
        calls = _normalize_ollama_tool_calls(
            [{"id": "x", "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'}}]
        )
        assert calls[0].arguments == {"path": "/tmp/x"}

    def test_normalizer_skips_entries_without_name(self):
        assert _normalize_ollama_tool_calls([{"id": "x", "function": {}}]) == []


# --------------------------------------------------------------------------- #
# llama-server                                                                #
# --------------------------------------------------------------------------- #


class TestLlamaNativeTools:
    def _runtime(self) -> LlamaRuntime:
        config = TowelConfig(identity="You are Towel.")
        return LlamaRuntime(config, skills=_skills_with_one_tool(), auto_start=False)

    def test_system_prompt_drops_inline_listing_when_native(self):
        rt = self._runtime()
        native = rt._build_system_prompt(include_tools_section=False)
        assert "- read_file" not in native
        assert "<tool_call>" not in native
        assert "Only call tools from the provided list" in native

    def test_build_inference_request_includes_tools_by_default(self):
        # llama-server always sends tools=[...] — newer versions render them via
        # the chat template, older versions ignore them harmlessly.
        rt = self._runtime()
        assert rt._native_tools_supported is True
        req = rt.build_inference_request(_empty_conversation())
        assert "tools" in req
        assert req["tools"][0]["function"]["name"] == "read_file"

    def test_build_inference_request_omits_tools_when_disabled(self):
        rt = self._runtime()
        rt._native_tools_supported = False
        req = rt.build_inference_request(_empty_conversation())
        assert "tools" not in req

    def test_openai_normalizer_parses_string_arguments(self):
        calls = _normalize_openai_tool_calls(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/etc/hosts"}'},
                }
            ]
        )
        assert len(calls) == 1
        assert calls[0].arguments == {"path": "/etc/hosts"}

    def test_openai_normalizer_handles_empty_arguments(self):
        calls = _normalize_openai_tool_calls(
            [{"function": {"name": "ping", "arguments": ""}}]
        )
        assert calls[0].arguments == {}


# --------------------------------------------------------------------------- #
# Claude                                                                      #
# --------------------------------------------------------------------------- #


class _ToolUseBlock:
    """Minimal stand-in for Anthropic SDK's ``ToolUseBlock`` for testing."""

    def __init__(self, name: str, input_: dict[str, Any], id_: str = "toolu_xyz"):
        self.type = "tool_use"
        self.id = id_
        self.name = name
        self.input = input_


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class TestClaudeNativeTools:
    def _runtime(self) -> ClaudeCodeRuntime:
        config = TowelConfig(identity="You are Towel.")
        return ClaudeCodeRuntime(config, skills=_skills_with_one_tool(), model="haiku")

    def test_system_prompt_drops_inline_listing_when_native(self):
        rt = self._runtime()
        native = rt._build_system_prompt(include_tools_section=False)
        assert "- read_file" not in native
        assert "<tool_call>" not in native
        assert "Only call the tools you have been given" in native

    def test_build_inference_request_includes_anthropic_tools(self):
        rt = self._runtime()
        req = rt.build_inference_request(_empty_conversation())
        assert "tools" in req
        # Anthropic schema uses input_schema, no function/type envelope.
        assert req["tools"][0]["name"] == "read_file"
        assert "input_schema" in req["tools"][0]

    def test_extract_anthropic_tool_calls_from_objects(self):
        blocks = [
            _TextBlock("I'll read it now."),
            _ToolUseBlock("read_file", {"path": "/etc/hosts"}),
        ]
        calls = _extract_anthropic_tool_calls(blocks)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "/etc/hosts"}

    def test_extract_anthropic_tool_calls_from_dicts(self):
        # The streaming SDK sometimes hands us dicts rather than typed objects.
        blocks = [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "x", "name": "read_file", "input": {"path": "a"}},
        ]
        calls = _extract_anthropic_tool_calls(blocks)
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "a"}


class TestLlamaTokenAccounting:
    """`generate_from_request` must report sensible completion_tokens
    even when llama-server's `usage` block is missing or wrong.

    Reasoning models (Qwen3, DeepSeek-R1) routinely return empty
    `content` plus a populated `reasoning_content`; the runtime
    substitutes the reasoning text for content, but the `usage`
    block still reflects the empty content. Reporting 0 tokens
    when we just handed the caller 500 chars of text confuses
    both the UI and OpenAI-compat clients tracking spend.
    """

    def _runtime_with_mock(self, response_data: dict[str, Any]) -> LlamaRuntime:
        import httpx

        config = TowelConfig(identity="You are Towel.")
        rt = LlamaRuntime(config, skills=_skills_with_one_tool(), auto_start=False)
        rt._loaded = True

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=response_data)

        rt._mock_transport = httpx.MockTransport(handler)
        return rt

    def _request(self) -> dict[str, Any]:
        return {
            "mode": "llama_chat",
            "messages": [{"role": "user", "content": "hello"}],
            "system": "You are Towel.",
        }

    async def _generate(self, rt: LlamaRuntime, request: dict[str, Any]):
        # Patch httpx.AsyncClient to use the mock transport. The simplest
        # path that doesn't require a fixture is monkey-patching at the
        # module level for the duration of the call.
        import httpx
        from towel.agent import llama_runtime as mod

        orig_async_client = mod.httpx.AsyncClient

        def _client_with_mock(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = rt._mock_transport
            return orig_async_client(*args, **kwargs)

        mod.httpx.AsyncClient = _client_with_mock
        try:
            return await rt.generate_from_request(request)
        finally:
            mod.httpx.AsyncClient = orig_async_client

    def test_reasoning_fallback_estimates_completion_tokens(self):
        # Worst case: content empty, reasoning_content populated,
        # usage reports zero. The runtime returns reasoning as text
        # and must estimate the count rather than reporting 0.
        import asyncio

        reasoning = "This is a long reasoning trace. " * 20  # ~120 tokens
        rt = self._runtime_with_mock({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": reasoning,
                }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 0},
        })
        result = asyncio.run(self._generate(rt, self._request()))
        assert result.text.startswith("This is a long reasoning trace.")
        assert result.completion_tokens > 0
        # ~4 chars/token, length-of-reasoning gives us a meaningful number.
        assert result.completion_tokens >= 100

    def test_zero_usage_with_real_content_estimates_too(self):
        # Some llama-server builds simply omit `usage` or return zeros.
        # Reporting 0 tokens for visible content is the same UX bug.
        import asyncio

        rt = self._runtime_with_mock({
            "choices": [{"message": {"content": "Hello there!"}}],
            "usage": {},
        })
        result = asyncio.run(self._generate(rt, self._request()))
        assert result.text == "Hello there!"
        assert result.completion_tokens > 0

    def test_real_usage_is_preserved(self):
        # When llama-server reports a real count, we MUST use it
        # rather than overriding with our estimate.
        import asyncio

        rt = self._runtime_with_mock({
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })
        result = asyncio.run(self._generate(rt, self._request()))
        assert result.completion_tokens == 3
        assert result.prompt_tokens == 12


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _empty_conversation():
    from towel.agent.conversation import Conversation, Role

    conv = Conversation()
    conv.add(Role.USER, "hello")
    return conv

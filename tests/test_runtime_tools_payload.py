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


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _empty_conversation():
    from towel.agent.conversation import Conversation, Role

    conv = Conversation()
    conv.add(Role.USER, "hello")
    return conv

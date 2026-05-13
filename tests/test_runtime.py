"""Tests for the MLX runtime helpers and prompt rules."""

from towel.agent.runtime import (
    AgentRuntime,
    format_tool_feedback,
    mlx_tokenizer_config,
    tool_result_is_error,
)
from towel.config import TowelConfig
from towel.skills.base import Skill, ToolDefinition
from towel.skills.registry import SkillRegistry


class TestMlxTokenizerConfig:
    def test_enables_mistral_regex_fix(self):
        assert mlx_tokenizer_config() == {"fix_mistral_regex": True}


class TestToolFeedback:
    def test_classifies_common_tool_errors(self):
        assert tool_result_is_error("File not found: /tmp/missing.txt")
        assert tool_result_is_error("Unknown tool: read_files")
        assert not tool_result_is_error("Written 12 bytes to /tmp/test.txt")

    def test_formats_recovery_guidance_for_errors(self):
        text = format_tool_feedback("read_file", "File not found: x.txt", is_error=True)
        assert "[read_file]" in text
        assert "status: error" in text
        assert "Retry with one corrected valid tool call" in text


class TestSystemPrompt:
    def test_system_prompt_requires_reporting_results_back(self):
        config = TowelConfig(identity="You are Towel.")
        runtime = AgentRuntime(config)

        system = runtime._build_system_content()

        assert "always answer the user's original question" in system
        assert "explicitly report that back to the user" in system

    def test_system_prompt_includes_tool_recovery_rules(self):
        class DummySkill(Skill):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "dummy"

            def tools(self) -> list[ToolDefinition]:
                return [
                    ToolDefinition(
                        name="read_file",
                        description="Read a file",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ]

            async def execute(self, tool_name: str, arguments: dict):
                return "ok"

        config = TowelConfig(identity="You are Towel.")
        skills = SkillRegistry()
        skills.register(DummySkill())
        runtime = AgentRuntime(config, skills=skills)

        system = runtime._build_system_content()

        assert "prefer emitting just the tool call" in system
        assert "one corrected retry" in system


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


class TestNativeToolsChannel:
    def _runtime_with_one_tool(self) -> AgentRuntime:
        config = TowelConfig(identity="You are Towel.")
        skills = SkillRegistry()
        skills.register(_ReadFileSkill())
        return AgentRuntime(config, skills=skills)

    def test_native_path_omits_inline_tool_listing(self):
        runtime = self._runtime_with_one_tool()
        native = runtime._build_system_content(include_tools_section=False)
        # The per-tool bullet and call-format spec are gone…
        assert "- read_file" not in native
        assert "<tool_call>" not in native
        # …but behavioral guardrails remain.
        assert "Only call tools from the provided list" in native
        assert "prefer emitting just the tool call" in native
        assert "one corrected retry" in native

    def test_tools_for_chat_template_returns_openai_function_dicts(self):
        runtime = self._runtime_with_one_tool()
        tools = runtime._tools_for_chat_template()
        assert tools == [
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

    def test_detect_native_tools_support_true_when_template_renders_tools(self):
        runtime = self._runtime_with_one_tool()

        class FakeTokenizer:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                if tools:
                    names = ",".join(t["function"]["name"] for t in tools)
                    return f"<system>tools={names}</system><user>{messages[-1]['content']}</user>"
                return f"<user>{messages[-1]['content']}</user>"

        runtime._tokenizer = FakeTokenizer()
        assert runtime._detect_native_tools_support() is True

    def test_detect_native_tools_support_false_when_template_ignores_tools(self):
        runtime = self._runtime_with_one_tool()

        class IgnoresToolsTokenizer:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                # Older templates silently drop the tools kwarg.
                return f"<user>{messages[-1]['content']}</user>"

        runtime._tokenizer = IgnoresToolsTokenizer()
        assert runtime._detect_native_tools_support() is False

    def test_detect_native_tools_support_false_when_template_raises(self):
        runtime = self._runtime_with_one_tool()

        class RaisesTokenizer:
            def apply_chat_template(self, messages, tools=None, **kwargs):
                if tools is not None:
                    raise TypeError("got unexpected keyword 'tools'")
                return f"<user>{messages[-1]['content']}</user>"

        runtime._tokenizer = RaisesTokenizer()
        assert runtime._detect_native_tools_support() is False


class TestRegistrySuggestions:
    def test_unknown_tool_error_suggests_close_match(self):
        class DummySkill(Skill):
            @property
            def name(self) -> str:
                return "dummy"

            @property
            def description(self) -> str:
                return "dummy"

            def tools(self) -> list[ToolDefinition]:
                return [
                    ToolDefinition(
                        name="read_file",
                        description="Read a file",
                    )
                ]

            async def execute(self, tool_name: str, arguments: dict):
                return "ok"

        skills = SkillRegistry()
        skills.register(DummySkill())

        assert skills.suggest_tools("read_files") == ["read_file"]

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

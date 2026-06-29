"""Tests for the skill registry."""

from typing import Any

import pytest

from towel.skills.base import Skill, ToolDefinition
from towel.skills.registry import SkillRegistry


class MockSkill(Skill):
    @property
    def name(self) -> str:
        return "mock"

    @property
    def description(self) -> str:
        return "A mock skill for testing"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="mock_tool", description="Does mock things"),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return {"result": "mocked", "tool": tool_name}


def test_registry_register():
    reg = SkillRegistry()
    reg.register(MockSkill())
    assert len(reg) == 1
    assert "mock" in reg.list_skills()


def test_registry_tool_definitions():
    reg = SkillRegistry()
    reg.register(MockSkill())
    defs = reg.tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "mock_tool"


def test_registry_unregister():
    reg = SkillRegistry()
    reg.register(MockSkill())
    reg.unregister("mock")
    assert len(reg) == 0


class CmdSkill(Skill):
    """Minimal skill with a `command`-keyed tool to exercise aliasing."""

    @property
    def name(self) -> str:
        return "cmd"

    @property
    def description(self) -> str:
        return "Runs a command"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="run_command",
                description="Run a command",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return {"ran": arguments["command"]}


def test_primary_arg_key_prefers_required():
    reg = SkillRegistry()
    reg.register(CmdSkill())
    assert reg._primary_arg_key("run_command") == "command"


def test_resolve_alias_remaps_input_positional():
    reg = SkillRegistry()
    reg.register(CmdSkill())
    # shell isn't a real tool; it should resolve to run_command and the lone
    # {"input": ...} positional should land on the `command` parameter.
    name, args = reg._resolve_alias("shell", {"input": "echo hi"})
    assert name == "run_command"
    assert args == {"command": "echo hi"}


def test_resolve_alias_preserves_correct_args():
    reg = SkillRegistry()
    reg.register(CmdSkill())
    name, args = reg._resolve_alias("bash", {"command": "echo hi"})
    assert name == "run_command"
    assert args == {"command": "echo hi"}


def test_resolve_alias_noop_for_unknown_tool():
    reg = SkillRegistry()
    reg.register(CmdSkill())
    assert reg._resolve_alias("frobnicate", {"input": "x"}) == ("frobnicate", {"input": "x"})


def test_resolve_alias_noop_when_target_unregistered():
    # Alias exists in the table but run_command isn't registered here.
    reg = SkillRegistry()
    reg.register(MockSkill())
    assert reg._resolve_alias("shell", {"input": "x"}) == ("shell", {"input": "x"})


async def test_execute_tool_runs_via_alias():
    reg = SkillRegistry()
    reg.register(CmdSkill())
    result = await reg.execute_tool("shell", {"input": "echo hi"})
    assert result == {"ran": "echo hi"}


async def test_execute_tool_unknown_still_raises():
    reg = SkillRegistry()
    reg.register(CmdSkill())
    with pytest.raises(ValueError, match="Unknown tool"):
        await reg.execute_tool("definitely_not_a_tool", {})


def test_coerce_arguments_remaps_misfit_positional():
    """A correctly-named tool called with the wrong single key gets that value
    moved onto the primary parameter (small-model robustness)."""
    reg = SkillRegistry()
    reg.register(CmdSkill())
    assert reg._coerce_arguments("run_command", {"input": "ls"}) == {"command": "ls"}
    assert reg._coerce_arguments("run_command", {"text": "ls"}) == {"command": "ls"}


def test_coerce_arguments_leaves_valid_and_multiarg_alone():
    reg = SkillRegistry()
    reg.register(CmdSkill())
    # already-correct key untouched
    assert reg._coerce_arguments("run_command", {"command": "ls"}) == {"command": "ls"}
    # multi-arg calls are never reshaped (too ambiguous to guess)
    two = {"command": "ls", "extra": 1}
    assert reg._coerce_arguments("run_command", two) == two


async def test_execute_tool_runs_with_misfit_key():
    """End-to-end: run_command({"input": ...}) actually executes."""
    reg = SkillRegistry()
    reg.register(CmdSkill())
    assert await reg.execute_tool("run_command", {"input": "echo hi"}) == {"ran": "echo hi"}

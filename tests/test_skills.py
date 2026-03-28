"""Tests for the skill registry."""

from typing import Any

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

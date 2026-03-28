"""Base skill interface."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDefinition:
    """A tool that a skill exposes to the agent."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class Skill(abc.ABC):
    """A skill is a bundle of tools the agent can use.

    Skills are loaded from the Laundromat (skills registry) or from
    local directories. Each skill exposes one or more tools.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique skill name."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @abc.abstractmethod
    def tools(self) -> list[ToolDefinition]:
        """Return the tools this skill provides."""
        ...

    @abc.abstractmethod
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments."""
        ...

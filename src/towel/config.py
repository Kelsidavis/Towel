"""Towel configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel, Field


TOWEL_HOME = Path(os.environ.get("TOWEL_HOME", Path.home() / ".towel"))


class ModelConfig(BaseModel):
    """MLX model configuration."""

    name: str = "Eldadalbajob/Huihui-Qwen3-Next-80B-A3B-Instruct-abliterated-mlx-3Bit"
    max_tokens: int = 4096
    context_window: int = 8192
    temperature: float = 0.7
    top_p: float = 0.95


class AgentProfile(BaseModel):
    """A named agent profile — model + identity + behavior."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    identity: str = "You are Towel, a helpful local AI assistant. Don't Panic."
    skills: list[str] = Field(default_factory=list)  # empty = all skills
    description: str = ""

    def effective_identity(self, base_identity: str) -> str:
        """Return this profile's identity, falling back to base."""
        if self.identity and self.identity != AgentProfile.model_fields["identity"].default:
            return self.identity
        return base_identity


class GatewayConfig(BaseModel):
    """Gateway server configuration."""

    host: str = "127.0.0.1"
    port: int = 18742  # 42 * 446 + 10, because 42
    ws_path: str = "/ws"


class ChannelDefaults(BaseModel):
    """Default channel routing configuration."""

    enabled: list[str] = Field(default_factory=lambda: ["cli", "webchat"])


# Built-in agent profiles
DEFAULT_AGENTS: dict[str, dict[str, Any]] = {
    "coder": {
        "model": {"name": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit", "context_window": 32768},
        "identity": (
            "You are Towel (coder mode), an expert software engineer. "
            "Write clean, efficient code. Explain your reasoning. "
            "Use tools to read files and run commands. Don't Panic."
        ),
        "description": "Code generation and software engineering",
    },
    "researcher": {
        "model": {"name": "mlx-community/Llama-3.3-70B-Instruct-4bit", "context_window": 16384},
        "identity": (
            "You are Towel (researcher mode), a thorough research assistant. "
            "Analyze information carefully, cite sources, and present balanced views. "
            "Use tools to fetch web content and read files. Don't Panic."
        ),
        "description": "Research, analysis, and information synthesis",
    },
    "writer": {
        "model": {"name": "mlx-community/Llama-3.3-70B-Instruct-4bit", "temperature": 0.9},
        "identity": (
            "You are Towel (writer mode), a creative writing assistant. "
            "Help with drafting, editing, and refining text. "
            "Adapt your tone to match the user's needs. Don't Panic."
        ),
        "description": "Creative and technical writing",
    },
}


class TowelConfig(BaseModel):
    """Root configuration for Towel."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    channels: ChannelDefaults = Field(default_factory=ChannelDefaults)
    skills_dirs: list[str] = Field(default_factory=lambda: ["~/.towel/skills", "./skills"])
    identity: str = "You are Towel, a helpful local AI assistant. Don't Panic."
    agents: dict[str, AgentProfile] = Field(default_factory=dict)
    default_agent: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> TowelConfig:
        """Load config from TOML file, falling back to defaults."""
        config_path = path or TOWEL_HOME / "config.toml"
        if config_path.exists():
            data: dict[str, Any] = toml.load(config_path)
            return cls.model_validate(data)
        return cls()

    def save(self, path: Path | None = None) -> None:
        """Save config to TOML file."""
        config_path = path or TOWEL_HOME / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(toml.dumps(self.model_dump()))

    def list_agents(self) -> dict[str, AgentProfile]:
        """List all available agents (user-defined + built-in)."""
        result: dict[str, AgentProfile] = {}
        # Built-in agents first
        for name, data in DEFAULT_AGENTS.items():
            result[name] = AgentProfile.model_validate(data)
        # User agents from agents.toml
        agents_file = TOWEL_HOME / "agents.toml"
        if agents_file.exists():
            try:
                import toml as _toml
                for name, data in _toml.load(agents_file).items():
                    result[name] = AgentProfile.model_validate(data)
            except Exception:
                pass
        # Config-defined override everything
        result.update(self.agents)
        return result

    def get_agent(self, name: str) -> AgentProfile | None:
        """Get an agent profile by name (config, agents.toml, or built-in)."""
        all_agents = self.list_agents()
        return all_agents.get(name)

    def resolve_agent(self, agent_name: str | None = None) -> tuple[ModelConfig, str]:
        """Resolve an agent name to (model_config, identity).

        Falls back to: explicit agent -> default_agent config -> base config.
        """
        name = agent_name or self.default_agent
        if name:
            profile = self.get_agent(name)
            if profile:
                return profile.model, profile.effective_identity(self.identity)
        return self.model, self.identity

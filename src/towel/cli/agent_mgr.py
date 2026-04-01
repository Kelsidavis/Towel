"""Agent management — create, edit, delete, and clone agent profiles.

User-created agents are stored in ~/.towel/agents.toml separately
from the main config to avoid clobbering comments.
"""

from __future__ import annotations

from typing import Any

import toml

from towel.config import TOWEL_HOME, AgentProfile

AGENTS_FILE = TOWEL_HOME / "agents.toml"


def load_user_agents() -> dict[str, dict[str, Any]]:
    """Load user-created agents from agents.toml."""
    if not AGENTS_FILE.exists():
        return {}
    try:
        return toml.load(AGENTS_FILE)
    except Exception:
        return {}


def save_user_agents(agents: dict[str, dict[str, Any]]) -> None:
    """Save user-created agents to agents.toml."""
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    AGENTS_FILE.write_text(toml.dumps(agents), encoding="utf-8")


def create_agent(
    name: str,
    model_name: str,
    identity: str,
    description: str = "",
    context_window: int = 8192,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    turboquant: bool = False,
    turboquant_bits: int = 3,
) -> AgentProfile:
    """Create and persist a new user agent profile."""
    agents = load_user_agents()

    model_cfg: dict = {
        "name": model_name,
        "context_window": context_window,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if turboquant:
        model_cfg["turboquant"] = True
        model_cfg["turboquant_bits"] = turboquant_bits

    agents[name] = {
        "description": description,
        "identity": identity,
        "model": model_cfg,
    }

    save_user_agents(agents)
    return AgentProfile.model_validate(agents[name])


def delete_agent(name: str) -> bool:
    """Delete a user-created agent. Returns True if it existed."""
    agents = load_user_agents()
    if name not in agents:
        return False
    del agents[name]
    save_user_agents(agents)
    return True


def clone_agent(source_name: str, new_name: str, config: Any) -> AgentProfile | None:
    """Clone an existing agent (built-in or user) under a new name."""
    profile = config.get_agent(source_name)
    if not profile:
        return None

    return create_agent(
        name=new_name,
        model_name=profile.model.name,
        identity=profile.identity,
        description=f"Clone of {source_name}. {profile.description}",
        context_window=profile.model.context_window,
        temperature=profile.model.temperature,
        max_tokens=profile.model.max_tokens,
        turboquant=profile.model.turboquant,
        turboquant_bits=profile.model.turboquant_bits,
    )

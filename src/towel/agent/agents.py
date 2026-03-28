"""Persistent agents — long-running autonomous agents with goals.

Unlike the orchestrator (one-shot delegation), persistent agents
run continuously, check conditions, and take actions autonomously.

Usage:
    agent = AutonomousAgent("monitor", goal="Watch the API and alert if it goes down",
        check_interval=300, tools=["uptime_check", "slack_message"])
    await agent.start(registry)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.agent.agents")

AGENTS_FILE = TOWEL_HOME / "agents.json"


@dataclass
class AgentLog:
    timestamp: str
    action: str
    result: str

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "action": self.action, "result": self.result[:200]}


@dataclass
class AutonomousAgent:
    """A persistent autonomous agent with a goal."""
    name: str
    goal: str
    check_interval: int = 300  # seconds
    tools: list[str] = field(default_factory=list)
    enabled: bool = True
    logs: list[AgentLog] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_runs: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "goal": self.goal,
            "check_interval": self.check_interval,
            "tools": self.tools, "enabled": self.enabled,
            "created_at": self.created_at, "total_runs": self.total_runs,
            "logs": [l.to_dict() for l in self.logs[-20:]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> AutonomousAgent:
        logs = [AgentLog(**l) for l in d.get("logs", [])]
        return cls(
            name=d["name"], goal=d["goal"],
            check_interval=d.get("check_interval", 300),
            tools=d.get("tools", []), enabled=d.get("enabled", True),
            created_at=d.get("created_at", ""), total_runs=d.get("total_runs", 0),
            logs=logs,
        )


def _load_agents() -> list[AutonomousAgent]:
    if not AGENTS_FILE.exists(): return []
    try: return [AutonomousAgent.from_dict(a) for a in json.loads(AGENTS_FILE.read_text())]
    except: return []


def _save_agents(agents: list[AutonomousAgent]) -> None:
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    AGENTS_FILE.write_text(json.dumps([a.to_dict() for a in agents], indent=2))


def create_agent(name: str, goal: str, interval: int = 300, tools: list[str] | None = None) -> AutonomousAgent:
    agents = _load_agents()
    agent = AutonomousAgent(name=name, goal=goal, check_interval=interval, tools=tools or [])
    agents = [a for a in agents if a.name != name]
    agents.append(agent)
    _save_agents(agents)
    return agent


def delete_agent(name: str) -> bool:
    agents = _load_agents()
    before = len(agents)
    agents = [a for a in agents if a.name != name]
    if len(agents) < before:
        _save_agents(agents)
        return True
    return False


def list_agents() -> list[AutonomousAgent]:
    return _load_agents()


def get_agent(name: str) -> AutonomousAgent | None:
    return next((a for a in _load_agents() if a.name == name), None)


def log_agent_action(name: str, action: str, result: str) -> None:
    agents = _load_agents()
    for a in agents:
        if a.name == name:
            a.logs.append(AgentLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=action, result=result,
            ))
            a.total_runs += 1
            if len(a.logs) > 50: a.logs = a.logs[-50:]
            break
    _save_agents(agents)

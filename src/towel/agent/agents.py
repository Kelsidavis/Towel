"""Persistent agents — long-running autonomous agents with goals.

Unlike the orchestrator (one-shot delegation), persistent agents
run continuously, check conditions, and take actions autonomously.

Usage:
    agent = AutonomousAgent("monitor", goal="Watch the API and alert if it goes down",
        check_interval=300, tools=["uptime_check", "slack_message"])
    await agent.start(registry)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

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
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_runs: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal,
            "check_interval": self.check_interval,
            "tools": self.tools,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "total_runs": self.total_runs,
            "logs": [entry.to_dict() for entry in self.logs[-20:]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> AutonomousAgent:
        logs = [AgentLog(**entry) for entry in d.get("logs", [])]
        return cls(
            name=d["name"],
            goal=d["goal"],
            check_interval=d.get("check_interval", 300),
            tools=d.get("tools", []),
            enabled=d.get("enabled", True),
            created_at=d.get("created_at", ""),
            total_runs=d.get("total_runs", 0),
            logs=logs,
        )


def _load_agents() -> list[AutonomousAgent]:
    if not AGENTS_FILE.exists():
        return []
    try:
        raw = AGENTS_FILE.read_text(encoding="utf-8")
        return [AutonomousAgent.from_dict(a) for a in json.loads(raw)]
    except Exception as exc:
        # Rename the corrupt file aside so the next _save_agents call
        # can't overwrite the bytes with a fresh (probably empty)
        # agent list. Same data-durability pattern the persistence
        # stores adopted (5512834, 98d1c68, 8a86987).
        from datetime import UTC, datetime
        backup = AGENTS_FILE.with_name(
            f"{AGENTS_FILE.name}.corrupted-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
        )
        try:
            AGENTS_FILE.replace(backup)
            import logging
            logging.getLogger("towel.agents").warning(
                "Failed to load agents: %s. Backed up the bad file to %s.",
                exc, backup,
            )
        except OSError:
            pass
        return []


def _save_agents(agents: list[AutonomousAgent]) -> None:
    AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: dump to a sibling .tmp then rename. Without this,
    # a kill / disk-full mid-write leaves a half-written agents file
    # that the next _load_agents classifies as corrupt and discards.
    tmp = AGENTS_FILE.with_name(AGENTS_FILE.name + ".tmp")
    tmp.write_text(json.dumps([a.to_dict() for a in agents], indent=2), encoding="utf-8")
    tmp.replace(AGENTS_FILE)


def create_agent(
    name: str, goal: str, interval: int = 300, tools: list[str] | None = None
) -> AutonomousAgent:
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
            a.logs.append(
                AgentLog(
                    timestamp=datetime.now(UTC).isoformat(),
                    action=action,
                    result=result,
                )
            )
            a.total_runs += 1
            if len(a.logs) > 50:
                a.logs = a.logs[-50:]
            break
    _save_agents(agents)

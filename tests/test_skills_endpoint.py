"""Tests for the /skills introspection endpoint."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore
from towel.skills.base import Skill, ToolDefinition
from towel.skills.registry import SkillRegistry


class _EchoSkill(Skill):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back text loudly or softly."

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="shout",
                description="Yell the message in caps.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "decibels": {"type": "number"}},
                    "required": ["text"],
                },
            ),
            ToolDefinition(
                name="whisper",
                description="Speak softly. Takes no arguments.",
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return "ok"


class _FakeAgent:
    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


def _make_gateway(store, registry: SkillRegistry | None = None) -> GatewayServer:
    sessions = SessionManager(store=store)
    pin_store = SessionPinStore(path=store.store_dir / "session_pins.json")
    worker_state_store = WorkerStateStore(path=store.store_dir / "worker_state.json")
    return GatewayServer(
        config=TowelConfig(),
        agent=_FakeAgent(registry or SkillRegistry()),
        sessions=sessions,
        pin_store=pin_store,
        worker_state_store=worker_state_store,
    )


class TestSkillsEndpoint:
    def test_lists_loaded_skills_with_their_tools(self, store):
        registry = SkillRegistry()
        registry.register(_EchoSkill())
        gw = _make_gateway(store, registry)
        client = TestClient(gw._build_http_app())

        resp = client.get("/skills")
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_tools"] == 2
        assert len(data["skills"]) == 1
        skill = data["skills"][0]
        assert skill["name"] == "echo"
        assert skill["description"] == "Echo back text loudly or softly."
        assert skill["tool_count"] == 2

        tool_names = {t["name"] for t in skill["tools"]}
        assert tool_names == {"shout", "whisper"}

        shout = next(t for t in skill["tools"] if t["name"] == "shout")
        assert shout["description"] == "Yell the message in caps."
        assert set(shout["parameters"]) == {"text", "decibels"}

        whisper = next(t for t in skill["tools"] if t["name"] == "whisper")
        assert whisper["parameters"] == []  # no-arg tools surface an empty list

    def test_returns_empty_when_agent_has_no_skill_registry(self, store):
        # An agent that doesn't carry a ``.skills`` attribute (e.g. a stripped
        # test agent) should produce a clean empty response, not crash.
        class _BareAgent:
            pass

        sessions = SessionManager(store=store)
        pin_store = SessionPinStore(path=store.store_dir / "session_pins.json")
        worker_state_store = WorkerStateStore(path=store.store_dir / "worker_state.json")
        gw = GatewayServer(
            config=TowelConfig(),
            agent=_BareAgent(),
            sessions=sessions,
            pin_store=pin_store,
            worker_state_store=worker_state_store,
        )
        client = TestClient(gw._build_http_app())

        resp = client.get("/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"skills": [], "total_tools": 0}

    def test_handles_empty_registry(self, store):
        gw = _make_gateway(store, SkillRegistry())
        client = TestClient(gw._build_http_app())
        resp = client.get("/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skills"] == []
        assert data["total_tools"] == 0

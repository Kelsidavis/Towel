"""Tests for session worker pin persistence."""

from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore


class TestSessionPinStore:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = SessionPinStore(path=tmp_path / "pins.json")

        store.save({"chat-1": "desktop-1", "chat-2": "desktop-2"})

        assert store.load() == {"chat-1": "desktop-1", "chat-2": "desktop-2"}

    def test_load_missing_file_returns_empty(self, tmp_path):
        store = SessionPinStore(path=tmp_path / "pins.json")

        assert store.load() == {}


class TestGatewayPinPersistence:
    def test_gateway_loads_persisted_pins(self, tmp_path):
        pin_store = SessionPinStore(path=tmp_path / "pins.json")
        pin_store.save({"chat-1": "desktop-1"})

        gateway = GatewayServer(
            config=TowelConfig(),
            agent=AgentRuntime(TowelConfig()),
            sessions=SessionManager(store=ConversationStore(store_dir=tmp_path / "conversations")),
            pin_store=pin_store,
        )

        assert gateway._session_pins == {"chat-1": "desktop-1"}

    def test_pin_and_unpin_write_store(self, tmp_path):
        pin_store = SessionPinStore(path=tmp_path / "pins.json")
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=AgentRuntime(TowelConfig()),
            sessions=SessionManager(store=ConversationStore(store_dir=tmp_path / "conversations")),
            pin_store=pin_store,
        )
        gateway._workers.register(
            "desktop-1", object(), {"backend": "mlx", "modes": ["mlx_prompt"]}
        )

        assert gateway.pin_session_worker("chat-1", "desktop-1") is True
        assert pin_store.load() == {"chat-1": "desktop-1"}

        assert gateway.unpin_session_worker("chat-1") is True
        assert pin_store.load() == {}

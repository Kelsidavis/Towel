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

    def test_save_is_atomic(self, tmp_path, monkeypatch):
        """A kill or disk-full mid-write must not corrupt the on-disk
        file — the previous pins must still load cleanly. Achieved by
        writing to a .tmp sibling then renaming."""
        pins_path = tmp_path / "pins.json"
        store = SessionPinStore(path=pins_path)
        store.save({"sess-a": "worker-1"})
        assert store.load() == {"sess-a": "worker-1"}

        # Simulate a failure between the write and the rename.
        from pathlib import Path
        original_replace = Path.replace

        def failing_replace(self, target):
            raise OSError("simulated disk-full at rename time")

        monkeypatch.setattr(Path, "replace", failing_replace)
        try:
            store.save({"sess-a": "worker-1", "sess-b": "worker-2"})
        except OSError:
            pass
        finally:
            monkeypatch.setattr(Path, "replace", original_replace)

        # The original file must be intact — half-written state in
        # the .tmp sibling is fine because load() ignores it.
        assert store.load() == {"sess-a": "worker-1"}


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

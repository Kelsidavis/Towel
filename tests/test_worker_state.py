"""Tests for worker operational state persistence."""

from towel.agent.runtime import AgentRuntime
from towel.config import TowelConfig
from towel.gateway.server import GatewayServer
from towel.gateway.sessions import SessionManager
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore


class TestWorkerStateStore:
    def test_save_and_load_roundtrip(self, tmp_path):
        store = WorkerStateStore(path=tmp_path / "worker_state.json")

        store.save(
            {
                "desktop-1": {"enabled": False, "draining": True},
                "desktop-2": {"enabled": True, "draining": False},
            }
        )

        assert store.load() == {
            "desktop-1": {"enabled": False, "draining": True},
            "desktop-2": {"enabled": True, "draining": False},
        }

    def test_load_missing_file_returns_empty(self, tmp_path):
        store = WorkerStateStore(path=tmp_path / "worker_state.json")

        assert store.load() == {}

    def test_save_is_atomic(self, tmp_path, monkeypatch):
        """A kill / disk-full mid-write must not destroy the existing
        on-disk state — the previous enabled / draining / tasks
        overrides must still load. Same atomic-rename pattern as
        SessionPinStore (and memory/store.py per 5512834)."""
        state_path = tmp_path / "worker_state.json"
        store = WorkerStateStore(path=state_path)
        store.save({"worker-a": {"enabled": False, "draining": False}})
        assert store.load() == {
            "worker-a": {"enabled": False, "draining": False},
        }

        from pathlib import Path
        original_replace = Path.replace

        def failing_replace(self, target):
            raise OSError("simulated disk-full at rename time")

        monkeypatch.setattr(Path, "replace", failing_replace)
        try:
            store.save(
                {
                    "worker-a": {"enabled": True, "draining": True},
                    "worker-b": {"enabled": True, "draining": False},
                }
            )
        except OSError:
            pass
        finally:
            monkeypatch.setattr(Path, "replace", original_replace)

        # The original file must remain readable with the previous
        # state.
        assert store.load() == {
            "worker-a": {"enabled": False, "draining": False},
        }


class TestGatewayWorkerStatePersistence:
    def test_gateway_loads_persisted_worker_state_on_register(self, tmp_path):
        worker_state_store = WorkerStateStore(path=tmp_path / "worker_state.json")
        worker_state_store.save({"desktop-1": {"enabled": False, "draining": True}})

        gateway = GatewayServer(
            config=TowelConfig(),
            agent=AgentRuntime(TowelConfig()),
            sessions=SessionManager(store=ConversationStore(store_dir=tmp_path / "conversations")),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=worker_state_store,
        )

        worker = gateway._workers.register("desktop-1", object(), {"backend": "mlx"})
        gateway._workers.apply_state(
            "desktop-1",
            enabled=gateway._worker_states["desktop-1"]["enabled"],
            draining=gateway._worker_states["desktop-1"]["draining"],
        )

        assert worker.enabled is False
        assert worker.draining is True

    def test_saving_worker_state_writes_store(self, tmp_path):
        worker_state_store = WorkerStateStore(path=tmp_path / "worker_state.json")
        gateway = GatewayServer(
            config=TowelConfig(),
            agent=AgentRuntime(TowelConfig()),
            sessions=SessionManager(store=ConversationStore(store_dir=tmp_path / "conversations")),
            pin_store=SessionPinStore(path=tmp_path / "pins.json"),
            worker_state_store=worker_state_store,
        )
        gateway._workers.register("desktop-1", object(), {"backend": "mlx"})
        gateway._workers.set_enabled("desktop-1", False)
        gateway._workers.set_draining("desktop-1", True)

        gateway._save_worker_states()

        assert worker_state_store.load() == {
            "desktop-1": {"enabled": False, "draining": True}
        }

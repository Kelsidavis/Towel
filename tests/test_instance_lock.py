"""Tests for the singleton Towel runtime lock."""

from pathlib import Path

import pytest

from towel.agent import instance_lock


class TestInstanceLock:
    def test_acquire_and_release(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "runtime.lock"
        monkeypatch.setattr(instance_lock, "_LOCK_PATH", lock_path)
        monkeypatch.setattr(instance_lock, "_LOCK_HELD", False)
        monkeypatch.setattr(instance_lock, "_LOCK_PID", None)

        instance_lock.acquire_runtime_lock()

        assert lock_path.exists()
        assert lock_path.read_text() == str(instance_lock.os.getpid())

        instance_lock.release_runtime_lock()
        assert not lock_path.exists()

    def test_reentrant_same_process_is_allowed(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "runtime.lock"
        monkeypatch.setattr(instance_lock, "_LOCK_PATH", lock_path)
        monkeypatch.setattr(instance_lock, "_LOCK_HELD", False)
        monkeypatch.setattr(instance_lock, "_LOCK_PID", None)

        instance_lock.acquire_runtime_lock()
        instance_lock.acquire_runtime_lock()

        assert lock_path.exists()
        assert lock_path.read_text() == str(instance_lock.os.getpid())

    def test_stale_lock_is_replaced(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "runtime.lock"
        lock_path.write_text("999999")
        monkeypatch.setattr(instance_lock, "_LOCK_PATH", lock_path)
        monkeypatch.setattr(instance_lock, "_LOCK_HELD", False)
        monkeypatch.setattr(instance_lock, "_LOCK_PID", None)
        monkeypatch.setattr(instance_lock, "_pid_is_running", lambda pid: False)

        instance_lock.acquire_runtime_lock()

        assert lock_path.read_text() == str(instance_lock.os.getpid())

    def test_active_foreign_lock_raises(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "runtime.lock"
        lock_path.write_text("424242")
        monkeypatch.setattr(instance_lock, "_LOCK_PATH", lock_path)
        monkeypatch.setattr(instance_lock, "_LOCK_HELD", False)
        monkeypatch.setattr(instance_lock, "_LOCK_PID", None)
        monkeypatch.setattr(instance_lock, "_pid_is_running", lambda pid: True)

        with pytest.raises(RuntimeError, match="already running"):
            instance_lock.acquire_runtime_lock()

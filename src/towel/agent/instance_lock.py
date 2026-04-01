"""Cross-process runtime lock to prevent multiple Towel instances."""

from __future__ import annotations

import atexit
import os
from pathlib import Path

from towel.config import TOWEL_HOME

_LOCK_PATH = TOWEL_HOME / "runtime.lock"
_LOCK_HELD = False
_LOCK_PID: int | None = None


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def release_runtime_lock() -> None:
    """Release the process-wide runtime lock if held by this process."""
    global _LOCK_HELD, _LOCK_PID

    if not _LOCK_HELD or _LOCK_PID != os.getpid():
        return

    try:
        if _LOCK_PATH.exists() and _read_lock_pid(_LOCK_PATH) == _LOCK_PID:
            _LOCK_PATH.unlink()
    except OSError:
        pass

    _LOCK_HELD = False
    _LOCK_PID = None


def acquire_runtime_lock() -> None:
    """Acquire the singleton runtime lock or raise if another instance owns it."""
    global _LOCK_HELD, _LOCK_PID

    pid = os.getpid()
    if _LOCK_HELD and _LOCK_PID == pid:
        return

    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner_pid = _read_lock_pid(_LOCK_PATH)
            if owner_pid is None or not _pid_is_running(owner_pid):
                try:
                    _LOCK_PATH.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    raise RuntimeError(
                        "Another Towel instance appears to be starting. "
                        "Please wait a moment and try again."
                    ) from None
                continue

            if owner_pid == pid:
                _LOCK_HELD = True
                _LOCK_PID = pid
                return

            raise RuntimeError(
                f"Towel is already running in another process (PID {owner_pid}). "
                "Stop the existing instance before starting a new one."
            ) from None
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(pid))
            _LOCK_HELD = True
            _LOCK_PID = pid
            atexit.register(release_runtime_lock)
            return

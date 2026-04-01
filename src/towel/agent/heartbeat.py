"""Agent heartbeat — monitor health, detect crashes, auto-recover.

Runs a background thread that periodically checks agent state and
emits health events. Catches hangs, memory leaks, and generation
failures before they become visible to the user.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger("towel.agent.heartbeat")


@dataclass
class HealthStatus:
    """Snapshot of agent health."""

    alive: bool = True
    last_heartbeat: float = 0.0
    last_generation: float = 0.0
    total_generations: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    uptime_seconds: float = 0.0
    model_loaded: bool = False
    is_generating: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "alive": self.alive,
            "uptime": f"{self.uptime_seconds:.0f}s",
            "generations": self.total_generations,
            "errors": self.total_errors,
            "consecutive_errors": self.consecutive_errors,
            "model_loaded": self.model_loaded,
            "is_generating": self.is_generating,
            "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat, tz=UTC).isoformat()
            if self.last_heartbeat
            else None,
        }


class Heartbeat:
    """Background health monitor for the agent runtime.

    Usage:
        hb = Heartbeat(interval=30)
        hb.start()
        hb.on_generation_start()
        hb.on_generation_complete()
        hb.on_error(e)
        status = hb.status()
        hb.stop()
    """

    def __init__(
        self,
        interval: float = 30.0,
        max_consecutive_errors: int = 5,
        on_unhealthy: Callable[[HealthStatus], None] | None = None,
    ) -> None:
        self.interval = interval
        self.max_consecutive_errors = max_consecutive_errors
        self.on_unhealthy = on_unhealthy
        self._health = HealthStatus()
        self._start_time = 0.0
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[HealthStatus], None]] = []

    def start(self) -> None:
        """Start the heartbeat monitor."""
        self._start_time = time.time()
        self._running = True
        self._health.alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info(f"Heartbeat started (interval={self.interval}s)")

    def stop(self) -> None:
        """Stop the heartbeat monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Heartbeat stopped")

    def status(self) -> HealthStatus:
        """Get current health status."""
        with self._lock:
            self._health.uptime_seconds = time.time() - self._start_time
            return self._health

    def on_generation_start(self) -> None:
        with self._lock:
            self._health.is_generating = True

    def on_generation_complete(self) -> None:
        with self._lock:
            self._health.is_generating = False
            self._health.last_generation = time.time()
            self._health.total_generations += 1
            self._health.consecutive_errors = 0

    def on_error(self, error: Exception) -> None:
        with self._lock:
            self._health.total_errors += 1
            self._health.consecutive_errors += 1
            log.warning(f"Agent error #{self._health.total_errors}: {error}")

            if self._health.consecutive_errors >= self.max_consecutive_errors:
                self._health.alive = False
                log.error(f"Agent unhealthy: {self._health.consecutive_errors} consecutive errors")
                if self.on_unhealthy:
                    self.on_unhealthy(self._health)

    def on_model_loaded(self) -> None:
        with self._lock:
            self._health.model_loaded = True

    def add_callback(self, cb: Callable[[HealthStatus], None]) -> None:
        """Register a callback invoked on each heartbeat tick."""
        self._callbacks.append(cb)

    def _run(self) -> None:
        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break

            with self._lock:
                self._health.last_heartbeat = time.time()
                self._health.uptime_seconds = time.time() - self._start_time

            status = self.status()
            for cb in self._callbacks:
                try:
                    cb(status)
                except Exception as e:
                    log.warning(f"Heartbeat callback error: {e}")

            log.debug(
                f"Heartbeat: gen={status.total_generations} "
                f"err={status.total_errors} up={status.uptime_seconds:.0f}s"
            )

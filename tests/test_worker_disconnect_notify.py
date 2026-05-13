"""Tests for fast disconnect-notification of in-flight job waiters.

When a worker disconnects mid-job, the coordinator should push a synthetic
``job_error`` into the corresponding ``_job_queues`` entry so callers that
are blocked on ``queue.get()`` wake up immediately instead of waiting for
their per-call timeout (5-300s depending on the call site).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from towel.config import TowelConfig
from towel.gateway.server import GatewayServer


class _FakeAgent:
    pass


def _make_gateway() -> GatewayServer:
    return GatewayServer(agent=_FakeAgent(), config=TowelConfig())


def _register_busy_worker(gw: GatewayServer, worker_id: str, job_id: str) -> asyncio.Queue:
    """Register a worker, mark it as running ``job_id``, and return its queue."""
    gw._workers.register(worker_id, ws=MagicMock(), capabilities={})
    gw._workers.assign(worker_id, job_id, session_id="sess")
    queue: asyncio.Queue = asyncio.Queue()
    gw._job_queues[job_id] = queue
    return queue


def test_notify_pushes_job_error_for_active_job():
    gw = _make_gateway()
    queue = _register_busy_worker(gw, "worker_a", "job_42")

    gw._notify_in_flight_disconnect("worker_a")

    # The waiter sees the synthetic error without ever calling the network.
    assert not queue.empty()
    msg = queue.get_nowait()
    assert msg["type"] == "job_error"
    assert msg["job_id"] == "job_42"
    assert msg["worker_id"] == "worker_a"
    assert "disconnect" in msg["error"].lower()


def test_notify_is_noop_when_worker_has_no_active_job():
    gw = _make_gateway()
    gw._workers.register("worker_idle", ws=MagicMock(), capabilities={})
    # No assign() — worker.current_job_id stays None.

    # Should not raise.
    gw._notify_in_flight_disconnect("worker_idle")


def test_notify_is_noop_for_unknown_worker():
    gw = _make_gateway()
    # Never registered.
    gw._notify_in_flight_disconnect("ghost")


def test_notify_is_noop_when_queue_already_cleaned_up():
    """If the waiter has already popped the queue (e.g. completed), the worker
    might still show current_job_id briefly. The notify path must handle the
    missing-queue case without raising."""
    gw = _make_gateway()
    gw._workers.register("worker_b", ws=MagicMock(), capabilities={})
    gw._workers.assign("worker_b", "job_99", session_id="sess")
    # Note: no entry in _job_queues for job_99
    gw._notify_in_flight_disconnect("worker_b")


def test_waiter_unblocks_within_milliseconds_after_notify():
    """End-to-end: a coroutine blocked on queue.get() with a long timeout
    should resume as soon as _notify_in_flight_disconnect runs."""
    gw = _make_gateway()
    queue = _register_busy_worker(gw, "worker_c", "job_55")

    async def waiter() -> dict:
        return await asyncio.wait_for(queue.get(), timeout=30.0)

    async def runner() -> dict:
        task = asyncio.create_task(waiter())
        # Give the waiter a moment to actually start blocking.
        await asyncio.sleep(0.01)
        gw._notify_in_flight_disconnect("worker_c")
        return await task

    msg = asyncio.run(runner())
    assert msg["type"] == "job_error"
    assert msg["worker_id"] == "worker_c"

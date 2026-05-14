"""Workers track when they became busy so operators can tell a normal
long-running job apart from a stuck/hung worker. Without this, /workers
shows ``busy: true`` for everything from a 5-second prompt to a 20-minute
wedge with no way to distinguish them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from towel.gateway.workers import WorkerRegistry


class TestBusySinceTracking:
    def test_busy_since_set_on_assign(self):
        reg = WorkerRegistry()
        before = datetime.now(UTC)
        reg.register("w1", MagicMock(), {"backend": "mlx"})
        reg.assign("w1", "job-1", "sess-1")
        after = datetime.now(UTC)

        worker = reg.get("w1")
        assert worker is not None
        assert worker.busy is True
        assert worker.busy_since is not None
        assert before <= worker.busy_since <= after

    def test_busy_since_cleared_on_release(self):
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        reg.assign("w1", "job-1", "sess-1")
        reg.release("w1")
        worker = reg.get("w1")
        assert worker is not None
        assert worker.busy is False
        assert worker.busy_since is None

    def test_to_dict_exposes_busy_for_seconds(self):
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        reg.assign("w1", "job-1", "sess-1")
        worker = reg.get("w1")
        assert worker is not None
        # Pretend the worker has been busy for 7 minutes.
        worker.busy_since = datetime.now(UTC) - timedelta(minutes=7)
        d = worker.to_dict()
        assert d["busy"] is True
        assert d["busy_since"] is not None
        assert d["busy_for_seconds"] >= 6 * 60  # roughly 7m, allow slop

    def test_to_dict_idle_worker_has_no_duration(self):
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        d = reg.get("w1").to_dict()
        assert d["busy"] is False
        assert d["busy_since"] is None
        assert d["busy_for_seconds"] is None

    def test_reassign_resets_busy_since(self):
        """A worker that finishes one job and picks up another should
        have ``busy_since`` reflect the new job's start, not the previous
        one — otherwise the stuck heuristic would falsely flag fast
        sequential jobs as wedged."""
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        reg.assign("w1", "job-1", "sess-1")
        first = reg.get("w1").busy_since
        assert first is not None
        reg.release("w1")
        # tiny sleep substitute — just advance the clock past microsecond
        # resolution so the second assign produces a strictly-later stamp.
        import time

        time.sleep(0.01)
        reg.assign("w1", "job-2", "sess-2")
        second = reg.get("w1").busy_since
        assert second is not None
        assert second > first


class TestStuckStat:
    def test_no_stuck_when_under_threshold(self):
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        reg.assign("w1", "job-1", "sess-1")
        # Default threshold is 5 minutes — assigning right now means not stuck.
        assert reg.stats()["stuck"] == 0

    def test_stuck_counted_past_threshold(self):
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        reg.register("w2", MagicMock(), {})
        reg.assign("w1", "job-1", "sess-1")
        reg.assign("w2", "job-2", "sess-2")
        # Pretend w1 has been busy for 10 minutes — past the default threshold.
        reg.get("w1").busy_since = datetime.now(UTC) - timedelta(minutes=10)
        stats = reg.stats()
        assert stats["stuck"] == 1
        assert stats["busy"] == 2  # both still counted as busy

    def test_custom_threshold(self):
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        reg.assign("w1", "job-1", "sess-1")
        # 1m old → not stuck at default 5m threshold, but stuck at 30s threshold.
        reg.get("w1").busy_since = datetime.now(UTC) - timedelta(seconds=60)
        assert reg.stats(stuck_threshold_secs=30)["stuck"] == 1
        assert reg.stats(stuck_threshold_secs=300)["stuck"] == 0

    def test_idle_workers_never_stuck(self):
        reg = WorkerRegistry()
        reg.register("w1", MagicMock(), {})
        # Even if busy_since is set on an idle worker (shouldn't happen but
        # defend), stuck only counts genuinely-busy workers.
        worker = reg.get("w1")
        worker.busy_since = datetime.now(UTC) - timedelta(hours=1)
        worker.busy = False
        assert reg.stats()["stuck"] == 0

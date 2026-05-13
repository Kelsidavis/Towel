"""Tests for the idle-task result cache TTL.

Previously :class:`IdleTaskManager` cached results forever, so a LINT report
from six hours ago could be returned to a user who had since fixed every
issue. ``get_result`` and ``all_results`` now honour a per-task TTL derived
from the cooldown table (and ``purge_expired`` evicts the rest on demand).
"""

from __future__ import annotations

import time

from towel.gateway.idle_tasks import (
    _DEFAULT_RESULT_TTL,
    IDLE_TASK_COOLDOWNS,
    IdleTask,
    IdleTaskManager,
    IdleTaskResult,
)


class TestDefaultTTLs:
    def test_ttl_is_at_least_one_cooldown_and_at_most_two_hours(self):
        # The TTL should give results enough time to be useful (longer than
        # cooldown) but never long enough to mask large fleet drift.
        for task, cooldown in IDLE_TASK_COOLDOWNS.items():
            ttl = _DEFAULT_RESULT_TTL[task]
            assert ttl >= cooldown
            assert ttl <= 7200

    def test_every_idle_task_has_a_ttl(self):
        for task in IDLE_TASK_COOLDOWNS:
            assert task in _DEFAULT_RESULT_TTL


class TestGetResult:
    def _seed(self, mgr: IdleTaskManager, task: IdleTask, age_seconds: float) -> None:
        """Inject a result with a synthetic age into the cache."""
        mgr._results[task] = IdleTaskResult(
            task=task,
            worker_id="w1",
            output="(test)",
            timestamp=time.time() - age_seconds,
        )

    def test_fresh_result_is_returned(self):
        mgr = IdleTaskManager()
        self._seed(mgr, IdleTask.LINT, age_seconds=10)
        result = mgr.get_result(IdleTask.LINT)
        assert result is not None
        assert result.output == "(test)"

    def test_stale_result_is_evicted_on_read(self):
        mgr = IdleTaskManager()
        ttl = _DEFAULT_RESULT_TTL[IdleTask.LINT]
        # Older than TTL by a safe margin
        self._seed(mgr, IdleTask.LINT, age_seconds=ttl + 60)
        assert mgr.get_result(IdleTask.LINT) is None
        # And the entry is gone from the cache afterwards.
        assert IdleTask.LINT not in mgr._results

    def test_explicit_max_age_overrides_default(self):
        mgr = IdleTaskManager()
        self._seed(mgr, IdleTask.LINT, age_seconds=120)
        # Default would still return it (TTL > 120s), but a tighter call
        # explicitly demands fresher data.
        assert mgr.get_result(IdleTask.LINT, max_age_seconds=60) is None

    def test_no_result_returns_none(self):
        mgr = IdleTaskManager()
        assert mgr.get_result(IdleTask.LINT) is None


class TestPurgeExpired:
    def test_purge_removes_only_stale_entries(self):
        mgr = IdleTaskManager()
        # Fresh LINT, stale TEST.
        mgr._results[IdleTask.LINT] = IdleTaskResult(
            task=IdleTask.LINT, worker_id="w1", output="ok", timestamp=time.time() - 30
        )
        mgr._results[IdleTask.TEST] = IdleTaskResult(
            task=IdleTask.TEST,
            worker_id="w1",
            output="ok",
            timestamp=time.time() - (_DEFAULT_RESULT_TTL[IdleTask.TEST] + 60),
        )
        removed = mgr.purge_expired()
        assert removed == 1
        assert IdleTask.LINT in mgr._results
        assert IdleTask.TEST not in mgr._results


class TestPeriodicSweeper:
    def test_sweeper_calls_purge_on_each_tick(self):
        """The background sweep loop in ``GatewayServer._sweep_idle_results``
        should call ``IdleTaskManager.purge_expired`` once per interval.

        We patch ``asyncio.sleep`` to fire a CancelledError on the third call
        so the loop runs twice and then exits cleanly, then assert
        ``purge_expired`` was invoked the matching number of times.
        """
        import asyncio
        from unittest.mock import patch

        from towel.config import TowelConfig
        from towel.gateway.server import GatewayServer

        class _FakeAgent:
            pass

        gw = GatewayServer(agent=_FakeAgent(), config=TowelConfig())

        purge_calls: list[int] = []

        def fake_purge() -> int:
            purge_calls.append(1)
            return len(purge_calls)

        gw._idle_manager.purge_expired = fake_purge  # type: ignore[assignment]

        sleep_count = 0

        async def fake_sleep(_interval: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                raise asyncio.CancelledError

        async def runner() -> None:
            with patch("towel.gateway.server.asyncio.sleep", fake_sleep):
                try:
                    await gw._sweep_idle_results(interval=0)
                except asyncio.CancelledError:
                    pass

        asyncio.run(runner())
        # Loop body runs on the 1st and 2nd sleeps; the 3rd raises before
        # purge_expired is reached. So we expect exactly 2 invocations.
        assert len(purge_calls) == 2


class TestAllResultsHidesStale:
    def test_all_results_filters_stale_entries(self):
        mgr = IdleTaskManager()
        mgr._results[IdleTask.LINT] = IdleTaskResult(
            task=IdleTask.LINT, worker_id="w1", output="fresh", timestamp=time.time() - 10
        )
        mgr._results[IdleTask.TEST] = IdleTaskResult(
            task=IdleTask.TEST,
            worker_id="w1",
            output="ancient",
            timestamp=time.time() - (_DEFAULT_RESULT_TTL[IdleTask.TEST] + 60),
        )
        live = mgr.all_results()
        assert str(IdleTask.LINT) in live
        assert str(IdleTask.TEST) not in live

"""Tests for the fleet-endpoint probes in ``towel doctor``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from towel.cli.doctor import Check, _probe_fleet_endpoints


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


class TestProbeFleetEndpoints:
    def test_workers_summary_when_fleet_has_idle_busy_and_hot(self):
        responses = {
            "/workers": {
                "workers": [
                    {
                        "id": "w-cool", "busy": False, "enabled": True, "draining": False,
                        "quality_tier": "high",
                        "capabilities": {
                            "live_resources": {"cpu_pressure": 0.1},
                            "max_param_b_est": 70.0,
                        },
                    },
                    {
                        "id": "w-busy", "busy": True, "enabled": True, "draining": False,
                        "quality_tier": "medium",
                        "capabilities": {
                            "live_resources": {"cpu_pressure": 0.5},
                            "max_param_b_est": 13.0,
                        },
                    },
                    {
                        "id": "w-hot", "busy": False, "enabled": True, "draining": False,
                        "quality_tier": "low",
                        "capabilities": {
                            "live_resources": {"cpu_pressure": 0.9},
                            "max_param_b_est": 3.0,
                        },
                    },
                ],
            },
            "/skills": {"skills": [{"name": "fs"}, {"name": "git"}], "total_tools": 7},
            "/fleet/inventory": {
                "models": [
                    {"name": "qwen3.6:27b", "workers": ["w-cool", "w-busy"], "cached_count": 2},
                    {"name": "haiku", "workers": ["w-hot"], "cached_count": 1},
                ],
                "total_unique": 2,
                "total_workers": 3,
                "fleet_max_param_b": 70.0,
            },
            "/dispatch/recent?limit=1": {
                "decisions": [{"reason": "task_type_match", "worker_id": "w-cool"}],
            },
            "/cluster/handoffs": {"stats": {"total": 0, "failed": 0, "pending": 0}, "recent": []},
        }

        def fake_get(url, timeout=None):
            for suffix, payload in responses.items():
                if url.endswith(suffix):
                    return _mock_response(payload)
            raise AssertionError(f"unexpected url: {url}")

        c = Check("test")
        with patch("httpx.get", side_effect=fake_get):
            _probe_fleet_endpoints(c, "localhost", 18743)

        joined = " | ".join(c.details)
        # Workers summary should mention idle, busy, and hot counts.
        # Both "cool" and "hot" workers have busy=False so both are idle —
        # cpu_pressure ≥ 0.8 marks the second as additionally "hot".
        assert "Workers: 3" in joined
        assert "2 idle" in joined
        assert "1 busy" in joined
        assert "1 hot" in joined
        # Tier breakdown rolls up to a single line.
        assert "Tiers: 1 high, 1 medium, 1 low" in joined
        # Size range: smallest 3B, largest 70B.
        assert "Fits: ~3.0B" in joined
        assert "70.0B params" in joined
        # Inventory rolls up to a one-liner with the most-replicated model.
        assert "Inventory: 2 unique model(s)" in joined
        assert "qwen3.6:27b" in joined
        assert "2× cached" in joined
        # Skills count.
        assert "Skills: 2 loaded (7 tools available)" in joined
        # Last dispatch decision.
        assert "Last dispatch: task_type_match → w-cool" in joined
        # No warnings on the happy path.
        assert c.warnings == []

    def test_uniform_fleet_says_every_worker_fits_the_same(self):
        responses = {
            "/workers": {
                "workers": [
                    {
                        "id": "a", "busy": False, "enabled": True, "draining": False,
                        "quality_tier": "high",
                        "capabilities": {"max_param_b_est": 32.0},
                    },
                    {
                        "id": "b", "busy": False, "enabled": True, "draining": False,
                        "quality_tier": "high",
                        "capabilities": {"max_param_b_est": 32.0},
                    },
                ],
            },
            "/skills": {"skills": [], "total_tools": 0},
            "/fleet/inventory": {
                "models": [], "total_unique": 0, "total_workers": 2,
                "fleet_max_param_b": 32.0,
            },
            "/dispatch/recent?limit=1": {"decisions": []},
            "/cluster/handoffs": {"stats": {"total": 0, "failed": 0, "pending": 0}, "recent": []},
        }

        def fake_get(url, timeout=None):
            for suffix, payload in responses.items():
                if url.endswith(suffix):
                    return _mock_response(payload)
            raise AssertionError(f"unexpected url: {url}")

        c = Check("test")
        with patch("httpx.get", side_effect=fake_get):
            _probe_fleet_endpoints(c, "localhost", 18743)

        joined = " | ".join(c.details)
        assert "Tiers: 2 high" in joined
        # Same size on every worker uses the singular phrasing.
        assert "up to ~32.0B params on every worker" in joined
        # No cached models reported → friendly "no cached models" line.
        assert "Inventory: no cached models reported" in joined

    def test_empty_fleet_does_not_emit_hot_count(self):
        responses = {
            "/workers": {"workers": []},
            "/skills": {"skills": [], "total_tools": 0},
            "/fleet/inventory": {
                "models": [], "total_unique": 0, "total_workers": 0,
                "fleet_max_param_b": 0.0,
            },
            "/dispatch/recent?limit=1": {"decisions": []},
            "/cluster/handoffs": {"stats": {"total": 0, "failed": 0, "pending": 0}, "recent": []},
        }

        def fake_get(url, timeout=None):
            for suffix, payload in responses.items():
                if url.endswith(suffix):
                    return _mock_response(payload)
            raise AssertionError(f"unexpected url: {url}")

        c = Check("test")
        with patch("httpx.get", side_effect=fake_get):
            _probe_fleet_endpoints(c, "localhost", 18743)

        joined = " | ".join(c.details)
        assert "Workers: none connected" in joined
        assert "Skills: 0 loaded (0 tools available)" in joined
        assert "Dispatch log: empty" in joined

    def test_endpoint_failure_becomes_a_warning_not_an_exception(self):
        # Each probe is wrapped in try/except — a 404, a connection refused,
        # or any other failure must degrade to a warn rather than crash the
        # whole doctor.
        c = Check("test")
        with patch("httpx.get", side_effect=Exception("boom")):
            _probe_fleet_endpoints(c, "localhost", 18743)
        # One warning per probe (workers, skills, inventory, dispatch, handoffs).
        assert len(c.warnings) == 5
        assert all("probe failed" in w for w in c.warnings)

    def test_garbage_cpu_pressure_does_not_count_as_hot(self):
        # Defensive: a worker reporting a non-numeric cpu_pressure shouldn't
        # be counted as hot or crash the summary.
        responses = {
            "/workers": {
                "workers": [
                    {
                        "id": "garbage", "busy": False, "enabled": True, "draining": False,
                        "capabilities": {"live_resources": {"cpu_pressure": "huge"}},
                    },
                ],
            },
            "/skills": {"skills": [], "total_tools": 0},
            "/fleet/inventory": {
                "models": [], "total_unique": 0, "total_workers": 1,
                "fleet_max_param_b": 0.0,
            },
            "/dispatch/recent?limit=1": {"decisions": []},
            "/cluster/handoffs": {"stats": {"total": 0, "failed": 0, "pending": 0}, "recent": []},
        }

        def fake_get(url, timeout=None):
            for suffix, payload in responses.items():
                if url.endswith(suffix):
                    return _mock_response(payload)
            raise AssertionError(f"unexpected url: {url}")

        c = Check("test")
        with patch("httpx.get", side_effect=fake_get):
            _probe_fleet_endpoints(c, "localhost", 18743)

        joined = " | ".join(c.details)
        assert "Workers: 1" in joined
        # No "hot" count when the only worker's pressure value is unusable.
        assert "hot" not in joined

    def test_handoff_failures_become_a_warning(self):
        """A nonzero failed-handoff count is exactly the kind of thing
        operators don't notice until something breaks. Doctor surfaces it
        as a warn so the line shows up next to a yellow WARN icon."""
        responses = {
            "/workers": {"workers": []},
            "/skills": {"skills": [], "total_tools": 0},
            "/fleet/inventory": {
                "models": [], "total_unique": 0, "total_workers": 0,
                "fleet_max_param_b": 0.0,
            },
            "/dispatch/recent?limit=1": {"decisions": []},
            "/cluster/handoffs": {
                "stats": {
                    "total": 7,
                    "successful": 5,
                    "failed": 2,
                    "pending": 0,
                    "avg_duration_ms": 120.5,
                },
                "recent": [],
            },
        }

        def fake_get(url, timeout=None):
            for suffix, payload in responses.items():
                if url.endswith(suffix):
                    return _mock_response(payload)
            raise AssertionError(f"unexpected url: {url}")

        c = Check("test")
        with patch("httpx.get", side_effect=fake_get):
            _probe_fleet_endpoints(c, "localhost", 18743)

        joined_details = " | ".join(c.details)
        joined_warnings = " | ".join(c.warnings)
        assert "Handoffs: 7 total" in joined_details
        assert "120.5ms" in joined_details
        # The 2 failed migrations earn a yellow warn, not silent success.
        assert "2 handoff(s) failed" in joined_warnings

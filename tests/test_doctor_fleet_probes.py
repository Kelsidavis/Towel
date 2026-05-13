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
                        "capabilities": {"live_resources": {"cpu_pressure": 0.1}},
                    },
                    {
                        "id": "w-busy", "busy": True, "enabled": True, "draining": False,
                        "capabilities": {"live_resources": {"cpu_pressure": 0.5}},
                    },
                    {
                        "id": "w-hot", "busy": False, "enabled": True, "draining": False,
                        "capabilities": {"live_resources": {"cpu_pressure": 0.9}},
                    },
                ],
            },
            "/skills": {"skills": [{"name": "fs"}, {"name": "git"}], "total_tools": 7},
            "/dispatch/recent?limit=1": {
                "decisions": [{"reason": "task_type_match", "worker_id": "w-cool"}],
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

        joined = " | ".join(c.details)
        # Workers summary should mention idle, busy, and hot counts.
        # Both "cool" and "hot" workers have busy=False so both are idle —
        # cpu_pressure ≥ 0.8 marks the second as additionally "hot".
        assert "Workers: 3" in joined
        assert "2 idle" in joined
        assert "1 busy" in joined
        assert "1 hot" in joined
        # Skills count.
        assert "Skills: 2 loaded (7 tools available)" in joined
        # Last dispatch decision.
        assert "Last dispatch: task_type_match → w-cool" in joined
        # No warnings on the happy path.
        assert c.warnings == []

    def test_empty_fleet_does_not_emit_hot_count(self):
        responses = {
            "/workers": {"workers": []},
            "/skills": {"skills": [], "total_tools": 0},
            "/dispatch/recent?limit=1": {"decisions": []},
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
        assert len(c.warnings) == 3
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
            "/dispatch/recent?limit=1": {"decisions": []},
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

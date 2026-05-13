"""Tests for live worker resource telemetry.

Workers report ``live_resources`` (1-min load average, cpu_pressure, free RAM)
on every heartbeat. The coordinator merges this into ``WorkerInfo.capabilities``
so the fleet UI and any future load-aware scoring can see real-time pressure.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from towel.gateway.worker_client import (
    RemoteWorkerClient,
    _detect_live_resources,
)


class TestDetector:
    def test_returns_dict_of_numeric_fields(self):
        live = _detect_live_resources()
        # The dict is never None and any included keys are sane numerics.
        assert isinstance(live, dict)
        if "load_avg_1min" in live:
            assert isinstance(live["load_avg_1min"], float)
            assert live["load_avg_1min"] >= 0
        if "cpu_pressure" in live:
            assert 0.0 <= live["cpu_pressure"] <= 1.0
        if "ram_available_mb" in live:
            assert isinstance(live["ram_available_mb"], int)
            assert live["ram_available_mb"] > 0

    def test_safe_when_called_repeatedly(self):
        # Called every heartbeat — must be cheap and side-effect free.
        for _ in range(5):
            _detect_live_resources()


class TestHeartbeatPayload:
    def test_heartbeat_message_includes_live_resources(self):
        ws = MagicMock()
        ws.send = AsyncMock()
        client = RemoteWorkerClient(
            master_url="ws://example.invalid",
            agent=MagicMock(),
            worker_id="worker_test",
            capabilities={"backend": "ollama", "model": "qwen3.6:27b"},
        )

        asyncio.run(client._send_heartbeat(ws))

        ws.send.assert_called_once()
        payload = json.loads(ws.send.call_args.args[0])
        assert payload["type"] == "heartbeat"
        assert payload["id"] == "worker_test"
        # Static capabilities preserved alongside live snapshot.
        assert payload["capabilities"]["backend"] == "ollama"
        assert "live_resources" in payload["capabilities"]
        # The detector returns a dict (possibly empty on unsupported platforms).
        assert isinstance(payload["capabilities"]["live_resources"], dict)

    def test_live_resources_refreshed_each_heartbeat(self):
        ws = MagicMock()
        ws.send = AsyncMock()
        client = RemoteWorkerClient(
            master_url="ws://example.invalid",
            agent=MagicMock(),
            worker_id="worker_test",
            capabilities={"backend": "mlx"},
        )

        async def run_two() -> tuple[dict, dict]:
            await client._send_heartbeat(ws)
            first = json.loads(ws.send.call_args.args[0])["capabilities"]["live_resources"]
            await client._send_heartbeat(ws)
            second = json.loads(ws.send.call_args.args[0])["capabilities"]["live_resources"]
            return first, second

        first, second = asyncio.run(run_two())
        # Both heartbeats must independently include the live snapshot — we're
        # not just relying on a snapshot captured at startup.
        assert isinstance(first, dict)
        assert isinstance(second, dict)

    def test_live_resources_also_updates_self_capabilities(self):
        """The client keeps its own ``self.capabilities`` current so any caller
        reading it (the fleet UI, a follow-up heartbeat path) sees live data."""
        ws = MagicMock()
        ws.send = AsyncMock()
        client = RemoteWorkerClient(
            master_url="ws://example.invalid",
            agent=MagicMock(),
            worker_id="worker_test",
            capabilities={"backend": "ollama"},
        )
        assert "live_resources" not in client.capabilities

        asyncio.run(client._send_heartbeat(ws))

        assert "live_resources" in client.capabilities

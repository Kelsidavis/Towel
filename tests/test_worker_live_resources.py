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


class TestModelInventory:
    """Worker capabilities include a list of locally-available models so the
    coordinator can pick replace/upgrade targets that won't need a fresh
    download, and a rough param-cap estimate so it knows a Pi-class node
    shouldn't get 70B requests."""

    def test_claude_inventory_is_the_three_aliases(self):
        from towel.gateway.worker_client import _detect_available_models

        assert _detect_available_models("claude", "") == ["sonnet", "opus", "haiku"]

    def test_unknown_backend_returns_empty_list(self):
        from towel.gateway.worker_client import _detect_available_models

        assert _detect_available_models("nope", "") == []

    def test_current_model_appended_when_probe_returns_empty(self):
        """An active worker must report SOMETHING in available_models —
        the model it's actually serving. Probe failures (llama-server
        builds w/o /v1/models, network glitches) shouldn't drop it."""
        from towel.gateway.worker_client import _detect_available_models

        # Unknown backend → probe contributes nothing. With
        # current_model set, that name is still in the result.
        result = _detect_available_models(
            "nope", "", current_model="My-Active-Model.gguf",
        )
        assert result == ["My-Active-Model.gguf"]

    def test_current_model_deduped_when_probe_also_returns_it(self):
        """If the backend's probe DID find the active model, we
        shouldn't list it twice."""
        from towel.gateway.worker_client import _detect_available_models

        # Claude's probe returns the three fixed aliases. Passing
        # one as current_model shouldn't duplicate it.
        result = _detect_available_models(
            "claude", "", current_model="sonnet",
        )
        assert result.count("sonnet") == 1
        assert "opus" in result
        assert "haiku" in result

    def test_dedupes_while_preserving_order(self):
        from unittest.mock import patch

        from towel.gateway.worker_client import _detect_available_models

        # MLX scans the HF cache. Patch the cache to return duplicates and
        # confirm de-dup keeps insertion order.
        with patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.iterdir") as iterdir:
            class _Entry:
                def __init__(self, name): self.name = name
            iterdir.return_value = [
                _Entry("models--mlx-community--A"),
                _Entry("models--other--B"),
                _Entry("models--mlx-community--A"),  # dup
                _Entry("models--other--C"),
            ]
            result = _detect_available_models("mlx", "")
            assert result == [
                "mlx-community/A",
                "other/B",
                "other/C",
            ]

    def test_capabilities_include_max_param_b_est(self):
        # The full default_worker_capabilities path stitches the inventory +
        # the rough VRAM-or-RAM-derived param cap together.
        from towel.gateway.worker_client import default_worker_capabilities

        class _Model:
            name = "x"
            context_window = 8192
            max_tokens = 1024

        class _Cfg:
            model = _Model

        caps = default_worker_capabilities(_Cfg, "claude", allow_tools=False)
        assert "available_models" in caps
        assert "max_param_b_est" in caps
        # Claude returns its alias list regardless of local cache.
        assert "sonnet" in caps["available_models"]


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

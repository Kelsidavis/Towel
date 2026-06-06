"""Tests for the llama.cpp runtime adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from towel.agent.discovery import GGUFModel
from towel.agent.llama_runtime import LlamaRuntime
from towel.config import TowelConfig


def test_auto_start_disables_llama_fit(monkeypatch):
    created: dict[str, object] = {}

    class FakeManagedLlamaServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.url = f"http://localhost:{kwargs['port']}"

        def start(self):
            created["started"] = True

        async def wait_healthy(self):
            created["waited"] = True

    def fake_detect_system():
        return SimpleNamespace(
            has_llama_server=True,
            llama_server_path="/usr/bin/llama-server",
            best_model=GGUFModel(
                path=Path("/models/qwen.gguf"),
                size_gb=14.5,
                name="qwen",
            ),
        )

    monkeypatch.setattr(
        "towel.agent.discovery.ManagedLlamaServer",
        FakeManagedLlamaServer,
    )
    monkeypatch.setattr(
        "towel.agent.discovery.detect_system",
        fake_detect_system,
    )

    runtime = LlamaRuntime(TowelConfig(), llama_url="http://localhost:18081")
    monkeypatch.setattr(runtime, "_check_health", lambda: asyncio.sleep(0, result=False))

    asyncio.run(runtime.load_model())

    assert created["extra_args"] == ["--fit", "off", "-c", "32768"]
    assert created["started"] is True
    assert created["waited"] is True

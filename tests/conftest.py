"""Shared pytest fixtures and safety guards for the Towel test suite."""

from __future__ import annotations

import pytest

from towel.agent.runtime import AgentRuntime


@pytest.fixture(autouse=True)
def _block_real_model_download(monkeypatch):
    """Never let a test trigger a real ``mlx_lm`` model download.

    The default configured model is an 80B HF repo. A test that accidentally
    reaches ``AgentRuntime._load_model_sync`` (e.g. a gateway endpoint test
    that falls back to the local MLX agent because no worker is connected)
    would call ``mlx_lm.load`` -> ``huggingface_hub.snapshot_download`` and
    hang on the network until the per-test timeout fires.

    On a dev box without ``mlx`` installed this already fails fast (ImportError)
    which is why the suite is green there; on CI runners that *do* have ``mlx``
    it would otherwise wedge. Make the load fail fast and deterministically so
    both environments behave identically. Tests that genuinely need generation
    stub ``agent.generate`` / ``agent.step`` / ``agent.stream`` directly and so
    never reach this method; tests that specifically exercise loading can
    re-patch it after this fixture runs.
    """

    def _blocked(self: AgentRuntime):  # pragma: no cover - guard, not behavior
        raise RuntimeError(
            "Real model loading is disabled under pytest (would download the "
            "configured model). Stub agent.generate/step/stream instead."
        )

    monkeypatch.setattr(AgentRuntime, "_load_model_sync", _blocked, raising=False)

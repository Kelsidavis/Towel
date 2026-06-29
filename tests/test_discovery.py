"""Tests for hardware/model discovery helpers."""

from __future__ import annotations

from pathlib import Path

from towel.agent.discovery import GGUFModel, SystemCapabilities


def test_best_model_prefers_abliterated_over_stock_and_coder():
    # Towel runs abliterated chat models; a stock safety-trained model
    # (gpt-oss) or a larger task-specific coder model must NOT be picked
    # over an abliterated build that fits.
    caps = SystemCapabilities(
        total_vram_mb=20480,
        gguf_models=[
            GGUFModel(
                path=Path("/models/Huihui-Qwen3.6-27B-abliterated.Q4_K_S.gguf"),
                size_gb=15.59,
                name="Huihui-Qwen3.6-27B-abliterated.Q4_K_S",
            ),
            GGUFModel(
                path=Path("/models/gpt-oss-20b-F16.gguf"),
                size_gb=13.79,
                name="gpt-oss-20b-F16",
            ),
            # Larger and fits, but a coder model — must lose to abliterated.
            GGUFModel(
                path=Path("/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"),
                size_gb=17.67,
                name="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL",
            ),
        ],
    )

    assert caps.best_model is not None
    assert caps.best_model.name == "Huihui-Qwen3.6-27B-abliterated.Q4_K_S"


def test_best_model_falls_back_to_largest_when_none_abliterated():
    caps = SystemCapabilities(
        total_vram_mb=20480,
        gguf_models=[
            GGUFModel(
                path=Path("/models/gpt-oss-20b-F16.gguf"),
                size_gb=13.79,
                name="gpt-oss-20b-F16",
            ),
            GGUFModel(
                path=Path("/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"),
                size_gb=17.67,
                name="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL",
            ),
        ],
    )

    assert caps.best_model is not None
    assert caps.best_model.name == "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL"

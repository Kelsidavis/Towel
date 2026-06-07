"""Tests for hardware/model discovery helpers."""

from __future__ import annotations

from pathlib import Path

from towel.agent.discovery import GGUFModel, SystemCapabilities


def test_best_model_prefers_gpt_oss_over_qwen35_family():
    caps = SystemCapabilities(
        total_vram_mb=20480,
        gguf_models=[
            GGUFModel(
                path=Path("/models/Huihui-Qwen3.6-27B-abliterated.Q4_K_S.gguf"),
                size_gb=14.52,
                name="Huihui-Qwen3.6-27B-abliterated.Q4_K_S",
            ),
            GGUFModel(
                path=Path("/models/gpt-oss-20b-F16.gguf"),
                size_gb=13.07,
                name="gpt-oss-20b-F16",
            ),
            GGUFModel(
                path=Path("/models/Qwen3.5-27B-Claude-4.6-OS-INSTRUCT.i1-Q3_K_L.gguf"),
                size_gb=13.07,
                name="Qwen3.5-27B-Claude-4.6-OS-INSTRUCT.i1-Q3_K_L",
            ),
        ],
    )

    assert caps.best_model is not None
    assert caps.best_model.name == "gpt-oss-20b-F16"

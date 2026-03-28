"""Model management — list, inspect, and pull MLX models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from towel.config import TowelConfig, DEFAULT_AGENTS

log = logging.getLogger("towel.models")
console = Console()

# Well-known MLX models with metadata for recommendations
RECOMMENDED_MODELS = [
    {
        "name": "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "params": "70B",
        "quant": "4-bit",
        "ram": "~40 GB",
        "use": "General, research",
    },
    {
        "name": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "params": "32B",
        "quant": "4-bit",
        "ram": "~20 GB",
        "use": "Code generation",
    },
    {
        "name": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "params": "32B",
        "quant": "4-bit",
        "ram": "~20 GB",
        "use": "General purpose",
    },
    {
        "name": "mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
        "params": "24B",
        "quant": "4-bit",
        "ram": "~14 GB",
        "use": "Fast general",
    },
    {
        "name": "mlx-community/Llama-3.2-8B-Instruct-4bit",
        "params": "8B",
        "quant": "4-bit",
        "ram": "~5 GB",
        "use": "Lightweight, fast",
    },
    {
        "name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "params": "7B",
        "quant": "4-bit",
        "ram": "~5 GB",
        "use": "Lightweight general",
    },
    {
        "name": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "params": "3B",
        "quant": "4-bit",
        "ram": "~2 GB",
        "use": "Ultra-light, testing",
    },
]


def get_hf_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def list_cached_models() -> list[CachedModel]:
    """Scan the HuggingFace cache for downloaded models."""
    hf_cache = get_hf_cache_dir()
    if not hf_cache.exists():
        return []

    models: list[CachedModel] = []
    for entry in sorted(hf_cache.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue

        name = entry.name.replace("models--", "").replace("--", "/")

        # Calculate size
        try:
            total_size = sum(
                f.stat().st_size for f in entry.rglob("*") if f.is_file()
            )
        except OSError:
            total_size = 0

        models.append(CachedModel(
            name=name,
            path=entry,
            size_bytes=total_size,
        ))

    return models


def is_model_cached(model_name: str) -> bool:
    """Check if a specific model is in the local cache."""
    slug = model_name.replace("/", "--")
    return (get_hf_cache_dir() / f"models--{slug}").exists()


def get_model_usage(config: TowelConfig) -> dict[str, list[str]]:
    """Map model names to which agent profiles use them."""
    usage: dict[str, list[str]] = {}
    # Default model
    usage.setdefault(config.model.name, []).append("default")
    # Agent profiles
    for agent_name, profile in config.list_agents().items():
        usage.setdefault(profile.model.name, []).append(agent_name)
    return usage


class CachedModel:
    """A locally cached model."""

    def __init__(self, name: str, path: Path, size_bytes: int) -> None:
        self.name = name
        self.path = path
        self.size_bytes = size_bytes

    @property
    def size_display(self) -> str:
        gb = self.size_bytes / (1024 ** 3)
        if gb >= 1:
            return f"{gb:.1f} GB"
        mb = self.size_bytes / (1024 ** 2)
        return f"{mb:.0f} MB"

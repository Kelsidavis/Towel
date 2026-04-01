"""Towel doctor — diagnose your setup and find problems before they find you.

Checks environment, configuration, model availability, skills, and gateway.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
from pathlib import Path

from rich.console import Console

from towel.config import TOWEL_HOME, TowelConfig

console = Console()


class Check:
    """Result of a single diagnostic check."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = False
        self.details: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.suggestions: list[str] = []

    def ok(self, detail: str) -> None:
        self.details.append(detail)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str, suggestion: str | None = None) -> None:
        self.errors.append(msg)
        if suggestion:
            self.suggestions.append(suggestion)

    def finalize(self) -> None:
        self.passed = len(self.errors) == 0

    def render(self) -> None:
        self.finalize()
        icon = "[green]OK[/green]" if self.passed else "[red]FAIL[/red]"
        if not self.passed:
            icon = "[red]FAIL[/red]"
        elif self.warnings:
            icon = "[yellow]WARN[/yellow]"

        console.print(f"\n  {icon}  [bold]{self.name}[/bold]")
        for d in self.details:
            console.print(f"       [dim]{d}[/dim]")
        for w in self.warnings:
            console.print(f"       [yellow]{w}[/yellow]")
        for e in self.errors:
            console.print(f"       [red]{e}[/red]")
        for s in self.suggestions:
            console.print(f"       [cyan]-> {s}[/cyan]")


def run_doctor(config: TowelConfig | None = None) -> list[Check]:
    """Run all diagnostic checks. Returns list of Check results."""
    config = config or TowelConfig.load()
    checks: list[Check] = []

    checks.append(check_environment())
    checks.append(check_config(config))
    checks.append(check_mlx())
    checks.append(check_model(config))
    checks.append(check_skills(config))
    checks.append(check_gateway(config))
    checks.append(check_storage())

    return checks


def check_environment() -> Check:
    """Check Python version, platform, and memory."""
    c = Check("Environment")

    # Python version
    v = sys.version_info
    c.ok(f"Python {v.major}.{v.minor}.{v.micro}")
    if v < (3, 11):
        c.fail(
            f"Python {v.major}.{v.minor} is below minimum (3.11)",
            "Install Python 3.11+ from python.org or via Homebrew",
        )

    # Platform
    machine = platform.machine()
    c.ok(f"{platform.system()} {platform.release()} ({machine})")
    if machine != "arm64" and platform.system() == "Darwin":
        c.warn("Not Apple Silicon — MLX performance will be limited")

    # Memory (macOS)
    try:
        import resource
        # Get available memory via os.sysconf on macOS/Linux
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_gb = (pages * page_size) / (1024 ** 3)
        c.ok(f"{total_gb:.0f} GB system memory")
        if total_gb < 8:
            c.warn("Less than 8 GB RAM — larger models may not fit")
    except (ValueError, OSError, ImportError):
        c.ok("Memory: could not determine")

    # Disk space for ~/.towel
    try:
        usage = shutil.disk_usage(TOWEL_HOME.parent)
        free_gb = usage.free / (1024 ** 3)
        c.ok(f"{free_gb:.1f} GB free disk space")
        if free_gb < 5:
            c.warn("Less than 5 GB free — model downloads may fail")
    except OSError:
        pass

    return c


def check_config(config: TowelConfig) -> Check:
    """Check configuration file and settings."""
    c = Check("Configuration")

    config_path = TOWEL_HOME / "config.toml"
    if config_path.exists():
        c.ok(f"Config: {config_path}")
    else:
        c.warn(f"No config file at {config_path} (using defaults)")
        c.suggestions.append("Run: towel init")

    c.ok(f"Model: {config.model.name}")
    c.ok(f"Context window: {config.model.context_window} tokens")
    c.ok(f"Max output: {config.model.max_tokens} tokens")
    if config.model.turboquant:
        c.ok(f"KV cache: TurboQuant {config.model.turboquant_bits}-bit (QJL ratio {config.model.turboquant_qjl_ratio})")
    else:
        c.ok("KV cache: float16 (standard)")
    c.ok(f"Gateway: {config.gateway.host}:{config.gateway.port}")

    if config.model.context_window <= config.model.max_tokens:
        c.fail(
            f"context_window ({config.model.context_window}) must be larger than max_tokens ({config.model.max_tokens})",
            "Increase context_window in config.toml",
        )

    # Check skills directories
    for d in config.skills_dirs:
        p = Path(d).expanduser()
        if p.exists():
            c.ok(f"Skills dir: {p}")
        else:
            c.ok(f"Skills dir: {p} [dim](not created yet)[/dim]")

    return c


def check_mlx() -> Check:
    """Check MLX installation and capabilities."""
    c = Check("MLX")

    # Core MLX
    try:
        import mlx
        import mlx.core as mx
        version = getattr(mlx, "__version__", None) or getattr(mx, "__version__", None)
        c.ok(f"mlx {version or '(version unknown)'}")
    except ImportError:
        c.fail("mlx not installed", "Run: pip install mlx")
        return c
    except Exception as e:
        c.warn(f"mlx issue: {e}")

    # mlx_lm
    try:
        import mlx_lm
        version = getattr(mlx_lm, "__version__", "(version unknown)")
        c.ok(f"mlx-lm {version}")
    except ImportError:
        c.fail("mlx-lm not installed", "Run: pip install mlx-lm")

    # Transformers (needed for tokenizers)
    try:
        import transformers
        c.ok(f"transformers {transformers.__version__}")
    except ImportError:
        c.fail("transformers not installed", "Run: pip install transformers")

    return c


def check_model(config: TowelConfig) -> Check:
    """Check if the configured model is available."""
    c = Check("Model")

    model_name = config.model.name
    c.ok(f"Configured: {model_name}")

    # Check HuggingFace cache for downloaded models
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    mlx_cache = Path.home() / ".cache" / "mlx"

    # Check common cache locations
    cached_models: list[str] = []
    for cache_dir in [hf_cache, mlx_cache]:
        if cache_dir.exists():
            for entry in cache_dir.iterdir():
                if entry.is_dir():
                    name = entry.name.replace("models--", "").replace("--", "/")
                    if name and not name.startswith("."):
                        cached_models.append(name)

    # Check if our model is cached
    model_slug = model_name.replace("/", "--")
    hf_model_dir = hf_cache / f"models--{model_slug}"

    if hf_model_dir.exists():
        c.ok("Model is cached locally")
        # Check approximate size
        try:
            total_size = sum(
                f.stat().st_size for f in hf_model_dir.rglob("*") if f.is_file()
            )
            size_gb = total_size / (1024 ** 3)
            c.ok(f"Cache size: {size_gb:.1f} GB")
        except OSError:
            pass
    else:
        c.warn("Model not cached locally — first run will download it")
        c.suggestions.append(f"Pre-download: python -c \"from mlx_lm import load; load('{model_name}')\"")

    # Suggest smaller alternatives if relevant
    small_models = [m for m in cached_models if any(
        q in m.lower() for q in ["4bit", "8bit", "3b", "7b", "1b"]
    )]
    if small_models and not hf_model_dir.exists():
        c.ok(f"Locally cached models: {len(cached_models)}")
        for m in small_models[:5]:
            c.ok(f"  Available: {m}")

    return c


def check_skills(config: TowelConfig) -> Check:
    """Check skill loading."""
    c = Check("Skills")

    from towel.skills.builtin import register_builtins
    from towel.skills.loader import SkillLoader
    from towel.skills.registry import SkillRegistry

    registry = SkillRegistry()

    # Built-in skills
    try:
        register_builtins(registry)
        builtin_names = registry.list_skills()
        c.ok(f"Built-in skills: {', '.join(builtin_names)} ({len(builtin_names)} skills, {len(registry.tool_definitions())} tools)")
    except Exception as e:
        c.fail(f"Failed to load built-in skills: {e}")

    # User skills
    loader = SkillLoader(registry)
    loaded = loader.load_from_dirs(config.skills_dirs)
    if loaded:
        user_names = [n for n in registry.list_skills() if n not in builtin_names]
        c.ok(f"User skills: {', '.join(user_names)} ({loaded} loaded)")

    for err in loader.errors:
        c.warn(f"Skill load error: {err.path.name}: {err.error}")

    return c


def check_gateway(config: TowelConfig) -> Check:
    """Check gateway port availability and running status."""
    c = Check("Gateway")

    host = config.gateway.host
    ws_port = config.gateway.port
    http_port = ws_port + 1

    # Check if gateway is already running
    try:
        import httpx
        resp = httpx.get(f"http://{host}:{http_port}/health", timeout=2)
        data = resp.json()
        c.ok(f"Gateway is running ({data.get('status', 'unknown')})")
        c.ok(f"WebSocket: ws://{host}:{ws_port}")
        c.ok(f"HTTP API: http://{host}:{http_port}")
        c.ok(f"Web UI: http://{host}:{http_port}/")
        c.ok(f"Connections: {data.get('connections', '?')}, Sessions: {data.get('sessions', '?')}")
        return c
    except Exception:
        pass

    # Gateway not running — check if ports are available
    for port, label in [(ws_port, "WebSocket"), (http_port, "HTTP")]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                c.warn(f"{label} port {port} is in use by another process")
                c.suggestions.append("Change gateway port in config or stop the other process")
            else:
                c.ok(f"{label} port {port} is available")
        except OSError:
            c.ok(f"{label} port {port} is available")

    c.ok("Gateway not running (start with: towel serve)")

    return c


def check_storage() -> Check:
    """Check conversation storage."""
    c = Check("Storage")

    conv_dir = TOWEL_HOME / "conversations"
    if conv_dir.exists():
        count = len(list(conv_dir.glob("*.json")))
        c.ok(f"Conversations: {count} saved in {conv_dir}")

        # Check total size
        try:
            total = sum(f.stat().st_size for f in conv_dir.glob("*.json"))
            if total > 100 * 1024 * 1024:
                c.warn(f"Conversation storage is {total / (1024**2):.0f} MB — consider cleanup")
            else:
                c.ok(f"Storage size: {total / 1024:.0f} KB")
        except OSError:
            pass
    else:
        c.ok("No conversations stored yet")

    # Check TOWEL_HOME
    if TOWEL_HOME.exists():
        c.ok(f"Towel home: {TOWEL_HOME}")
    else:
        c.ok(f"Towel home not initialized ({TOWEL_HOME})")
        c.suggestions.append("Run: towel init")

    return c

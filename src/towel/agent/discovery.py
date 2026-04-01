"""Hardware and model auto-discovery for zero-config worker setup.

Detects NVIDIA GPUs, finds llama-server binaries, scans for GGUF models,
and can auto-start llama-server as a managed subprocess.
"""

from __future__ import annotations

import asyncio
import glob
import logging
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("towel.agent.discovery")

# Directories to scan for GGUF models (in priority order)
GGUF_SEARCH_DIRS = [
    Path.home() / ".towel" / "models",
    Path.home() / "models",
    Path("/mnt"),
    Path.home() / "Downloads",
    Path.home() / "downloads",
    Path.home() / ".cache" / "huggingface",
]

# Additional paths to search for llama-server binary
LLAMA_SERVER_SEARCH_PATHS = [
    "/tmp/llama.cpp/build/bin/llama-server",
    str(Path.home() / ".local" / "bin" / "llama-server"),
    "/usr/local/bin/llama-server",
]


@dataclass
class GPUInfo:
    """Detected NVIDIA GPU."""

    index: int
    name: str
    vram_mb: int
    compute_capability: str = ""


@dataclass
class GGUFModel:
    """A discovered GGUF model file on disk."""

    path: Path
    size_gb: float
    name: str  # filename stem


@dataclass
class SystemCapabilities:
    """Auto-detected system capabilities."""

    gpus: list[GPUInfo] = field(default_factory=list)
    total_vram_mb: int = 0
    llama_server_path: str | None = None
    gguf_models: list[GGUFModel] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def has_llama_server(self) -> bool:
        return self.llama_server_path is not None

    @property
    def best_model(self) -> GGUFModel | None:
        """Pick the largest model that fits in VRAM (with ~2GB overhead margin)."""
        if not self.gguf_models:
            return None
        usable_vram_gb = (self.total_vram_mb - 2048) / 1024.0
        fitting = [m for m in self.gguf_models if m.size_gb <= usable_vram_gb]
        if fitting:
            return max(fitting, key=lambda m: m.size_gb)
        # Nothing fits cleanly — return smallest as a fallback (partial offload)
        return min(self.gguf_models, key=lambda m: m.size_gb)


def detect_gpus() -> list[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            log.warning("nvidia-smi failed: %s", result.stderr.strip())
            return []

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_mb=int(parts[2]),
                    )
                )
        return gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as exc:
        log.warning("GPU detection failed: %s", exc)
        return []


def find_llama_server() -> str | None:
    """Find the llama-server binary."""
    # Check PATH first
    in_path = shutil.which("llama-server")
    if in_path:
        return in_path

    # Check known locations
    for path in LLAMA_SERVER_SEARCH_PATHS:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def scan_gguf_models(extra_dirs: list[str] | None = None) -> list[GGUFModel]:
    """Scan filesystem for GGUF model files."""
    search_dirs = list(GGUF_SEARCH_DIRS)
    if extra_dirs:
        search_dirs = [Path(d) for d in extra_dirs] + search_dirs

    seen: set[str] = set()
    models: list[GGUFModel] = []

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        # Search up to 3 levels deep to avoid traversing entire filesystems
        for depth_pattern in ["*.gguf", "*/*.gguf", "*/*/*.gguf"]:
            for match in glob.glob(str(base_dir / depth_pattern)):
                real = os.path.realpath(match)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    size_bytes = os.path.getsize(real)
                    models.append(
                        GGUFModel(
                            path=Path(real),
                            size_gb=round(size_bytes / (1024**3), 2),
                            name=Path(real).stem,
                        )
                    )
                except OSError:
                    continue

    models.sort(key=lambda m: m.size_gb, reverse=True)
    return models


def detect_system(extra_model_dirs: list[str] | None = None) -> SystemCapabilities:
    """Run full system detection: GPUs, llama-server, GGUF models."""
    gpus = detect_gpus()
    caps = SystemCapabilities(
        gpus=gpus,
        total_vram_mb=sum(g.vram_mb for g in gpus),
        llama_server_path=find_llama_server(),
        gguf_models=scan_gguf_models(extra_model_dirs),
    )
    log.info(
        "System detected: %d GPU(s) (%d MB VRAM), llama-server=%s, %d GGUF model(s)",
        len(caps.gpus),
        caps.total_vram_mb,
        caps.llama_server_path or "not found",
        len(caps.gguf_models),
    )
    return caps


class ManagedLlamaServer:
    """Start and manage a llama-server subprocess."""

    def __init__(
        self,
        binary_path: str,
        model_path: str,
        port: int = 8080,
        n_gpu_layers: int = 99,
        extra_args: list[str] | None = None,
    ) -> None:
        self.binary_path = binary_path
        self.model_path = model_path
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.extra_args = extra_args or []
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Start llama-server as a background process."""
        if self.is_running:
            return

        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-ngl", str(self.n_gpu_layers),
            "--port", str(self.port),
            "--host", "0.0.0.0",
        ] + self.extra_args

        log.info("Starting llama-server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log.info("llama-server started (PID %d) on port %d", self._process.pid, self.port)

    async def wait_healthy(self, timeout: float = 120.0) -> None:
        """Wait for llama-server to become healthy."""
        import httpx

        url = f"{self.url}/health"
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            if self._process and self._process.poll() is not None:
                stderr = ""
                if self._process.stderr:
                    stderr = self._process.stderr.read().decode(errors="replace")[-500:]
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}: {stderr}"
                )

            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        health = resp.json()
                        if health.get("status") == "ok":
                            log.info("llama-server is healthy")
                            return
            except (httpx.ConnectError, httpx.ReadError):
                pass

            await asyncio.sleep(1.0)

        raise RuntimeError(f"llama-server did not become healthy within {timeout}s")

    def stop(self) -> None:
        """Stop the managed llama-server process."""
        if not self._process:
            return
        if self._process.poll() is None:
            log.info("Stopping llama-server (PID %d)", self._process.pid)
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("llama-server did not stop, killing")
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None

    def __del__(self) -> None:
        self.stop()

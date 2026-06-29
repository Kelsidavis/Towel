"""Towel doctor — diagnose your setup and find problems before they find you.

Checks environment, configuration, model availability, skills, and gateway.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

from rich.console import Console

from towel.config import TOWEL_HOME, TowelConfig

log = logging.getLogger("towel.cli.doctor")
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
    checks.append(check_persisted_worker_state())
    checks.append(check_sqlite_fts5())
    checks.append(check_memory_embeddings())
    checks.append(check_memory_store())

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
        # Get available memory via os.sysconf on macOS/Linux
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_gb = (pages * page_size) / (1024**3)
        c.ok(f"{total_gb:.0f} GB system memory")
        if total_gb < 8:
            c.warn("Less than 8 GB RAM — larger models may not fit")
    except (ValueError, OSError, ImportError):
        c.ok("Memory: could not determine")

    # Disk space for ~/.towel
    try:
        usage = shutil.disk_usage(TOWEL_HOME.parent)
        free_gb = usage.free / (1024**3)
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
        c.suggestions.append("Run: towel setup  (browser GUI)  or  towel init")

    c.ok(f"Model: {config.model.name}")
    c.ok(f"Context window: {config.model.context_window} tokens")
    c.ok(f"Max output: {config.model.max_tokens} tokens")
    # TurboQuant is an MLX-runtime KV cache feature. On the llama/ollama/claude
    # backends it is inert (those manage their own KV cache), so don't claim it.
    if config.backend in ("llama", "ollama", "claude"):
        c.ok(f"KV cache: managed by {config.backend}")
    elif config.model.turboquant:
        c.ok(
            f"KV cache: TurboQuant {config.model.turboquant_bits}-bit "
            f"(QJL ratio {config.model.turboquant_qjl_ratio})"
        )
    else:
        c.ok("KV cache: float16 (standard)")
    c.ok(f"Gateway: {config.gateway.host}:{config.gateway.port}")

    if config.model.context_window <= config.model.max_tokens:
        c.fail(
            f"context_window ({config.model.context_window}) must be "
            f"larger than max_tokens ({config.model.max_tokens})",
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
            total_size = sum(f.stat().st_size for f in hf_model_dir.rglob("*") if f.is_file())
            size_gb = total_size / (1024**3)
            c.ok(f"Cache size: {size_gb:.1f} GB")
        except OSError:
            pass
    else:
        c.warn("Model not cached locally — first run will download it")
        c.suggestions.append(
            f"Pre-download: python -c \"from mlx_lm import load; load('{model_name}')\""
        )

    # Suggest smaller alternatives if relevant
    small_models = [
        m for m in cached_models if any(q in m.lower() for q in ["4bit", "8bit", "3b", "7b", "1b"])
    ]
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
        c.ok(
            f"Built-in skills: {', '.join(builtin_names)} "
            f"({len(builtin_names)} skills, "
            f"{len(registry.tool_definitions())} tools)"
        )
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


def _probe_missing_persisted_workers(c: Check, host: str, http_port: int) -> None:
    """Warn when persisted workers aren't currently connected.

    Each worker that ever registered leaves an entry in
    ``$TOWEL_HOME/worker_state.json``. When a previously-registered
    worker isn't in the live /workers list, the operator probably
    has a launcher crash, network partition, or failed self-upgrade.
    Without this surface, the doctor would say "Workers: 1" with no
    indication that the OTHER expected worker is missing.

    Skips workers explicitly disabled in the persisted state —
    absence there is intentional. Errors are swallowed: this is a
    nice-to-have observability bonus, not a hard check.
    """
    import httpx

    try:
        from towel.persistence.worker_state import WorkerStateStore
        store = WorkerStateStore()
        if not store.path.exists():
            return
        persisted = store.load() or {}
    except Exception as exc:
        log.debug("doctor: persisted worker state read failed: %s", exc)
        return
    expected = {
        wid for wid, s in persisted.items() if s.get("enabled", True)
    }
    if not expected:
        return

    try:
        data = httpx.get(f"http://{host}:{http_port}/workers", timeout=2).json()
    except Exception as exc:
        log.debug("doctor: /workers probe failed during missing check: %s", exc)
        return
    connected = {w.get("id") for w in (data.get("workers") or []) if w.get("id")}
    missing = sorted(expected - connected)
    if missing:
        c.warn(f"Persisted workers not connected: {', '.join(missing)}")
        c.suggestions.append(
            "Check the worker host(s) — process may have crashed (often "
            "during a failed self-upgrade) or the launcher needs a restart"
        )


def _probe_fleet_endpoints(c: Check, host: str, http_port: int) -> None:
    """Probe the fleet/dispatch/skills/memory endpoints on a running gateway.

    Each probe is best-effort: a 404 (older deploy) becomes a warn, a network
    failure becomes a warn with the underlying error, and a successful call
    appends an ``ok`` line summarising what came back. Errors here never fail
    the doctor as a whole — the gateway itself is up, these are bonus checks.
    """
    import httpx

    base = f"http://{host}:{http_port}"

    # /workers — show fleet shape if anything is connected
    try:
        data = httpx.get(f"{base}/workers", timeout=2).json()
        workers = data.get("workers", [])
        if workers:
            hot = 0
            idle = 0
            busy = 0
            tier_counts = {"high": 0, "medium": 0, "low": 0}
            fits_values: list[float] = []
            for w in workers:
                live = (w.get("capabilities") or {}).get("live_resources") or {}
                cp = live.get("cpu_pressure")
                if isinstance(cp, (int, float)) and cp >= 0.8:
                    hot += 1
                if w.get("busy"):
                    busy += 1
                elif w.get("enabled", True) and not w.get("draining"):
                    idle += 1
                tier = w.get("quality_tier")
                if tier in tier_counts:
                    tier_counts[tier] += 1
                fits = (w.get("capabilities") or {}).get("max_param_b_est")
                if isinstance(fits, (int, float)) and fits > 0:
                    fits_values.append(float(fits))
            summary = f"Workers: {len(workers)} ({idle} idle, {busy} busy"
            if hot:
                summary += f", {hot} hot"
            summary += ")"
            c.ok(summary)
            # Coordinator/worker version drift: workers running pre-fix
            # code silently behave differently from current. The
            # capability advertisement now carries `towel_version` —
            # flag any worker that doesn't match the coordinator.
            try:
                from towel import __version__ as _coord_version
            except Exception:
                _coord_version = "0.0.0"
            mismatched: list[str] = []
            unknown: list[str] = []
            for w in workers:
                wv = (w.get("capabilities") or {}).get("towel_version")
                if not wv:
                    unknown.append(w.get("id", "?"))
                elif wv != _coord_version:
                    mismatched.append(f"{w.get('id','?')}={wv}")
            if mismatched:
                c.warn(
                    f"Worker version mismatch (coordinator={_coord_version}): "
                    + ", ".join(mismatched)
                )
                c.suggestions.append(
                    "Restart workers, or click 'upgrade' in the fleet panel"
                )
            if unknown:
                # Workers without the field are running pre-version-
                # advertisement code, which is itself an "update me"
                # signal.
                c.warn(
                    f"{len(unknown)} worker(s) don't advertise towel_version "
                    "— probably pre-fix code"
                )
            # Surface any failed self-upgrade attempts so the operator
            # sees WHY a worker is still on an old version after
            # clicking the upgrade button.
            for w in workers:
                ua = (w.get("capabilities") or {}).get("last_upgrade_attempt")
                if ua:
                    status = ua.get("status", "unknown")
                    strategy = ua.get("strategy", "?")
                    rc = ua.get("returncode")
                    err = ua.get("error") or ua.get("tail") or ""
                    err = err[:80] if err else ""
                    extra = f"rc={rc}, " if rc is not None else ""
                    suffix = f" ({err})" if err else ""
                    c.warn(
                        f"Worker {w.get('id','?')} last upgrade failed "
                        f"({strategy}, {extra}{status}){suffix}"
                    )
            # Tier distribution — quick glance at whether the fleet has the
            # capability mix the workload needs.
            tier_parts = [
                f"{tier_counts[t]} {t}"
                for t in ("high", "medium", "low")
                if tier_counts[t]
            ]
            if tier_parts:
                c.ok(f"Tiers: {', '.join(tier_parts)}")
            # Size range — smallest and largest "fits up to" so operators
            # can see at a glance whether a 70B model has anywhere to land.
            if fits_values:
                lo = min(fits_values)
                hi = max(fits_values)
                if lo == hi:
                    c.ok(f"Fits: up to ~{lo:.1f}B params on every worker")
                else:
                    c.ok(f"Fits: ~{lo:.1f}B…{hi:.1f}B params across workers")
        else:
            c.ok("Workers: none connected (coordinator handles requests locally)")
    except Exception as exc:
        c.warn(f"/workers probe failed: {exc.__class__.__name__}")

    # /skills — confirm the agent loaded its skills
    try:
        data = httpx.get(f"{base}/skills", timeout=2).json()
        skills = data.get("skills", [])
        tools = data.get("total_tools", 0)
        c.ok(f"Skills: {len(skills)} loaded ({tools} tools available)")
    except Exception as exc:
        c.warn(f"/skills probe failed: {exc.__class__.__name__}")

    # /fleet/inventory — total cached model coverage across the fleet
    try:
        data = httpx.get(f"{base}/fleet/inventory", timeout=2).json()
        unique = data.get("total_unique", 0)
        models = data.get("models", [])
        if unique:
            top = models[0] if models else None
            note = f"Inventory: {unique} unique model(s)"
            if top and top.get("cached_count", 0) > 1:
                note += (
                    f"; most replicated: {top['name']} "
                    f"({top['cached_count']}× cached)"
                )
            c.ok(note)
        else:
            c.ok("Inventory: no cached models reported")
    except Exception as exc:
        c.warn(f"/fleet/inventory probe failed: {exc.__class__.__name__}")

    # /dispatch/recent — confirm the dispatcher exists and has been used
    try:
        data = httpx.get(f"{base}/dispatch/recent?limit=1", timeout=2).json()
        decisions = data.get("decisions", [])
        if decisions:
            last = decisions[-1]
            c.ok(
                "Last dispatch: "
                f"{last.get('reason', 'unknown')} → {last.get('worker_id') or '<coordinator>'}"
            )
        else:
            c.ok("Dispatch log: empty (no routing decisions yet)")
        # Flag flaky chat workers — workers whose chat dispatches
        # are routinely producing empty text (tool calls instead of
        # chat) and forcing a retry on the alt. Each such turn costs
        # the user the primary's full latency; surfacing the tally
        # from the dispatch buffer means operators see the offender
        # without curl-ing the dispatch log themselves.
        #
        # Threshold: only warn when at least one worker has produced
        # ≥3 empty-text retries in the buffer. A single one-off retry
        # from a transient failure is noise; a recurring pattern (the
        # live observation that drove this code: 12 retries from one
        # worker in a 500-entry buffer) is real signal worth surfacing.
        retries = (data.get("log_status") or {}).get(
            "empty_text_retries_by_worker"
        ) or {}
        flaky = {wid: n for wid, n in retries.items() if n >= 3}
        if flaky:
            top = sorted(flaky.items(), key=lambda kv: -kv[1])
            summary = ", ".join(f"{wid}={n}" for wid, n in top)
            c.warn(
                f"Empty-text retries by worker (in current buffer): {summary}"
            )
            c.suggestions.append(
                "A worker with many empty-text retries is emitting tool "
                "calls instead of chat — consider pinning chat sessions "
                "away from it or disabling it for chat intent"
            )
        # Total empty-text responses per worker — counts every empty
        # response, including single-worker cases where no retry was
        # possible. The retries tally above misses those because the
        # retry path requires an alternate worker. Surfaces "k-Precision
        # returns empty 100% of the time" on a one-worker fleet that
        # would otherwise look idle on the retry tally alone.
        counts = (data.get("log_status") or {}).get(
            "empty_text_counts_by_worker"
        ) or {}
        broken = {wid: n for wid, n in counts.items() if n >= 3}
        if broken:
            # Filter out workers already flagged in the retry tally
            # above to avoid double-warning operators.
            extra = {wid: n for wid, n in broken.items() if wid not in flaky}
            if extra:
                top = sorted(extra.items(), key=lambda kv: -kv[1])
                summary = ", ".join(f"{wid}={n}" for wid, n in top)
                c.warn(
                    f"Empty-text responses by worker (no alt retry): {summary}"
                )
                c.suggestions.append(
                    "A worker producing empty text with no retry possible "
                    "is usually a model/llama-server problem on that host "
                    "— check the worker's log and consider a fresh restart"
                )
        # Quality-degraded count: dispatches forced onto an
        # under-spec worker because no better-fit candidate was
        # available. Threshold ≥5 distinguishes a recurring fleet/
        # workload mismatch from a one-off ("the big worker was
        # briefly busy"). Surfacing this answers the operator
        # question "why is my code-gen taking forever?" without
        # them having to grep /dispatch/recent for the flag.
        degraded_count = int(
            (data.get("log_status") or {}).get("quality_degraded_count", 0) or 0
        )
        if degraded_count >= 5:
            c.warn(
                f"{degraded_count} quality-degraded dispatch(es) in the "
                "current buffer — tasks are landing on under-spec workers"
            )
            c.suggestions.append(
                "Either the workload needs a bigger worker (more VRAM / "
                "context), or the existing big worker is too often busy "
                "with idle tasks — the bigger-is-better preempt fix "
                "(commit 881031c) reroutes quality tasks onto the big "
                "worker when idle tasks are running there"
            )
        # Timeout count: decisions that hit worker_inference_timeout.
        # Different signal from "stuck workers" (busy_for >= 5min):
        # stuck means a worker is currently wedged; this means past
        # requests gave up at the timeout boundary. Threshold ≥5
        # mirrors the degraded-count threshold — recurring timeouts
        # in the buffer means real model-quality or routing issues,
        # not a one-off slow request.
        timeout_count = int(
            (data.get("log_status") or {}).get("timeout_count", 0) or 0
        )
        if timeout_count >= 5:
            c.warn(
                f"{timeout_count} dispatch(es) hit worker_inference_timeout "
                "in the current buffer — workers are giving up on requests"
            )
            c.suggestions.append(
                "Check /dispatch/recent?min_total_ms=300000 for the "
                "specific sessions. Common causes: tool-loop on a "
                "small worker (use enable-worker / disable-worker to "
                "restrict capable workers), prompt too long for the "
                "model context, or genuine model hang"
            )
    except Exception as exc:
        c.warn(f"/dispatch/recent probe failed: {exc.__class__.__name__}")

    # /cluster/handoffs — flag failed migrations. A nonzero failed count is
    # the kind of thing operators want to know about without having to
    # crack open the fleet panel; doctor surfaces it at CLI level.
    try:
        data = httpx.get(f"{base}/cluster/handoffs", timeout=2).json()
        stats = data.get("stats", {}) or {}
        total = int(stats.get("total", 0) or 0)
        failed = int(stats.get("failed", 0) or 0)
        pending = int(stats.get("pending", 0) or 0)
        if total == 0 and pending == 0:
            c.ok("Handoffs: none recorded (no worker drains / disconnects yet)")
        else:
            avg_ms = stats.get("avg_duration_ms")
            summary = f"Handoffs: {total} total"
            if avg_ms is not None and total:
                summary += f", avg {avg_ms}ms"
            if pending:
                summary += f", {pending} pending"
            c.ok(summary)
            if failed:
                c.warn(
                    f"{failed} handoff(s) failed — see /cluster/handoffs in "
                    "the fleet panel for the error details"
                )
    except Exception as exc:
        c.warn(f"/cluster/handoffs probe failed: {exc.__class__.__name__}")


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
        # Stuck workers: busy_for_seconds >= 5 minutes per /health's
        # workers.stuck count. The fleet panel surfaces this with a
        # red border, but operators running `towel doctor` from the
        # CLI had no equivalent signal — they'd see "Gateway is
        # running ✓" and miss that a worker was wedged on a long-
        # dead request. Translate the count into an actionable warn
        # so the doctor matches the panel's visibility.
        workers_stats = data.get("workers") or {}
        stuck = int(workers_stats.get("stuck", 0) or 0)
        if stuck:
            c.warn(
                f"{stuck} worker(s) stuck (busy ≥ 5 min) — likely wedged "
                "on a request that won't return"
            )
            c.suggestions.append(
                "POST /workers/<id>/state with {\"enabled\": false} to "
                "drain the stuck worker, or restart it"
            )
        _probe_fleet_endpoints(c, host, http_port)
        _probe_missing_persisted_workers(c, host, http_port)
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


def check_persisted_worker_state() -> Check:
    """Surface persisted worker state so operators can audit it.

    The coordinator stores enabled/draining flags and manual task overrides
    in ``$TOWEL_HOME/worker_state.json``. When operators wonder "why is this
    worker disabled?" or "why is my GPU host only picking up chat?", the
    answer often lives here — exposing it from doctor avoids needing to
    crack open the JSON file by hand.
    """
    from towel.persistence.worker_state import WorkerStateStore

    c = Check("Persisted worker state")
    store = WorkerStateStore()
    if not store.path.exists():
        c.ok("No persisted worker state (clean slate)")
        return c

    try:
        states = store.load()
    except Exception as exc:
        c.fail(f"Failed to read {store.path}: {exc}")
        return c

    if not states:
        c.ok(f"{store.path} is present but empty")
        return c

    disabled = [wid for wid, s in states.items() if not s.get("enabled", True)]
    draining = [wid for wid, s in states.items() if s.get("draining", False)]
    overrides = {
        wid: s.get("tasks") for wid, s in states.items() if s.get("tasks")
    }

    c.ok(f"{len(states)} worker(s) with persisted state in {store.path}")
    if disabled:
        c.warn(f"Disabled (excluded from dispatch): {', '.join(disabled)}")
        c.suggestions.append(
            "Re-enable via the fleet panel or POST /workers/{id}/enable"
        )
    if draining:
        c.warn(f"Draining (no new sessions): {', '.join(draining)}")
    if overrides:
        for worker_id, tasks in overrides.items():
            c.details.append(
                f"Manual task override for {worker_id}: {', '.join(tasks)}"
            )
        c.suggestions.append(
            "Clear an override by POSTing an empty tasks list to "
            "/workers/{id}/tasks"
        )

    return c


def check_sqlite_fts5() -> Check:
    """Verify the host SQLite supports FTS5.

    The memory store relies on the FTS5 virtual table for BM25-ranked
    retrieval. Almost every modern sqlite ships it, but stripped-down
    builds (some Alpine/musl images, custom embedded sqlite) leave it
    out. Surface this on doctor rather than waiting for the first
    ``remember`` call to error out.
    """
    import sqlite3

    c = Check("SQLite FTS5")
    c.ok(f"sqlite3 module {sqlite3.sqlite_version}")
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE _probe USING fts5(content)")
        con.close()
        c.ok("FTS5 virtual table available")
    except sqlite3.OperationalError as exc:
        c.fail(f"FTS5 not compiled into this sqlite ({exc})")
        c.suggestions.append(
            "Install Python with a sqlite that has FTS5 (Homebrew, "
            "Debian 11+, official python.org installers all do). On "
            "Alpine, install sqlite-dev and rebuild Python, or use the "
            "glibc image."
        )
    return c


def check_memory_embeddings() -> Check:
    """Report whether the optional vector-recall extra is installed.

    Pure informational: warn (not fail) when missing, since the
    memory store works fine without — retrieval just doesn't get
    paraphrase recall, and ``fused_search`` degrades to BM25 alone.
    Also flags dimension drift across the corpus, which silently
    breaks cosine ranking for the minority-dimension rows.
    """
    from towel.memory import embeddings
    from towel.memory.store import MemoryStore

    c = Check("Memory embeddings")
    if embeddings.is_available():
        c.ok(
            f"sentence-transformers installed — vector recall enabled "
            f"(model: {embeddings.DEFAULT_MODEL})"
        )
    else:
        c.warn("Embeddings extra not installed — retrieval is BM25 + graph only")
        c.suggestions.append(
            "Install the extra for paraphrase recall: "
            "pip install 'towel-ai[embeddings]'"
        )

    # Dimension consistency across stored vectors. Mixed dims usually
    # mean $TOWEL_EMBED_MODEL changed without a re-encode pass; the
    # old vectors then never match the new query embedding and silently
    # contribute nothing to ranking.
    try:
        dims = MemoryStore().embedding_dims()
    except Exception:
        # If we can't read the store (locked, missing, etc.) the
        # embeddings part of doctor is best-effort; the main store
        # check below will fail loudly with a real error.
        return c
    if len(dims) > 1:
        sizes = ", ".join(f"{d}d:{n}" for d, n in sorted(dims.items(), key=lambda kv: -kv[1]))
        c.warn(f"Mixed embedding dimensions in corpus ({sizes})")
        c.suggestions.append(
            "Re-encode every row: towel memory reembed --all"
        )
    return c


def check_memory_store() -> Check:
    """Verify the agent's persistent memory DB is readable.

    Flags two notable conditions:
    - The store fails to open (FTS5 missing, perms, disk full). The
      FTS5 case is also surfaced by ``check_sqlite_fts5`` but is worth
      reporting here too so the operator sees the symptom adjacent to
      the count.
    - A legacy ``memories.json.migrated-*`` archive is present —
      informational, not a problem, but useful for operators wondering
      "where did my JSON go".
    """
    from towel.memory.store import DEFAULT_MEMORY_DIR, MemoryStore

    c = Check("Memory store")
    db_path = DEFAULT_MEMORY_DIR / "memory.db"

    # Surface any migration markers so operators see what happened.
    archives = sorted(DEFAULT_MEMORY_DIR.glob("memories.json.migrated-*"))
    if archives:
        c.ok(
            f"Migrated from JSON store ({len(archives)} archive(s); "
            f"latest: {archives[-1].name})"
        )

    try:
        store = MemoryStore()
        count = store.count
    except Exception as exc:
        c.fail(f"Memory store unreadable: {exc}")
        return c

    if count == 0 and not db_path.exists():
        c.ok(f"No memories stored yet ({db_path})")
        return c
    c.ok(f"Memory store: {count} entries in {db_path}")

    # Recall-log sanity: with a populated corpus, we'd expect the
    # log to have at least a few rows if the agent has ever
    # retrieved anything. Surface this only when count > 5 — small
    # stores legitimately don't have recall history yet.
    try:
        log_size = store.recall_log_size()
        if count > 5 and log_size == 0:
            c.warn(
                "No recall events logged — has the agent run any "
                "queries since the recall_log was introduced?"
            )
            c.suggestions.append(
                "Send a message through the agent or run "
                "`towel memory search QUERY` to populate the log."
            )
        elif log_size > 0:
            cap = store.RECALL_LOG_CAP
            pct = (log_size * 100 // cap) if cap else 0
            c.ok(f"Recall log: {log_size} events ({pct}% of cap {cap})")
            if log_size >= cap:
                c.warn(
                    "Recall log is at cap — older events are being "
                    "pruned. Bump TowelConfig.memory_recall_log_cap "
                    "to extend the audit window."
                )
    except Exception:
        pass

    # Noisy auto-capture: flag when most of the corpus is heuristic
    # captures that the agent has never actually used. Either the
    # patterns are firing too aggressively, or the operator hasn't
    # had a conversation that exercised those memories yet — either
    # way it's worth a heads-up so they know to run `memory tidy`
    # (or `memory stats` to inspect).
    entries = store.recall_all()
    auto = [e for e in entries if (e.source or "").startswith("auto_capture:")]
    if entries:
        unused_auto = [e for e in auto if e.recall_count == 0]
        if len(unused_auto) >= 10 and len(unused_auto) >= len(entries) // 2:
            c.warn(
                f"{len(unused_auto)} auto-captured memor(ies) have never "
                f"been recalled (>50% of corpus)"
            )
            c.suggestions.append(
                "Run `towel memory stats` to inspect, then `towel memory "
                "tidy --auto-only` to prune the heuristic captures."
            )
        # Pattern-level cold callout: a single pattern with ≥5 captures
        # and zero recalls is more diagnostic than the aggregate above
        # because it points at the specific regex that needs review.
        per_pattern: dict[str, list[int]] = {}
        for e in auto:
            label = e.source.split(":", 1)[1] if ":" in e.source else e.source
            stats = per_pattern.setdefault(label, [0, 0])
            stats[0] += 1
            if e.recall_count > 0:
                stats[1] += 1
        cold = [p for p, (cap, rec) in per_pattern.items() if cap >= 5 and rec == 0]
        if cold:
            c.warn(
                f"Cold auto-capture pattern(s) (captures ≥5, recalls=0): "
                f"{', '.join(sorted(cold))}"
            )
            c.suggestions.append(
                "Review the regex in src/towel/memory/auto_capture.py; "
                "if the matches look right, leave it — the relevant "
                "conversations may not have happened yet."
            )
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

        # Surface corruption backups left by ConversationStore.load()
        # when a file failed to parse. Same as the memory-store check
        # so operators see them on the very next `doctor` rather than
        # only when something else breaks. Also surfaces orphan .tmp
        # files from interrupted atomic saves (2b9060c).
        corrupted = sorted(conv_dir.glob("*.json.corrupted-*"))
        if corrupted:
            c.warn(
                f"Found {len(corrupted)} corrupted-conversation backup(s) "
                f"— most recent: {corrupted[-1].name}"
            )
            c.suggestions.append(
                "Inspect each backup and either recover content or rm if "
                "no longer needed."
            )
        orphan_tmps = sorted(conv_dir.glob("*.json.tmp"))
        if orphan_tmps:
            c.warn(
                f"Found {len(orphan_tmps)} orphan .json.tmp file(s) from "
                "interrupted atomic saves"
            )
            c.suggestions.append(
                "DELETE /conversations?confirm=yes also sweeps these, "
                "or rm them manually."
            )
    else:
        c.ok("No conversations stored yet")

    # Check TOWEL_HOME
    if TOWEL_HOME.exists():
        c.ok(f"Towel home: {TOWEL_HOME}")
    else:
        c.ok(f"Towel home not initialized ({TOWEL_HOME})")
        c.suggestions.append("Run: towel setup  (browser GUI)  or  towel init")

    return c

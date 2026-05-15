"""Towel launcher — a tiny HTTP daemon that spawns ``towel worker`` on demand.

The launcher lets a coordinator (or any client with the shared bearer token)
ask a remote host to start a fresh Towel worker process without SSH access.
A host running ``towel launcher`` listens on an HTTP port and exposes:

  - ``GET /health``        — liveness probe (no auth)
  - ``POST /launch``       — spawn a ``towel worker`` subprocess (token required)
  - ``GET /launches/{pid}`` — fetch boot log for a previously-spawned worker
    (token required)

The launcher is intentionally minimal: it doesn't track worker lifecycles,
restart crashed children, or load-balance. Its job is "the controller asked
us to bring up a worker; do it and report the PID." Once the worker is
running it talks directly to the controller over its own WebSocket, so the
launcher can be killed afterwards without affecting the worker.

To make boot failures visible (model not found, port in use, ImportError,
OOM), the launcher writes each spawned worker's stdout+stderr to a
per-pid file under ``$TOWEL_HOME/launcher-logs/`` and waits ~1s after
spawn before responding. If the worker has already exited at that point,
the response is 500 with the captured tail instead of the optimistic
200/ok that previously masked instant crashes.

Auth: a shared bearer token sourced from the ``TOWEL_TRIGGER_TOKEN`` env
var. The launcher refuses to start if the env var is unset — fail-secure
is safer than running open on a LAN by accident.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from towel.config import TOWEL_HOME

log = logging.getLogger("towel.launcher")

DEFAULT_PORT = 18751
TOKEN_ENV = "TOWEL_TRIGGER_TOKEN"
_VALID_BACKENDS = {"mlx", "ollama", "llama", "claude"}
# Where boot logs land. The pid-keyed log file gives operators something
# concrete to read when a worker doesn't register with the coordinator.
LAUNCHER_LOG_DIR = TOWEL_HOME / "launcher-logs"
# How long to wait after spawn before deciding the worker booted cleanly.
# Workers that don't crash within this window almost always go on to
# register with the coordinator; ones that do crash usually do so on
# import or argument-parse, well within 1s.
_BOOT_GRACE_SECS = 1.0
# Cap the in-response tail so a chatty traceback doesn't blow up the body.
_LOG_TAIL_BYTES = 4000


def _check_token(request: Request, token: str) -> JSONResponse | None:
    """Return a 401 response if the request lacks the bearer token, else None."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse(
            {"error": "missing bearer token"}, status_code=401
        )
    presented = auth[len("Bearer ") :].strip()
    if presented != token:
        return JSONResponse({"error": "invalid token"}, status_code=401)
    return None


def _build_worker_argv(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    """Build the argv for ``towel worker`` from a launch payload.

    Returns ``(argv, error)``. When ``error`` is non-None the caller should
    return it as a 400 without spawning anything.
    """
    controller = payload.get("controller") or payload.get("master")
    if not controller:
        return [], "controller (ws:// or wss:// URL) is required"
    # Reject non-string controller. A list / dict / number passed here
    # previously got Python-repr'd into the argv ("['ws://x']") which
    # produced a worker that silently failed to connect to a bogus URL
    # — the operator saw a process spawn but no worker register.
    if not isinstance(controller, str):
        return [], "controller must be a string"
    # Reject controller URLs that don't look like ws:// or wss:// —
    # otherwise a typo'd "http://..." would launch a worker that
    # immediately fails its websockets.connect with an opaque error.
    if not (controller.startswith("ws://") or controller.startswith("wss://")):
        return [], "controller must be a ws:// or wss:// URL"
    backend = payload.get("backend")
    if backend is not None and backend not in _VALID_BACKENDS:
        return [], f"unknown backend: {backend!r}"

    # Find the towel binary — fall back to ``python -m towel.cli.main`` style
    # invocation if the binary isn't on PATH (e.g. an editable install in a
    # virtualenv that wasn't activated for this process).
    binary = shutil.which("towel")
    if binary:
        argv = [binary, "worker"]
    else:
        argv = ["python", "-m", "towel", "worker"]

    argv.extend(["--master", controller])
    if backend:
        argv.extend(["--backend", backend])
    for opt in ("ollama_url", "llama_url", "llama_model", "claude_model"):
        val = payload.get(opt)
        if val:
            argv.extend([f"--{opt.replace('_', '-')}", str(val)])
    # ``model`` overrides the worker's config.model.name at startup — the
    # primary knob the coordinator uses to distribute different models to
    # different workers.
    model = payload.get("model")
    if model:
        argv.extend(["--model", str(model)])
    worker_id = payload.get("worker_id")
    if worker_id:
        argv.extend(["--worker-id", str(worker_id)])
    # Forward the operator's allow_tools choice. Default at the worker CLI
    # is enabled, so we only emit a flag when the payload explicitly pins
    # it — and the flag must match Click's registered names
    # (--allow-tools / --no-allow-tools), not the shorthand --no-tools.
    allow = payload.get("allow_tools")
    if allow is False:
        argv.append("--no-allow-tools")
    elif allow is True:
        argv.append("--allow-tools")
    return argv, None


def _log_path_for(pid: int) -> Path:
    """Where to write a given worker's boot log."""
    LAUNCHER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LAUNCHER_LOG_DIR / f"worker-{pid}.log"


def _tail_bytes(path: Path, limit: int = _LOG_TAIL_BYTES) -> str:
    """Return the last ``limit`` bytes of a file as a UTF-8 string.

    Returns an empty string if the file doesn't exist or can't be read —
    boot logs are best-effort, so callers shouldn't have to special-case
    missing files.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > limit:
        data = data[-limit:]
        return "…" + data.decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")


def _spawn_worker(
    argv: list[str], env_overrides: dict[str, str] | None
) -> tuple[subprocess.Popen[bytes], Path]:
    """Spawn the worker as a detached subprocess.

    The child inherits the launcher's environment (so HF cache locations,
    Ollama URLs, etc. work). ``env_overrides`` lets the caller add or replace
    individual variables — useful for forwarding ``TOWEL_HOME`` or specifying
    a different model cache without restarting the launcher.

    Returns ``(proc, log_path)``. Stdout and stderr are merged into the
    log file so operators have something to read when a worker fails to
    register with the coordinator. The log is named after the spawned
    pid so ``GET /launches/{pid}`` can find it deterministically.
    """
    import tempfile

    env = dict(os.environ)
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    LAUNCHER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Unique tmp name per call so concurrent /launch requests don't stomp
    # on each other's log file. Rename to pid-keyed path post-spawn.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix="spawn-", suffix=".log", dir=LAUNCHER_LOG_DIR
    )
    tmp_path = Path(tmp_name)
    try:
        log_handle = os.fdopen(tmp_fd, "wb")
    except Exception:
        os.close(tmp_fd)
        raise
    proc = subprocess.Popen(
        argv,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        # ``start_new_session`` detaches from this launcher's process group, so
        # killing the launcher doesn't drag the worker down with it.
        start_new_session=True,
    )
    log_handle.close()
    # Rename to the pid-keyed path now that we know the pid. The child has
    # its own fd pointing at the inode so writes follow the file, not the
    # path; the rename is safe.
    final_path = _log_path_for(proc.pid)
    try:
        tmp_path.replace(final_path)
    except OSError:
        # Cross-filesystem or permission failure — keep the tmp path so
        # the response can still show *something*.
        final_path = tmp_path
    return proc, final_path


_DEFAULT_UPGRADE_COMMANDS: dict[str, list[str]] = {
    # Plain pip install from PyPI. Suitable for production deployments where
    # towel is installed as a regular package.
    "pip": ["pip", "install", "--upgrade", "towel"],
    # pip + git: pull latest main and reinstall in editable mode. The
    # operator must run the launcher from inside the repo checkout for this
    # to be meaningful.
    "git-pull": ["sh", "-c", "git pull --ff-only && pip install -e ."],
    # uv equivalent for hosts on the uv toolchain.
    "uv": ["uv", "pip", "install", "--upgrade", "towel"],
}


# Public alias — re-used by `towel worker --auto-update` so the in-process
# self-upgrade path runs the exact same commands as the remote upgrade RPC.
# Renaming any strategy below changes behavior on both endpoints in lock-step.
UPGRADE_STRATEGIES = _DEFAULT_UPGRADE_COMMANDS


# Module-state record of the most recent upgrade attempt. Read by
# default_worker_capabilities so a failed self_upgrade surfaces in
# the next capability heartbeat — without this, operators see only
# "ok=true" on the dispatcher endpoint and have no way to tell
# success from "command exited 1 silently". Reset to None on
# successful re-exec (the new process starts fresh).
_last_upgrade_attempt: dict[str, Any] | None = None


def get_last_upgrade_attempt() -> dict[str, Any] | None:
    """Return the failure record for the most recent in-process upgrade."""
    return _last_upgrade_attempt


def _record_upgrade_attempt(
    strategy: str,
    status: str,
    *,
    returncode: int | None = None,
    error: str | None = None,
    tail: str | None = None,
) -> None:
    """Stash an upgrade outcome where the capability advertiser can find it."""
    global _last_upgrade_attempt
    from datetime import datetime, UTC

    _last_upgrade_attempt = {
        "ts": datetime.now(UTC).isoformat(),
        "strategy": strategy,
        "status": status,  # "failed_command" / "failed_exit" / "timeout"
    }
    if returncode is not None:
        _last_upgrade_attempt["returncode"] = returncode
    if error:
        _last_upgrade_attempt["error"] = error
    if tail:
        _last_upgrade_attempt["tail"] = tail


def self_upgrade_and_reexec(strategy: str) -> bool:
    """Run the named upgrade strategy in-process, then re-exec on success.

    Shared by two paths:

    * ``towel worker --auto-update`` runs this before connecting so a
      freshly-booted worker is on the latest code.
    * The coordinator can send a ``self_upgrade`` WS message to a running
      worker; the worker then calls this and reboots itself without any
      separate launcher daemon.

    Re-exec is mandatory: Python pins imported modules, so a plain
    ``pip install`` followed by ``return`` would keep running the old
    code. ``TOWEL_AUTO_UPDATE_DONE=1`` is set on the re-exec env so the
    ``--auto-update`` startup check does not loop forever.

    On failure (unknown strategy, command not found, non-zero exit,
    300 s timeout) we log a warning, stash the outcome in module
    state via :func:`_record_upgrade_attempt`, and return ``False``.
    The caller keeps running with the existing code — a stale worker
    is more useful than no worker — and the failure surfaces in the
    next capability heartbeat so the operator sees what happened.
    """
    import os
    import subprocess
    import sys

    cmd = UPGRADE_STRATEGIES.get(strategy)
    if cmd is None:
        log.warning(
            "self-upgrade: unknown strategy %r; known: %s",
            strategy, sorted(UPGRADE_STRATEGIES),
        )
        _record_upgrade_attempt(
            strategy, "unknown_strategy",
            error=f"known: {sorted(UPGRADE_STRATEGIES)}",
        )
        return False
    log.info("self-upgrade: running %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError as exc:
        log.warning("self-upgrade: command not found (%s); continuing.", exc)
        _record_upgrade_attempt(strategy, "command_not_found", error=str(exc))
        return False
    except subprocess.TimeoutExpired:
        log.warning("self-upgrade: timed out after 300s; continuing.")
        _record_upgrade_attempt(strategy, "timeout")
        return False
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-3:])
        log.warning(
            "self-upgrade: exit %d; continuing. Last lines:\n%s",
            result.returncode, tail,
        )
        _record_upgrade_attempt(
            strategy, "failed_exit",
            returncode=result.returncode, tail=tail,
        )
        return False
    log.info("self-upgrade: succeeded, re-executing with new code.")
    env = dict(os.environ)
    env["TOWEL_AUTO_UPDATE_DONE"] = "1"
    # sys.argv[0] is whatever invoked us (bare "towel" on PATH, or the
    # absolute path systemd's ExecStart pointed at). execvpe resolves
    # bare names via PATH so both forms work.
    os.execvpe(sys.argv[0], sys.argv, env)
    # Unreachable, but keep the type checker happy.
    return True


def build_app(token: str) -> Starlette:
    """Build the launcher's Starlette app bound to a specific bearer token."""

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "towel-launcher"})

    async def upgrade(request: Request) -> JSONResponse:
        """Run a software-upgrade command on this host.

        Body shape::

            {"strategy": "pip" | "git-pull" | "uv"}   # use a built-in recipe
            {"command": ["sh", "-c", "..."]}          # or a custom argv

        The launcher executes the chosen command synchronously (so the
        caller can wait for the upgrade to finish before spawning a
        replacement worker), with a 5-minute timeout. Stdout + stderr +
        exit code come back in the JSON response. Auth: same bearer token
        as ``/launch``.
        """
        denied = _check_token(request, token)
        if denied is not None:
            return denied
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse(
                {"error": f"invalid JSON: {exc}"}, status_code=400
            )
        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": "payload must be a JSON object"}, status_code=400
            )

        custom = payload.get("command")
        if custom is not None:
            if not isinstance(custom, list) or not all(isinstance(p, str) for p in custom):
                return JSONResponse(
                    {"error": "command must be a list of strings"}, status_code=400
                )
            cmd = list(custom)
            strategy_used = "custom"
        else:
            strategy = (payload.get("strategy") or "pip").strip()
            cmd = _DEFAULT_UPGRADE_COMMANDS.get(strategy)
            if cmd is None:
                return JSONResponse(
                    {
                        "error": f"unknown strategy: {strategy!r}; "
                        f"known strategies: {sorted(_DEFAULT_UPGRADE_COMMANDS)}"
                    },
                    status_code=400,
                )
            strategy_used = strategy

        log.info("Upgrade requested (strategy=%s): %s", strategy_used, cmd)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except FileNotFoundError as exc:
            return JSONResponse(
                {"error": f"command not found: {exc}"}, status_code=500
            )
        except subprocess.TimeoutExpired:
            return JSONResponse(
                {"error": "upgrade timed out after 300s", "strategy": strategy_used},
                status_code=504,
            )

        # Cap stdout/stderr so a chatty pip resolver doesn't blow up the
        # response. Operators can ssh in for full logs if needed.
        def _tail(text: str, limit: int = 4000) -> str:
            return text if len(text) <= limit else "…" + text[-limit:]

        return JSONResponse(
            {
                "ok": result.returncode == 0,
                "strategy": strategy_used,
                "command": cmd,
                "returncode": result.returncode,
                "stdout": _tail(result.stdout or ""),
                "stderr": _tail(result.stderr or ""),
            },
            status_code=200 if result.returncode == 0 else 500,
        )

    async def launch(request: Request) -> JSONResponse:
        denied = _check_token(request, token)
        if denied is not None:
            return denied
        try:
            payload = await request.json()
        except Exception as exc:
            return JSONResponse({"error": f"invalid JSON: {exc}"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "payload must be a JSON object"}, status_code=400)

        argv, err = _build_worker_argv(payload)
        if err is not None:
            return JSONResponse({"error": err}, status_code=400)

        env_overrides = payload.get("env") or {}
        if not isinstance(env_overrides, dict):
            return JSONResponse(
                {"error": "env must be an object of string→string"}, status_code=400
            )

        try:
            proc, log_path = _spawn_worker(argv, env_overrides)
        except FileNotFoundError as exc:
            return JSONResponse(
                {"error": f"towel binary not found: {exc}"}, status_code=500
            )
        except OSError as exc:
            return JSONResponse({"error": f"spawn failed: {exc}"}, status_code=500)

        log.info("Spawned worker pid=%s argv=%s log=%s", proc.pid, argv, log_path)
        # Give the worker a short window to crash on import/argparse before
        # claiming success. Workers that survive this almost always go on
        # to register with the coordinator.
        time.sleep(_BOOT_GRACE_SECS)
        exit_code = proc.poll()
        log_tail = _tail_bytes(log_path)
        if exit_code is not None:
            log.warning(
                "Spawned worker pid=%s exited immediately with code=%s — "
                "see %s for the full log",
                proc.pid,
                exit_code,
                log_path,
            )
            return JSONResponse(
                {
                    "ok": False,
                    "pid": proc.pid,
                    "argv": argv,
                    "exit_code": exit_code,
                    "log_path": str(log_path),
                    "log_tail": log_tail,
                    "error": (
                        f"worker exited within {_BOOT_GRACE_SECS}s of spawn "
                        f"(code {exit_code}); see log_tail for the cause"
                    ),
                },
                status_code=500,
            )
        return JSONResponse(
            {
                "ok": True,
                "pid": proc.pid,
                "argv": argv,
                "log_path": str(log_path),
                # Include any startup output that landed in the grace
                # window so a curl client can sanity-check what was logged
                # even on a successful spawn.
                "log_tail": log_tail,
            }
        )

    async def get_launch_log(request: Request) -> JSONResponse:
        """Return the captured boot log for a previously-spawned worker.

        Useful when a worker registered fine but later crashed, or when
        the operator wants the full tail rather than the snippet bundled
        in the original ``/launch`` response. The log file persists for
        the lifetime of ``LAUNCHER_LOG_DIR`` — operators can rotate it
        externally if disk pressure becomes a concern.
        """
        denied = _check_token(request, token)
        if denied is not None:
            return denied
        try:
            pid = int(request.path_params["pid"])
        except (KeyError, ValueError):
            return JSONResponse({"error": "pid must be an integer"}, status_code=400)
        log_path = _log_path_for(pid)
        if not log_path.exists():
            return JSONResponse(
                {"error": f"no log for pid {pid}"}, status_code=404
            )
        return JSONResponse(
            {
                "pid": pid,
                "log_path": str(log_path),
                "log_tail": _tail_bytes(log_path),
            }
        )

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/launch", launch, methods=["POST"]),
            Route("/upgrade", upgrade, methods=["POST"]),
            Route("/launches/{pid}", get_launch_log),
        ]
    )


def run(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> None:
    """Run the launcher daemon. Reads the bearer token from ``TOWEL_TRIGGER_TOKEN``."""
    import uvicorn

    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"{TOKEN_ENV} is unset — the launcher refuses to start without a "
            "shared bearer token. Set the env var to any sufficiently random "
            "string and pass the same value in the Authorization header on "
            "every /launch request."
        )
    # Surface INFO-level events (upgrade requested, worker spawned, etc.)
    # in the daemon's terminal. Shared helper so the format is consistent
    # with the other Towel daemons.
    from towel.logging_setup import configure_terminal_logging

    configure_terminal_logging()
    log.info("Towel launcher listening on http://%s:%d (token from $%s)", host, port, TOKEN_ENV)
    uvicorn.run(build_app(token), host=host, port=port, log_level="warning")

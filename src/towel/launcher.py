"""Towel launcher — a tiny HTTP daemon that spawns ``towel worker`` on demand.

The launcher lets a coordinator (or any client with the shared bearer token)
ask a remote host to start a fresh Towel worker process without SSH access.
A host running ``towel launcher`` listens on an HTTP port and exposes:

  - ``GET /health``  — liveness probe (no auth)
  - ``POST /launch`` — spawn a ``towel worker`` subprocess (token required)

The launcher is intentionally minimal: it doesn't track worker lifecycles,
restart crashed children, or load-balance. Its job is "the controller asked
us to bring up a worker; do it and report the PID." Once the worker is
running it talks directly to the controller over its own WebSocket, so the
launcher can be killed afterwards without affecting the worker.

Auth: a shared bearer token sourced from the ``TOWEL_TRIGGER_TOKEN`` env
var. The launcher refuses to start if the env var is unset — fail-secure
is safer than running open on a LAN by accident.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

log = logging.getLogger("towel.launcher")

DEFAULT_PORT = 18751
TOKEN_ENV = "TOWEL_TRIGGER_TOKEN"
_VALID_BACKENDS = {"mlx", "ollama", "llama", "claude"}


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
    worker_id = payload.get("worker_id")
    if worker_id:
        argv.extend(["--worker-id", str(worker_id)])
    if payload.get("allow_tools") is False:
        argv.append("--no-tools")
    return argv, None


def _spawn_worker(argv: list[str], env_overrides: dict[str, str] | None) -> subprocess.Popen[bytes]:
    """Spawn the worker as a detached subprocess.

    The child inherits the launcher's environment (so HF cache locations,
    Ollama URLs, etc. work). ``env_overrides`` lets the caller add or replace
    individual variables — useful for forwarding ``TOWEL_HOME`` or specifying
    a different model cache without restarting the launcher.
    """
    env = dict(os.environ)
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})
    return subprocess.Popen(
        argv,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # ``start_new_session`` detaches from this launcher's process group, so
        # killing the launcher doesn't drag the worker down with it.
        start_new_session=True,
    )


def build_app(token: str) -> Starlette:
    """Build the launcher's Starlette app bound to a specific bearer token."""

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "towel-launcher"})

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
            proc = _spawn_worker(argv, env_overrides)
        except FileNotFoundError as exc:
            return JSONResponse(
                {"error": f"towel binary not found: {exc}"}, status_code=500
            )
        except OSError as exc:
            return JSONResponse({"error": f"spawn failed: {exc}"}, status_code=500)

        log.info("Spawned worker pid=%s argv=%s", proc.pid, argv)
        return JSONResponse(
            {
                "ok": True,
                "pid": proc.pid,
                "argv": argv,
            }
        )

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/launch", launch, methods=["POST"]),
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
    log.info("Towel launcher listening on http://%s:%d (token from $%s)", host, port, TOKEN_ENV)
    uvicorn.run(build_app(token), host=host, port=port, log_level="warning")

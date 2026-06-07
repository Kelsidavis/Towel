"""Setup wizard — a tiny local-only HTTP server backing the setup GUI.

Exposes three endpoints used by ``web/setup.html``:

- ``GET /api/setup/state``  →  current config + per-backend availability probe
- ``GET /api/setup/backends/<name>/models``  →  installed/cached models for that backend
- ``POST /api/setup/save``  →  validate, write ``~/.towel/config.toml``

It can run two ways:

1. **Standalone**: ``towel setup`` boots a one-shot Starlette server, opens the
   browser, and the server keeps running until the user closes the tab or hits
   Ctrl-C. This is what first-time users get — no need to have ``towel serve``
   already running.

2. **Inside the gateway**: the same handlers are also exported as a Starlette
   route list (``setup_routes()``) so the main gateway server can serve the
   setup page at ``/setup`` for live reconfiguration.

The wizard is intentionally limited to fields the CLI honours today:
``backend``, ``identity``, ``model.name``, ``ollama_url``, ``llama_url``,
``claude_model``. Other config (TurboQuant knobs, gateway port, agent
profiles) is left untouched — power users edit the TOML directly.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from towel.config import TOWEL_HOME, TowelConfig

log = logging.getLogger("towel.setup")

_WEB_DIR = Path(__file__).parent / "web"


# --------------------------------------------------------------------------- #
# Backend probes                                                              #
# --------------------------------------------------------------------------- #


async def _probe_mlx() -> dict[str, Any]:
    try:
        import mlx_lm  # noqa: F401
    except Exception as exc:
        return {"available": False, "reason": f"mlx_lm not importable ({exc})"}
    return {"available": True, "reason": "mlx_lm installed"}


async def _probe_ollama(url: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{url.rstrip('/')}/api/version")
            resp.raise_for_status()
            ver = resp.json().get("version", "?")
        return {"available": True, "reason": f"Ollama {ver} at {url}"}
    except Exception as exc:
        return {"available": False, "reason": f"not reachable at {url} ({exc.__class__.__name__})"}


async def _probe_llama(url: str) -> dict[str, Any]:
    binary = shutil.which("llama-server")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{url.rstrip('/')}/health")
            resp.raise_for_status()
        return {"available": True, "reason": f"llama-server running at {url}"}
    except Exception:
        if binary:
            return {
                "available": True,
                "reason": f"llama-server binary at {binary} (will auto-start)",
            }
        return {
            "available": False,
            "reason": "no running server and no llama-server binary in PATH",
        }


def _probe_claude() -> dict[str, Any]:
    # Two credential locations: macOS keychain (Claude Code default) and a
    # plaintext fallback at ~/.claude/.credentials.json (Linux / opt-in).
    plain = Path.home() / ".claude" / ".credentials.json"
    if plain.exists():
        return {"available": True, "reason": "credentials at ~/.claude/.credentials.json"}
    try:
        username = os.environ.get("USER") or os.getlogin()
        result = subprocess.run(
            ["security", "find-generic-password", "-a", username, "-s", "Claude Code-credentials"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            return {"available": True, "reason": "credentials in macOS keychain"}
    except Exception:
        pass
    return {
        "available": False,
        "reason": "no Claude Code credentials found — run `claude` and log in first",
    }


async def _probe_backends(config: TowelConfig) -> dict[str, dict[str, Any]]:
    ollama_url = config.ollama_url or "http://localhost:11434"
    llama_url = config.llama_url or "http://localhost:8080"
    return {
        "mlx": await _probe_mlx(),
        "ollama": await _probe_ollama(ollama_url),
        "llama": await _probe_llama(llama_url),
        "claude": _probe_claude(),
    }


# --------------------------------------------------------------------------- #
# Model listings                                                              #
# --------------------------------------------------------------------------- #


def _list_mlx_cached_models(limit: int = 50) -> list[str]:
    """List MLX-style model identifiers found in the local Hugging Face cache."""
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
    ]
    names: list[str] = []
    for root in candidates:
        if not root or not root.is_dir():
            continue
        for entry in root.iterdir():
            if not entry.name.startswith("models--"):
                continue
            # models--<org>--<name> → <org>/<name>
            stripped = entry.name[len("models--"):]
            if "--" in stripped:
                org, _, model_name = stripped.partition("--")
                names.append(f"{org}/{model_name}")
    # Deduplicate while preserving order, prefer MLX-ish names first.
    seen: set[str] = set()
    unique: list[str] = []
    for n in sorted(names, key=lambda s: (0 if "mlx" in s.lower() else 1, s)):
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique[:limit]


async def _list_ollama_models(url: str) -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url.rstrip('/')}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", []) if m.get("name")]
    except Exception as exc:
        log.debug("Ollama tag listing failed: %s", exc)
        return []


# --------------------------------------------------------------------------- #
# Save                                                                        #
# --------------------------------------------------------------------------- #


_VALID_BACKENDS = {"", "mlx", "ollama", "llama", "claude"}


def _apply_form_to_config(config: TowelConfig, form: dict[str, Any]) -> tuple[bool, str | None]:
    """Mutate ``config`` from the wizard form. Returns ``(ok, error)``.

    Every string field is read via ``_form_str`` so a non-string
    value (the wizard JSON shipping ``backend: 42`` because a buggy
    client serialised the form wrong) falls back to "" instead of
    crashing ``.strip()`` with AttributeError. Without this, the
    save handler returned HTTP 500 on any form with a non-string
    field — which the operator saw as "the wizard is broken" with
    no usable error message.
    """
    def _form_str(key: str, fallback: str = "") -> str:
        v = form.get(key)
        return v.strip() if isinstance(v, str) else fallback

    backend = _form_str("backend")
    if backend not in _VALID_BACKENDS:
        return False, f"Unknown backend: {backend!r}"
    config.backend = backend
    identity = _form_str("identity")
    config.identity = identity or config.identity
    config.ollama_url = _form_str("ollama_url")
    config.llama_url = _form_str("llama_url")
    config.claude_model = _form_str("claude_model")

    model_name = _form_str("model_name")
    # For backends that drive their own model selection (llama-server picks at
    # startup; Claude is selected via claude_model), leave the existing
    # config.model.name alone. For mlx/ollama, persist the user's choice.
    if backend in {"mlx", "ollama"} and model_name:
        config.model.name = model_name

    # Security: tool-gating policy. Only the mode is exposed in the
    # wizard ("audit" vs "enforce"); the blocked-tier and allow/deny
    # lists keep their saved/default values unless edited in config.toml.
    tool_policy = (form.get("tool_policy") or "").strip().lower()
    if tool_policy in {"audit", "enforce"}:
        config.security.tool_policy = tool_policy
    return True, None


# --------------------------------------------------------------------------- #
# Handlers                                                                    #
# --------------------------------------------------------------------------- #


async def _state_handler(_request: Request) -> JSONResponse:
    config = TowelConfig.load()
    return JSONResponse(
        {
            "config": config.model_dump(),
            "backends": await _probe_backends(config),
            "config_path": str(TOWEL_HOME / "config.toml"),
        }
    )


async def _models_handler(request: Request) -> JSONResponse:
    backend = request.path_params["backend"]
    config = TowelConfig.load()
    if backend == "mlx":
        return JSONResponse({"models": _list_mlx_cached_models()})
    if backend == "ollama":
        url = config.ollama_url or "http://localhost:11434"
        return JSONResponse({"models": await _list_ollama_models(url)})
    # llama-server and claude are valid backends in _VALID_BACKENDS
    # but neither has a meaningful "list installed models" concept
    # the wizard can probe — llama-server picks its model at
    # startup, claude is selected via the claude_model config field.
    # Return an empty list (not an error) so the wizard UI can
    # still query without tripping on these.
    if backend in {"llama", "claude"}:
        return JSONResponse({"models": []})
    # Unknown backend (typo, hand-rolled curl). Fail loud with the
    # list of supported values — otherwise an empty `models` list
    # silently masks a typo as "you have no models installed."
    return JSONResponse(
        {
            "error": f"unknown backend {backend!r}",
            "supported": ["mlx", "ollama", "llama", "claude"],
        },
        status_code=400,
    )


async def _save_handler(request: Request) -> JSONResponse:
    try:
        form = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"Invalid JSON: {exc}"}, status_code=400)
    # Top-level body must be an object — an array / string / null body
    # would crash inside `_apply_form_to_config` on `form.get(...)`
    # with an AttributeError and surface as HTTP 500. Same defensive
    # shape every other JSON-body endpoint applies.
    if not isinstance(form, dict):
        return JSONResponse(
            {"error": "body must be a JSON object"}, status_code=400,
        )

    config = TowelConfig.load()
    ok, err = _apply_form_to_config(config, form)
    if not ok:
        return JSONResponse({"error": err}, status_code=400)

    try:
        TOWEL_HOME.mkdir(parents=True, exist_ok=True)
        config.save()
    except Exception as exc:
        return JSONResponse({"error": f"Could not write config: {exc}"}, status_code=500)

    log.info("Setup wizard wrote %s (backend=%s)", TOWEL_HOME / "config.toml", config.backend)
    return JSONResponse(
        {
            "ok": True,
            "config_path": str(TOWEL_HOME / "config.toml"),
            "shutdown_after": True,
        }
    )


async def _setup_page_handler(_request: Request) -> HTMLResponse | FileResponse:
    page = _WEB_DIR / "setup.html"
    if page.exists():
        return FileResponse(page)
    return HTMLResponse("<h1>Setup page not found.</h1>", status_code=404)


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def setup_routes() -> list[Route | Mount]:
    """Routes that any Starlette app can mount to host the setup UI."""
    routes: list[Route | Mount] = [
        Route("/setup", _setup_page_handler),
        Route("/api/setup/state", _state_handler),
        Route("/api/setup/backends/{backend}/models", _models_handler),
        Route("/api/setup/save", _save_handler, methods=["POST"]),
    ]
    if _WEB_DIR.is_dir():
        routes.append(
            Mount("/setup-static", StaticFiles(directory=str(_WEB_DIR)), name="setup-static")
        )
    return routes


def build_app() -> Starlette:
    """Build the standalone setup app served by ``towel setup``."""
    routes = setup_routes()
    # Make ``/`` serve the wizard directly so the browser tab is friendly.
    routes.insert(0, Route("/", _setup_page_handler))
    return Starlette(routes=routes)


def run_standalone(host: str = "127.0.0.1", port: int = 18749, open_browser: bool = True) -> None:
    """Run the setup server until the user kills it."""
    import uvicorn

    # Surface INFO-level events (the "config saved" line is the operator's
    # confirmation that the wizard wrote ~/.towel/config.toml).
    from towel.logging_setup import configure_terminal_logging

    configure_terminal_logging()

    if open_browser:
        import threading
        import webbrowser

        url = f"http://{host}:{port}/"
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        log.info("Opening %s in your default browser…", url)

    uvicorn.run(build_app(), host=host, port=port, log_level="warning")

"""Remote worker client for controller-managed Towel execution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import socket
from dataclasses import dataclass, field
from typing import Any

import websockets

from towel.agent.conversation import Conversation
from towel.agent.events import AgentEvent

log = logging.getLogger("towel.gateway.worker")


@dataclass
class RemoteWorkerClient:
    """Connects to a controller and executes remote jobs."""

    master_url: str
    agent: Any
    worker_id: str
    capabilities: dict[str, Any] = field(default_factory=dict)
    reconnect_delay: float = 3.0
    max_reconnect_delay: float = 60.0
    heartbeat_interval: float = 15.0
    _jobs: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    _consecutive_failures: int = field(default=0, init=False)
    # Set when the controller asks the worker to exit (graceful model swap or
    # fleet rebalance). The reconnect loop checks this and stops re-attempting
    # rather than treating the close as a transient drop.
    _shutdown_requested: bool = field(default=False, init=False)

    async def run_forever(self) -> None:
        """Maintain a persistent worker connection to the controller."""
        while True:
            try:
                log.info("Connecting to controller %s ...", self.master_url)
                async with websockets.connect(
                    self.master_url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=10,
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "register",
                                "role": "worker",
                                "id": self.worker_id,
                                "capabilities": self.capabilities,
                            }
                        )
                    )
                    raw = await ws.recv()
                    resp = json.loads(raw)
                    if resp.get("type") != "registered":
                        raise RuntimeError("Worker registration was rejected")
                    log.info(
                        "Registered worker %s with controller %s", self.worker_id, self.master_url
                    )
                    self._consecutive_failures = 0
                    heartbeat = asyncio.create_task(self._heartbeat_loop(ws))

                    try:
                        async for raw in ws:
                            msg = json.loads(raw)
                            msg_type = msg.get("type")
                            if msg_type == "run":
                                job_id = msg["job_id"]
                                task = asyncio.create_task(self._run_job(ws, msg))
                                self._jobs[job_id] = task
                            elif msg_type == "infer":
                                job_id = msg["job_id"]
                                task = asyncio.create_task(self._run_inference(ws, msg))
                                self._jobs[job_id] = task
                            elif msg_type == "cancel_job":
                                await self._cancel_job(msg.get("job_id"))
                            elif msg_type == "ping":
                                await self._send_heartbeat(ws)
                            elif msg_type == "self_upgrade":
                                # Coordinator-initiated in-process upgrade. The
                                # worker is already connected, so this skips
                                # the launcher-daemon hop entirely — one round
                                # trip from the operator's click. On success
                                # self_upgrade_and_reexec re-execs and the
                                # reconnect loop comes back on the new code;
                                # on failure we just keep running.
                                strategy = msg.get("strategy") or "pip"
                                log.info(
                                    "Received self_upgrade from controller (strategy=%s)",
                                    strategy,
                                )
                                for jid, jtask in list(self._jobs.items()):
                                    if not jtask.done():
                                        jtask.cancel()
                                    self._jobs.pop(jid, None)
                                try:
                                    await ws.close(1000, "self-upgrade in progress")
                                except Exception:
                                    pass
                                from towel.launcher import self_upgrade_and_reexec
                                self_upgrade_and_reexec(strategy)
                                # If we get here, the upgrade failed. Break out
                                # so the reconnect loop kicks back in with the
                                # existing code.
                                break
                            elif msg_type == "shutdown":
                                # The controller is asking us to exit gracefully —
                                # used by /fleet/replace-worker to swap models or
                                # backends without leaving zombies. Cancel any
                                # active jobs, close the socket, and signal the
                                # reconnect loop to stop.
                                log.info(
                                    "Received shutdown from controller: %s",
                                    msg.get("reason") or "no reason given",
                                )
                                for jid, jtask in list(self._jobs.items()):
                                    if not jtask.done():
                                        jtask.cancel()
                                    self._jobs.pop(jid, None)
                                self._shutdown_requested = True
                                try:
                                    await ws.close(1000, "shutdown requested by controller")
                                except Exception:
                                    pass
                                break
                    finally:
                        heartbeat.cancel()
            except Exception as exc:
                self._consecutive_failures += 1
                log.warning("Worker connection lost: %s", exc)
            finally:
                for job_id, task in list(self._jobs.items()):
                    if not task.done():
                        task.cancel()
                    self._jobs.pop(job_id, None)

            if self._shutdown_requested:
                log.info("Shutdown requested — stopping reconnect loop.")
                return

            delay = self._backoff_delay()
            log.info("Reconnecting in %.1fs (attempt %d)...", delay, self._consecutive_failures)
            await asyncio.sleep(delay)

    def _backoff_delay(self) -> float:
        """Exponential backoff with jitter, capped at max_reconnect_delay."""
        exp = min(self._consecutive_failures, 10)
        base = min(self.reconnect_delay * (2 ** (exp - 1)), self.max_reconnect_delay)
        return base * (0.5 + random.random() * 0.5)

    async def _run_job(self, ws: Any, msg: dict[str, Any]) -> None:
        """Execute a remote generation job."""
        job_id = msg["job_id"]
        session_id = msg.get("session", "default")
        conversation = Conversation.from_dict(msg["conversation"])
        stream = msg.get("stream", True)

        # Apply coordinator's project context so the worker uses it
        project_ctx = msg.get("project_context")
        if project_ctx and hasattr(self.agent, "project_context"):
            self.agent.project_context = project_ctx

        try:
            if stream:
                async for event in self.agent.step_streaming(conversation):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "job_event",
                                "job_id": job_id,
                                "event": event.to_ws_message(session_id),
                            }
                        )
                    )
            else:
                response = await self.agent.step(conversation)
                conversation.messages.append(response)
                await ws.send(
                    json.dumps(
                        {
                            "type": "job_done",
                            "job_id": job_id,
                            "conversation": conversation.to_dict(),
                            "response": {
                                "content": response.content,
                                "metadata": response.metadata,
                            },
                        }
                    )
                )
                return

            await ws.send(
                json.dumps(
                    {
                        "type": "job_done",
                        "job_id": job_id,
                        "conversation": conversation.to_dict(),
                    }
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await ws.send(
                json.dumps(
                    {
                        "type": "job_error",
                        "job_id": job_id,
                        "message": str(exc),
                    }
                )
            )
        finally:
            self._jobs.pop(job_id, None)

    async def _run_inference(self, ws: Any, msg: dict[str, Any]) -> None:
        """Execute a remote inference-only job from a controller-built payload."""
        job_id = msg["job_id"]
        session_id = msg.get("session", "default")
        request = msg["request"]
        stream = msg.get("stream", True)

        try:
            if stream:
                full_text = ""
                async for chunk in self.agent.stream_from_request(request):
                    full_text += chunk
                    await ws.send(
                        json.dumps(
                            {
                                "type": "job_event",
                                "job_id": job_id,
                                "event": AgentEvent.token(chunk).to_ws_message(session_id),
                            }
                        )
                    )
                await ws.send(
                    json.dumps(
                        {
                            "type": "job_done",
                            "job_id": job_id,
                            "result": {"text": full_text, "metadata": {}},
                        }
                    )
                )
                return

            result = await self.agent.generate_from_request(request)
            metadata = {}
            if hasattr(result, "tokens_per_second"):
                metadata["tps"] = result.tokens_per_second
            # Two naming conventions in the runtimes: MLX uses
            # total_tokens / input_tokens / output_tokens; llama-server
            # uses prompt_tokens / completion_tokens. Probe both so
            # remote-worker responses come back with real counts
            # regardless of which backend the worker is running.
            if hasattr(result, "total_tokens"):
                metadata["tokens"] = result.total_tokens
            if hasattr(result, "input_tokens"):
                metadata["input_tokens"] = result.input_tokens
            if hasattr(result, "output_tokens"):
                metadata["output_tokens"] = result.output_tokens
            if hasattr(result, "prompt_tokens"):
                metadata.setdefault("input_tokens", result.prompt_tokens)
            if hasattr(result, "completion_tokens"):
                metadata.setdefault("output_tokens", result.completion_tokens)
            # When the backend only reports input/output, sum them
            # into `tokens` so /api/ask and the UI don't show 0.
            if "tokens" not in metadata and (
                "input_tokens" in metadata or "output_tokens" in metadata
            ):
                metadata["tokens"] = (
                    metadata.get("input_tokens", 0)
                    + metadata.get("output_tokens", 0)
                )
            # Empty-text + non-empty tool_calls fallback: when the
            # model decides to invoke a tool instead of producing
            # prose, generate_from_request returns text="" with
            # populated tool_calls. The coordinator's quick-infer
            # path doesn't surface a tool channel, so the caller
            # would see "" with no explanation. Format the tool-call
            # intent as text so at least SOMETHING comes back.
            response_text = result.text or ""
            tool_calls = getattr(result, "tool_calls", None) or []
            if not response_text.strip() and tool_calls:
                lines = ["(The model emitted tool calls instead of text:)"]
                for tc in tool_calls:
                    name = getattr(tc, "name", str(tc))
                    args = getattr(tc, "arguments", "")
                    lines.append(f"  - {name}({args})")
                response_text = "\n".join(lines)
                metadata["empty_text_tool_call_fallback"] = True
            await ws.send(
                json.dumps(
                    {
                        "type": "job_done",
                        "job_id": job_id,
                        "result": {"text": response_text, "metadata": metadata},
                    }
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await ws.send(
                json.dumps(
                    {
                        "type": "job_error",
                        "job_id": job_id,
                        "message": str(exc),
                    }
                )
            )
        finally:
            self._jobs.pop(job_id, None)

    async def _cancel_job(self, job_id: str | None) -> None:
        """Cancel an active remote job if present."""
        if not job_id:
            return
        self.agent.cancel()
        task = self._jobs.get(job_id)
        if task and not task.done():
            task.cancel()

    async def _send_heartbeat(self, ws: Any) -> None:
        """Send a liveness heartbeat back to the controller.

        Refreshes ``live_resources`` (load average, free RAM) on every tick so
        the coordinator's scoring and fleet UI see current load, not just the
        static counts captured at startup.
        """
        caps = dict(self.capabilities)
        caps["live_resources"] = _detect_live_resources()
        # Keep the running record current too, so a future caller reading
        # ``self.capabilities`` sees the latest snapshot.
        self.capabilities["live_resources"] = caps["live_resources"]
        await ws.send(
            json.dumps(
                {
                    "type": "heartbeat",
                    "id": self.worker_id,
                    "capabilities": caps,
                }
            )
        )

    async def _heartbeat_loop(self, ws: Any) -> None:
        """Send periodic heartbeats while connected."""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self._send_heartbeat(ws)
            except Exception:
                log.debug("Heartbeat send failed, connection likely closing")
                return


def _detect_llama_model(llama_url: str) -> dict[str, Any] | None:
    """Query a running llama-server for its actual model metadata."""
    try:
        import httpx

        resp = httpx.get(f"{llama_url}/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data") or data.get("models") or []
        if models:
            m = models[0]
            meta = m.get("meta", {})
            return {
                "model_id": m.get("id", ""),
                "n_params": meta.get("n_params", 0),
                "n_ctx_train": meta.get("n_ctx_train", 0),
                "n_embd": meta.get("n_embd", 0),
                "size_bytes": meta.get("size", 0),
            }
    except Exception:
        pass
    return None


def _detect_live_resources() -> dict[str, Any]:
    """Sample dynamic load metrics. Refreshed on every heartbeat.

    Keys returned (any may be absent if unsupported on the host):
      - ``load_avg_1min``  — 1-minute load average from ``os.getloadavg``
      - ``cpu_pressure``   — load_avg_1min / cpu_count, capped at 1.0; a rough
                             "how close to fully loaded" signal the
                             coordinator can mix into worker scoring.
      - ``ram_available_mb`` — current free RAM from /proc/meminfo (Linux) or
                             sysctl vm_stat (macOS).
    """
    out: dict[str, Any] = {}
    try:
        load1, _load5, _load15 = os.getloadavg()
        out["load_avg_1min"] = round(load1, 2)
        cpu_count = os.cpu_count() or 0
        if cpu_count > 0:
            out["cpu_pressure"] = round(min(load1 / cpu_count, 1.0), 3)
    except (AttributeError, OSError):
        # os.getloadavg is unavailable on Windows
        pass

    try:
        import platform

        system = platform.system()
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        out["ram_available_mb"] = int(line.split()[1]) // 1024
                        break
        elif system == "Darwin":
            import subprocess

            # vm_stat reports pages free; multiply by page size for bytes.
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                pages_free = 0
                pages_inactive = 0
                page_size = 4096
                for line in result.stdout.splitlines():
                    if "page size of" in line:
                        try:
                            page_size = int(line.split()[-2])
                        except (ValueError, IndexError):
                            pass
                    elif "Pages free:" in line:
                        try:
                            pages_free = int(line.split()[-1].rstrip("."))
                        except ValueError:
                            pass
                    elif "Pages inactive:" in line:
                        try:
                            pages_inactive = int(line.split()[-1].rstrip("."))
                        except ValueError:
                            pass
                avail_bytes = (pages_free + pages_inactive) * page_size
                if avail_bytes:
                    out["ram_available_mb"] = avail_bytes // (1024 * 1024)
    except Exception:
        # Live RAM sampling is best-effort — never block the heartbeat path.
        pass

    # Disk free space on the user's home filesystem (where HF cache and
    # ~/.ollama live). Lets the coordinator warn before scheduling a
    # download that won't fit.
    try:
        import shutil
        from pathlib import Path

        usage = shutil.disk_usage(str(Path.home()))
        out["disk_free_gb"] = round(usage.free / (1024**3), 1)
        out["disk_total_gb"] = round(usage.total / (1024**3), 1)
    except Exception:
        pass

    return out


def _detect_system_resources() -> dict[str, Any]:
    """Detect RAM and CPU info for the current machine."""
    resources: dict[str, Any] = {"hostname": socket.gethostname()}
    try:
        import os

        resources["cpu_count"] = os.cpu_count() or 0
    except Exception:
        pass
    try:
        import shutil

        total, used, free = shutil.disk_usage("/")
        # Try /proc/meminfo (Linux) or sysctl (macOS) for RAM
        import platform

        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        resources["ram_total_mb"] = int(line.split()[1]) // 1024
                    elif line.startswith("MemAvailable:"):
                        resources["ram_available_mb"] = int(line.split()[1]) // 1024
        elif platform.system() == "Darwin":
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            if result.returncode == 0:
                resources["ram_total_mb"] = int(result.stdout.strip()) // (1024 * 1024)
    except Exception:
        pass
    return resources


def default_worker_capabilities(
    config: Any,
    backend: str,
    allow_tools: bool,
    llama_url: str = "",
) -> dict[str, Any]:
    """Describe this worker's runtime so the controller can schedule it.

    For llama/ollama backends, queries the running server to detect the
    actual model instead of relying on the global config (which may be
    stale or point to a different model).
    """
    if backend == "claude":
        mode = "anthropic_messages"
    elif backend == "ollama":
        mode = "ollama_chat"
    elif backend == "llama":
        mode = "llama_chat"
    else:
        mode = "mlx_prompt"

    model_name = getattr(config.model, "name", "")
    context_window = getattr(config.model, "context_window", 0)
    max_tokens = getattr(config.model, "max_tokens", 0)

    # For llama backend, detect the actual running model
    llama_meta: dict[str, Any] = {}
    if backend == "llama" and llama_url:
        detected = _detect_llama_model(llama_url)
        if detected:
            llama_meta = detected
            # Use detected model name and training context if available
            if detected["model_id"]:
                model_name = detected["model_id"]
            if detected["n_ctx_train"] > 0 and context_window <= 0:
                context_window = detected["n_ctx_train"]

    caps: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "backend": backend,
        "model": model_name,
        "modes": [mode],
        "context_window": context_window,
        "max_tokens": max_tokens,
        "tools": allow_tools,
    }

    if llama_meta:
        caps["model_meta"] = llama_meta

    # Add system resource info (RAM, CPU)
    caps["resources"] = _detect_system_resources()

    # Add GPU info if available
    try:
        from towel.agent.discovery import detect_gpus

        gpus = detect_gpus()
        if gpus:
            caps["gpus"] = [{"name": g.name, "vram_mb": g.vram_mb} for g in gpus]
            caps["total_vram_mb"] = sum(g.vram_mb for g in gpus)
    except Exception:
        pass

    # Model inventory — what this worker could actually run without a fresh
    # download. Lets the coordinator pick a model the worker already has,
    # and avoid sending a 70B request to a Pi.
    caps["available_models"] = _detect_available_models(backend, llama_url)
    # Advertise a co-located launcher so the coordinator's replace/upgrade
    # UI doesn't need the operator to retype the URL for every worker. The
    # worker reports the externally-routable form (hostname + port) since
    # 127.0.0.1 is useless from the coordinator's process.
    launcher_url = _detect_local_launcher()
    if launcher_url:
        caps["launcher_url"] = launcher_url
    # Rough size cap derived from advertised VRAM + RAM. A 4-bit quant
    # needs ≈ 0.6 GB per billion params; we leave 50% headroom for the
    # KV cache and activations. RAM-only nodes can still run small CPU
    # models, just slower.
    vram_mb = int(caps.get("total_vram_mb") or 0)
    ram_mb = int((caps.get("resources") or {}).get("ram_total_mb") or 0)
    # Prefer VRAM where it exists; fall back to half of system RAM.
    usable_mb = vram_mb if vram_mb else (ram_mb // 2)
    if usable_mb:
        caps["max_param_b_est"] = round(usable_mb / 1024.0 / 0.6, 1)
    else:
        caps["max_param_b_est"] = 0.0

    return caps


def _detect_local_launcher() -> str | None:
    """Detect a co-located ``towel launcher`` daemon on this host.

    Strategy:
    1. If ``$TOWEL_LAUNCHER_URL`` is set, trust it verbatim — operator
       knows their network topology better than we can guess.
    2. Otherwise probe ``http://127.0.0.1:18751/health`` with a short
       timeout. The launcher's ``/health`` endpoint is unauthenticated
       and returns 200 instantly.
    3. If the probe succeeds, return the externally-routable form
       (``http://<hostname>:18751``) because 127.0.0.1 means nothing
       to the coordinator process that ultimately calls this URL.

    Returns ``None`` when no launcher is reachable so the capability key
    stays absent rather than carrying a misleading value.
    """
    override = os.environ.get("TOWEL_LAUNCHER_URL", "").strip()
    if override:
        return override

    from towel.launcher import DEFAULT_PORT

    try:
        import httpx

        resp = httpx.get(
            f"http://127.0.0.1:{DEFAULT_PORT}/health", timeout=0.5
        )
        if resp.status_code != 200:
            return None
    except Exception:
        return None

    return f"http://{socket.gethostname()}:{DEFAULT_PORT}"


def _detect_available_models(backend: str, llama_url: str) -> list[str]:
    """Enumerate models the worker can run *without* a fresh download.

    Lets the coordinator pick a target the worker has already cached and
    skip launching ``towel worker --model X`` against a host that would
    need to pull tens of gigabytes first. Returned list is best-effort —
    empty doesn't mean "nothing supported", just "couldn't enumerate".
    """
    found: list[str] = []
    try:
        if backend == "ollama":
            import httpx

            url = "http://localhost:11434"
            try:
                resp = httpx.get(f"{url}/api/tags", timeout=2.0)
                if resp.status_code == 200:
                    for entry in resp.json().get("models") or []:
                        name = entry.get("name")
                        if name:
                            found.append(name)
            except Exception:
                pass
        elif backend == "mlx":
            import os
            from pathlib import Path

            roots = [
                Path.home() / ".cache" / "huggingface" / "hub",
                Path(os.environ.get("HF_HOME", "")) / "hub"
                if os.environ.get("HF_HOME")
                else None,
            ]
            for root in roots:
                if not root or not root.is_dir():
                    continue
                for entry in root.iterdir():
                    if entry.name.startswith("models--"):
                        stripped = entry.name[len("models--") :]
                        if "--" in stripped:
                            org, _, name = stripped.partition("--")
                            found.append(f"{org}/{name}")
        elif backend == "llama" and llama_url:
            # llama-server exposes /v1/models; we already query it for model
            # metadata in _detect_llama_model. Reuse that signal.
            meta = _detect_llama_model(llama_url)
            if meta and meta.get("model_id"):
                found.append(meta["model_id"])
        elif backend == "claude":
            # Claude API supports a fixed set of model aliases.
            found.extend(["sonnet", "opus", "haiku"])
    except Exception:
        pass

    # Dedupe while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for name in found:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique

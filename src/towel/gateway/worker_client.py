"""Remote worker client for controller-managed Towel execution."""

from __future__ import annotations

import asyncio
import json
import logging
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
            if hasattr(result, "total_tokens"):
                metadata["tokens"] = result.total_tokens
            if hasattr(result, "input_tokens"):
                metadata["input_tokens"] = result.input_tokens
            if hasattr(result, "output_tokens"):
                metadata["output_tokens"] = result.output_tokens
            await ws.send(
                json.dumps(
                    {
                        "type": "job_done",
                        "job_id": job_id,
                        "result": {"text": result.text, "metadata": metadata},
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
        """Send a liveness heartbeat back to the controller."""
        await ws.send(
            json.dumps(
                {
                    "type": "heartbeat",
                    "id": self.worker_id,
                    "capabilities": self.capabilities,
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

    return caps

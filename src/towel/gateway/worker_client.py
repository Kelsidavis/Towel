"""Remote worker client for controller-managed Towel execution."""

from __future__ import annotations

import asyncio
import json
import logging
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
    heartbeat_interval: float = 15.0
    _jobs: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    async def run_forever(self) -> None:
        """Maintain a persistent worker connection to the controller."""
        while True:
            try:
                async with websockets.connect(self.master_url) as ws:
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
                    log.info("Registered worker %s with controller %s", self.worker_id, self.master_url)
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
                log.warning("Worker connection dropped: %s", exc)
            finally:
                for job_id, task in list(self._jobs.items()):
                    if not task.done():
                        task.cancel()
                    self._jobs.pop(job_id, None)
            await asyncio.sleep(self.reconnect_delay)

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
            await self._send_heartbeat(ws)


def default_worker_capabilities(config: Any, backend: str, allow_tools: bool) -> dict[str, Any]:
    """Describe this worker's runtime so the controller can schedule it."""
    if backend == "claude":
        mode = "anthropic_messages"
    elif backend == "ollama":
        mode = "ollama_chat"
    else:
        mode = "mlx_prompt"
    return {
        "hostname": socket.gethostname(),
        "backend": backend,
        "model": getattr(config.model, "name", ""),
        "modes": [mode],
        "context_window": getattr(config.model, "context_window", 0),
        "max_tokens": getattr(config.model, "max_tokens", 0),
        "tools": allow_tools,
    }

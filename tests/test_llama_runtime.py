"""Tests for the llama.cpp runtime adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from towel.agent.discovery import GGUFModel
from towel.agent.llama_runtime import LlamaRuntime
from towel.config import TowelConfig


def test_auto_start_disables_llama_fit(monkeypatch):
    created: dict[str, object] = {}

    class FakeManagedLlamaServer:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.url = f"http://localhost:{kwargs['port']}"

        def start(self):
            created["started"] = True

        async def wait_healthy(self):
            created["waited"] = True

    def fake_detect_system():
        return SimpleNamespace(
            has_llama_server=True,
            llama_server_path="/usr/bin/llama-server",
            best_model=GGUFModel(
                path=Path("/models/qwen.gguf"),
                size_gb=14.5,
                name="qwen",
            ),
        )

    monkeypatch.setattr(
        "towel.agent.discovery.ManagedLlamaServer",
        FakeManagedLlamaServer,
    )
    monkeypatch.setattr(
        "towel.agent.discovery.detect_system",
        fake_detect_system,
    )

    runtime = LlamaRuntime(TowelConfig(), llama_url="http://localhost:18081")
    monkeypatch.setattr(runtime, "_check_health", lambda: asyncio.sleep(0, result=False))

    asyncio.run(runtime.load_model())

    assert created["extra_args"] == ["--fit", "off", "-c", "32768"]
    assert created["started"] is True
    assert created["waited"] is True


def _runtime_with_tools():
    """A LlamaRuntime backed by the full builtin skill registry.

    Exercises the tools-attached vs tools-omitted branches of
    build_inference_request against a realistic tool set.
    """
    from towel.skills.builtin import register_builtins
    from towel.skills.registry import SkillRegistry

    reg = SkillRegistry()
    register_builtins(reg)
    return LlamaRuntime(TowelConfig(), skills=reg, llama_url="http://localhost:8080")


def _request_for(rt, text):
    from towel.agent.conversation import Conversation, Role

    conv = Conversation(id="t")
    conv.add(Role.USER, text)
    return rt.build_inference_request(conv)


def test_build_request_chat_omits_tools_and_thinking():
    rt = _runtime_with_tools()
    req = _request_for(rt, "hi there")
    # Pure chat: no tool payload, thinking suppressed (reasoning_effort=none).
    assert "tools" not in req
    assert req.get("reasoning_effort") == "none"


def test_build_request_explain_omits_tools_keeps_fast():
    rt = _runtime_with_tools()
    req = _request_for(rt, "Explain what a towel is for, one sentence.")
    assert "tools" not in req
    assert req.get("reasoning_effort") == "none"


def test_build_request_tool_task_attaches_tools_no_thinking():
    rt = _runtime_with_tools()
    req = _request_for(rt, "fetch https://example.com and show the title")
    # Tool-heavy task: tools attached, but no slow <think> phase.
    assert req.get("tools")
    assert req.get("reasoning_effort") == "none"


def test_build_request_reasoning_task_thinks():
    rt = _runtime_with_tools()
    req = _request_for(rt, "Analyze the tradeoffs of mutexes versus channels.")
    # Reasoning task (analyze): thinking enabled (no reasoning_effort=none).
    # Tools are also attached — analyze frequently means "analyze this repo",
    # which needs to read files, so the classifier errs toward keeping tools
    # rather than silently disarming a work request (see TASK_REQUIREMENTS).
    assert "reasoning_effort" not in req
    assert req.get("tools")

"""Pipelines — chain tools together in reusable workflows.

A pipeline is a sequence of tool calls where each step's output
feeds into the next step's input. Think Unix pipes but for AI tools.

Usage:
    pipe = Pipeline("deploy-check", [
        PipeStep("git_status"),
        PipeStep("lint_file", {"path": "src/main.py"}),
        PipeStep("security_scan", {"path": "."}),
    ])
    result = await pipe.run(registry)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("towel.agent.pipelines")


@dataclass
class PipeStep:
    """A single step in a pipeline."""
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    input_key: str | None = None  # which arg receives previous step's output
    condition: str | None = None  # skip if previous output contains this string
    result: str = ""
    elapsed: float = 0.0
    status: str = "pending"


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    name: str
    steps: list[PipeStep]
    total_elapsed: float = 0.0

    @property
    def success(self) -> bool:
        return all(s.status == "completed" for s in self.steps)

    def summary(self) -> str:
        lines = [f"Pipeline: {self.name} ({self.total_elapsed:.1f}s)"]
        for i, s in enumerate(self.steps):
            icon = {"completed":"✓","failed":"✗","skipped":"○","pending":" "}.get(s.status,"?")
            preview = s.result[:60].replace("\n"," ") if s.result else ""
            lines.append(f"  [{icon}] {i+1}. {s.tool} ({s.elapsed:.1f}s) {preview}")
        return "\n".join(lines)


class Pipeline:
    """A reusable chain of tool calls."""

    def __init__(self, name: str, steps: list[PipeStep]) -> None:
        self.name = name
        self.steps = steps

    async def run(self, registry: Any) -> PipelineResult:
        start = time.perf_counter()
        prev_output = ""

        for i, step in enumerate(self.steps):
            # Check condition
            if step.condition and step.condition in prev_output:
                step.status = "skipped"
                continue

            # Inject previous output
            args = dict(step.args)
            if step.input_key and prev_output:
                args[step.input_key] = prev_output

            # Execute
            step_start = time.perf_counter()
            try:
                result = await registry.execute_tool(step.tool, args)
                step.result = str(result)
                step.status = "completed"
                prev_output = step.result
            except Exception as e:
                step.result = f"Error: {e}"
                step.status = "failed"
                log.error(f"Pipeline {self.name} step {i} failed: {e}")
                break
            finally:
                step.elapsed = time.perf_counter() - step_start

        return PipelineResult(
            name=self.name,
            steps=self.steps,
            total_elapsed=time.perf_counter() - start,
        )


# Built-in pipeline templates
BUILTIN_PIPELINES: dict[str, list[dict]] = {
    "security-audit": [
        {"tool": "scan_secrets", "args": {"path": "."}},
        {"tool": "lint_file", "args": {"path": "."}, "input_key": None},
        {"tool": "typo_check_file", "args": {"path": "src/main.py"}},
    ],
    "project-health": [
        {"tool": "git_status"},
        {"tool": "pip_venv_info"},
        {"tool": "system_info"},
    ],
    "morning-briefing": [
        {"tool": "weather_now", "args": {"city": "Portland"}},
        {"tool": "hn_top", "args": {"limit": 5}},
        {"tool": "random_quote"},
    ],
}


def get_pipeline(name: str) -> Pipeline | None:
    """Get a built-in pipeline by name."""
    spec = BUILTIN_PIPELINES.get(name)
    if not spec:
        return None
    steps = [PipeStep(tool=s["tool"], args=s.get("args", {}), input_key=s.get("input_key")) for s in spec]
    return Pipeline(name, steps)


def list_pipelines() -> list[str]:
    return list(BUILTIN_PIPELINES.keys())

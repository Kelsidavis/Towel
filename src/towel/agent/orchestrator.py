"""Agent orchestrator — spawn and coordinate specialist sub-agents.

Enables multi-agent workflows where a coordinator agent delegates
subtasks to specialists (coder, researcher, reviewer, writer) and
synthesizes their results.

Usage:
    orch = Orchestrator(config, skills)
    result = await orch.run("Build a REST API for user management", [
        AgentTask(role="architect", prompt="Design the API schema"),
        AgentTask(role="coder", prompt="Implement the endpoints"),
        AgentTask(role="reviewer", prompt="Review the code for issues"),
    ])
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from towel.agent.conversation import Conversation, Role
from towel.config import TowelConfig

log = logging.getLogger("towel.agent.orchestrator")


@dataclass
class AgentTask:
    """A subtask to be executed by a specialist agent."""
    role: str  # e.g., "coder", "researcher", "reviewer", "writer"
    prompt: str
    depends_on: list[int] = field(default_factory=list)  # indices of tasks this depends on
    context: str = ""  # additional context injected from parent or dependencies
    result: str = ""
    status: str = "pending"  # pending, running, completed, failed
    elapsed: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "prompt": self.prompt[:100],
            "status": self.status,
            "elapsed": f"{self.elapsed:.1f}s",
            "result_length": len(self.result),
        }


@dataclass
class OrchestratorResult:
    """Result of a multi-agent orchestration run."""
    tasks: list[AgentTask]
    synthesis: str = ""
    total_elapsed: float = 0.0

    @property
    def success(self) -> bool:
        return all(t.status == "completed" for t in self.tasks)

    def summary(self) -> str:
        lines = [f"Orchestration: {len(self.tasks)} tasks, {self.total_elapsed:.1f}s total"]
        for i, t in enumerate(self.tasks):
            icon = {"completed": "+", "failed": "!", "running": "~", "pending": " "}.get(t.status, "?")
            lines.append(f"  [{icon}] {i}. {t.role}: {t.status} ({t.elapsed:.1f}s, {len(t.result)} chars)")
        return "\n".join(lines)


# Role-specific system prompts
ROLE_PROMPTS: dict[str, str] = {
    "coder": (
        "You are an expert software engineer. Write clean, production-quality code. "
        "Include error handling, types, and brief comments for complex logic. "
        "Output code in fenced blocks with the language specified."
    ),
    "researcher": (
        "You are a thorough research analyst. Find relevant information, cite sources, "
        "and present findings in a structured format. Be comprehensive but concise."
    ),
    "reviewer": (
        "You are a senior code reviewer. Analyze code for bugs, security issues, "
        "performance problems, and style. Be specific — cite line numbers. "
        "Rate overall quality 1-10."
    ),
    "writer": (
        "You are a technical writer. Write clear, well-structured documentation. "
        "Use headers, bullet points, and code examples where appropriate."
    ),
    "architect": (
        "You are a software architect. Design systems with clear separation of concerns, "
        "scalability, and maintainability. Provide schemas, data flow diagrams (as ASCII), "
        "and API specifications."
    ),
    "tester": (
        "You are a QA engineer. Write comprehensive tests covering edge cases, error "
        "conditions, and typical usage. Use the appropriate testing framework."
    ),
    "debugger": (
        "You are a debugging expert. Analyze errors systematically — identify root causes, "
        "explain why the bug occurs, and provide verified fixes."
    ),
    "default": (
        "You are a helpful AI assistant. Be concise and accurate."
    ),
}


class Orchestrator:
    """Coordinates multiple specialist agents on a complex task."""

    def __init__(self, config: TowelConfig, skills: Any = None, memory: Any = None) -> None:
        self.config = config
        self.skills = skills
        self.memory = memory

    async def run(self, goal: str, tasks: list[AgentTask]) -> OrchestratorResult:
        """Execute a sequence of agent tasks, respecting dependencies."""
        start = time.perf_counter()
        result = OrchestratorResult(tasks=tasks)

        log.info(f"Orchestrating {len(tasks)} tasks for: {goal[:80]}")

        for i, task in enumerate(tasks):
            # Wait for dependencies
            for dep_idx in task.depends_on:
                if dep_idx < len(tasks) and tasks[dep_idx].status != "completed":
                    log.warning(f"Task {i} depends on incomplete task {dep_idx}")

            # Inject dependency results as context
            dep_context = ""
            if task.depends_on:
                dep_results = []
                for dep_idx in task.depends_on:
                    if dep_idx < len(tasks) and tasks[dep_idx].result:
                        dep_results.append(
                            f"[Result from {tasks[dep_idx].role} (task {dep_idx})]:\n"
                            f"{tasks[dep_idx].result}"
                        )
                if dep_results:
                    dep_context = "\n\n".join(dep_results) + "\n\n"

            # Build the agent prompt
            full_prompt = ""
            if dep_context:
                full_prompt += f"Context from previous tasks:\n{dep_context}\n"
            if task.context:
                full_prompt += f"{task.context}\n\n"
            full_prompt += f"Goal: {goal}\n\nYour task: {task.prompt}"

            # Execute
            task.status = "running"
            task_start = time.perf_counter()

            try:
                task.result = await self._run_agent(task.role, full_prompt)
                task.status = "completed"
            except Exception as e:
                task.result = f"Error: {e}"
                task.status = "failed"
                log.error(f"Task {i} ({task.role}) failed: {e}")

            task.elapsed = time.perf_counter() - task_start
            log.info(f"Task {i} ({task.role}): {task.status} in {task.elapsed:.1f}s")

        result.total_elapsed = time.perf_counter() - start

        # Synthesize results
        if result.success and len(tasks) > 1:
            synthesis_parts = [f"# Results for: {goal}\n"]
            for i, t in enumerate(tasks):
                synthesis_parts.append(f"## {t.role.title()} (Task {i})\n{t.result}\n")
            result.synthesis = "\n".join(synthesis_parts)

        return result

    async def run_parallel(self, goal: str, tasks: list[AgentTask]) -> OrchestratorResult:
        """Execute independent tasks in parallel."""
        start = time.perf_counter()

        async def _exec(i: int, task: AgentTask) -> None:
            task.status = "running"
            task_start = time.perf_counter()
            try:
                full_prompt = f"Goal: {goal}\n\nYour task: {task.prompt}"
                task.result = await self._run_agent(task.role, full_prompt)
                task.status = "completed"
            except Exception as e:
                task.result = f"Error: {e}"
                task.status = "failed"
            task.elapsed = time.perf_counter() - task_start

        await asyncio.gather(*[_exec(i, t) for i, t in enumerate(tasks)])

        result = OrchestratorResult(tasks=tasks, total_elapsed=time.perf_counter() - start)
        return result

    async def _run_agent(self, role: str, prompt: str) -> str:
        """Run a single agent step with role-specific system prompt."""
        # Create a temporary config with role-specific identity
        import copy

        from towel.agent.runtime import AgentRuntime
        agent_config = copy.deepcopy(self.config)
        agent_config.identity = ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])

        runtime = AgentRuntime(agent_config, skills=self.skills, memory=self.memory)
        conv = Conversation(channel=f"orchestrator:{role}")
        conv.add(Role.USER, prompt)

        response = await runtime.step(conv)
        return response.content

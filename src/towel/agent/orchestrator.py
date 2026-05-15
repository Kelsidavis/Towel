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
from typing import Any, Protocol

from towel.agent.conversation import Conversation, Role
from towel.config import TowelConfig

log = logging.getLogger("towel.agent.orchestrator")


class RoleDispatcher(Protocol):
    """Protocol the Orchestrator uses to dispatch a single role task.

    Implemented by the gateway server so each orchestrator subtask can land
    on the best-fit remote worker (per `_route_by_role`) instead of running
    locally on the coordinator. Defined as a Protocol so the agent package
    stays free of a hard dependency on the gateway package — that
    direction would create a circular import (gateway → orchestrator →
    gateway).
    """

    async def dispatch_role_task(
        self,
        role: str,
        role_system: str,
        prompt: str,
        *,
        session_id: str,
        max_tokens: int,
        temperature: float,
        with_tools: bool,
    ) -> str:
        ...


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
    # When True the subtask runs through the worker's tool loop so it
    # can call write_file/read_file/edit_file. Defaults False so simple
    # text-only roles (writer, default) stay on the faster path.
    with_tools: bool = False
    # How many times this subtask retried before completing or failing.
    # Surfaced so operators reading the response body can see when the
    # cluster needed multiple workers to satisfy a request.
    attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "prompt": self.prompt[:100],
            "status": self.status,
            "elapsed": f"{self.elapsed:.1f}s",
            "result_length": len(self.result),
            "attempts": self.attempts,
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
            icon = {"completed": "+", "failed": "!", "running": "~", "pending": " "}.get(
                t.status, "?"
            )
            lines.append(
                f"  [{icon}] {i}. {t.role}: {t.status} ({t.elapsed:.1f}s, {len(t.result)} chars)"
            )
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
    "default": ("You are a helpful AI assistant. Be concise and accurate."),
}


class Orchestrator:
    """Coordinates multiple specialist agents on a complex task.

    With `dispatcher` set, each role's subtask is dispatched to the best-fit
    remote worker via the gateway's routing pipeline — so a "coder" subtask
    can land on the bigger worker while a "writer" subtask runs in parallel
    on a smaller one. Without `dispatcher`, falls back to a local
    AgentRuntime per subtask (useful for tests and single-node setups).
    """

    def __init__(
        self,
        config: TowelConfig,
        skills: Any = None,
        memory: Any = None,
        dispatcher: RoleDispatcher | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.config = config
        self.skills = skills
        self.memory = memory
        self.dispatcher = dispatcher
        # Single retry by default. Mirrors `/api/ask`'s primary→alt
        # fallback: if a worker emits empty text or times out, a second
        # attempt typically lands on the alternate worker (since the
        # first is now busy/draining) and succeeds. Setting this to 1
        # disables retries, which is occasionally useful for explicit
        # benchmarking of a particular worker.
        self.max_attempts = max(1, int(max_attempts))

    async def _execute_with_retry(self, task: AgentTask, full_prompt: str) -> None:
        """Run a subtask, retrying once on failure.

        Updates `task` in place — populates result/status/elapsed/attempts.
        Captures the last error message as the result on terminal failure
        so the caller and downstream synthesis still see what went wrong.
        """
        task_start = time.perf_counter()
        task.status = "running"
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            task.attempts = attempt + 1
            try:
                task.result = await self._run_agent(
                    task.role, full_prompt, with_tools=task.with_tools,
                )
                task.status = "completed"
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                log.warning(
                    "Task (%s, attempt %d/%d) failed: %s",
                    task.role, attempt + 1, self.max_attempts, e,
                )
                # Brief yield between attempts so the dispatcher can
                # release the failed worker's slot — without this, a
                # tight retry on the same event-loop tick lands back on
                # the same worker that just failed. Same idea as the
                # `await asyncio.sleep(0)` after `_preempt_idle_task`.
                await asyncio.sleep(0)
        if last_exc is not None:
            task.result = f"Error: {last_exc}"
            task.status = "failed"
        task.elapsed = time.perf_counter() - task_start

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
            await self._execute_with_retry(task, full_prompt)
            log.info(
                "Task %d (%s): %s in %.1fs (attempts=%d)",
                i, task.role, task.status, task.elapsed, task.attempts,
            )

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

        async def _exec(i: int, task: AgentTask) -> None:  # noqa: ARG001
            full_prompt = f"Goal: {goal}\n\nYour task: {task.prompt}"
            await self._execute_with_retry(task, full_prompt)

        await asyncio.gather(*[_exec(i, t) for i, t in enumerate(tasks)])

        result = OrchestratorResult(tasks=tasks, total_elapsed=time.perf_counter() - start)
        return result

    async def _run_agent(
        self, role: str, prompt: str, *, with_tools: bool = False,
    ) -> str:
        """Run a single agent step with role-specific system prompt.

        Uses the configured remote dispatcher when present so each role's
        subtask can land on the best-fit worker; otherwise falls back to
        a local AgentRuntime. ``with_tools`` flips the dispatcher onto
        the tool-loop path so the subtask can call write_file etc.
        """
        role_system = ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])

        if self.dispatcher is not None:
            # Per-subtask session keeps role contexts isolated — a coder
            # subtask shouldn't reuse the writer's affinity-pinned worker.
            import uuid
            session_id = f"orch-{role}-{uuid.uuid4().hex[:8]}"
            return await self.dispatcher.dispatch_role_task(
                role,
                role_system,
                prompt,
                session_id=session_id,
                max_tokens=2048,
                temperature=0.4,
                with_tools=with_tools,
            )

        # Local fallback: spin up a coordinator-side AgentRuntime with
        # the role's identity. Used by tests and single-node deployments.
        import copy

        from towel.agent.runtime import AgentRuntime

        agent_config = copy.deepcopy(self.config)
        agent_config.identity = role_system

        runtime = AgentRuntime(agent_config, skills=self.skills, memory=self.memory)
        conv = Conversation(channel=f"orchestrator:{role}")
        conv.add(Role.USER, prompt)

        response = await runtime.step(conv)
        return response.content

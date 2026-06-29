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


class WorkerDispatchError(RuntimeError):
    """Raised by `RoleDispatcher.dispatch_role_task` on failure.

    Carries the worker id (if a worker was picked) so the orchestrator
    can exclude it on the next retry attempt. Without this, retries
    against a flaky-but-pickable worker bounce back to the same worker
    every time — the dispatcher's session affinity is fresh per
    subtask but the task_type routing still steers to the same
    prefer_quality/prefer_fast pick.
    """

    def __init__(self, message: str, *, worker_id: str | None = None) -> None:
        super().__init__(message)
        self.worker_id = worker_id


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
        task_type: str | None,
        exclude_workers: set[str] | None,
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
    # When set, the orchestrator extracts the first fenced code block
    # from the subtask's response and writes it to this workspace-
    # relative path. Lets a chat-fast (no-tools) coder produce code
    # without needing the slow tool loop — the worker emits a fenced
    # block, the coordinator writes the file. Lives in `AgentTask`
    # rather than on the subtask prompt so callers can use the same
    # response.
    extract_to: str | None = None
    extracted_path: str | None = None
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
            icon = {
                "completed": "+",
                "failed": "!",
                "skipped": "/",
                "running": "~",
                "pending": " ",
            }.get(t.status, "?")
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


# Map orchestrator roles to TaskType strings the dispatcher recognises.
# Without this, the workspace preamble the orchestrator prepends to every
# subtask prompt prevents the keyword classifier from triggering — the
# prompt no longer starts with "write …" or "plan …", so it falls all
# the way through to None and dispatches via role_match. Role_match
# happens to pick the biggest INFERENCE worker (which is fine for
# coder) but skips the dispatcher's prefer_quality preempt path that
# would, e.g., pull SparklesMint off an idle task for an architect
# request. Explicit mapping closes the gap.
ROLE_TASK_TYPES: dict[str, str] = {
    "architect": "plan",
    "coder": "generate",
    "researcher": "research",
    "reviewer": "code_review",
    "writer": "draft",
    "tester": "test_gen",
    "debugger": "analyze",
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

    async def _execute_with_retry(
        self,
        task: AgentTask,
        full_prompt: str,
        *,
        workspace_dir: str | None = None,
    ) -> None:
        """Run a subtask, retrying once on failure.

        Updates `task` in place — populates result/status/elapsed/attempts.
        Captures the last error message as the result on terminal failure
        so the caller and downstream synthesis still see what went wrong.

        When a `WorkerDispatchError` carries a worker_id, that worker is
        excluded from the next attempt's dispatch — so the cluster's
        prefer_quality routing doesn't bounce back to the exact worker
        that just timed out. Other RuntimeErrors (no worker available,
        etc.) don't exclude anything since there's no worker to blame.

        When the task has `extract_to` set and `workspace_dir` is
        provided, the extraction + validation runs INSIDE the retry
        loop — a written file with a SyntaxError raises to trigger
        another attempt rather than leaving broken code on disk.
        Model-quality issues are often stochastic; re-rolling the
        same prompt frequently succeeds where the first try didn't.
        """
        task_start = time.perf_counter()
        task.status = "running"
        last_exc: Exception | None = None
        exclude_workers: set[str] = set()
        for attempt in range(self.max_attempts):
            task.attempts = attempt + 1
            try:
                task.result = await self._run_agent(
                    task.role, full_prompt,
                    with_tools=task.with_tools,
                    exclude_workers=exclude_workers,
                )
                # Extract-and-validate runs in-loop so a write that
                # fails syntax check counts as an attempt and triggers
                # the next retry rather than a terminal "failed" task.
                if task.extract_to and workspace_dir:
                    self._extract_and_write(task, workspace_dir)
                task.status = "completed"
                last_exc = None
                break
            except WorkerDispatchError as e:
                last_exc = e
                if e.worker_id:
                    exclude_workers.add(e.worker_id)
                log.warning(
                    "Task (%s, attempt %d/%d) failed on %s: %s",
                    task.role, attempt + 1, self.max_attempts,
                    e.worker_id or "no-worker", e,
                )
                await asyncio.sleep(0)
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

    @staticmethod
    def _extract_and_write(task: AgentTask, workspace_dir: str) -> None:
        """Pull the first fenced code block out of `task.result` and
        write it to `workspace_dir / task.extract_to`.

        Handles three common shapes the model produces:
          ```python\n...```      — language tag we strip
          ```\n...```            — no language tag
          {code with no fences}  — falls through, writes the whole
                                   stripped result if no fence found

        On a successful write `task.extracted_path` is populated with
        the absolute path so callers downstream can read it back.
        """
        import re
        from pathlib import Path
        if task.extract_to is None:
            return
        text = task.result or ""
        # Match fenced blocks; tolerate language tags and trailing
        # whitespace. DOTALL so newlines in the body are kept.
        match = re.search(
            r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```",
            text,
            re.DOTALL,
        )
        body = match.group(1) if match else text.strip()
        if not body.endswith("\n"):
            body += "\n"
        # Reject path traversal — task.extract_to should land inside
        # the workspace.
        target = (Path(workspace_dir) / task.extract_to).resolve()
        ws_root = Path(workspace_dir).resolve()
        if ws_root not in target.parents and target != ws_root:
            raise ValueError(
                f"extract_to path {task.extract_to!r} resolves outside "
                f"workspace {workspace_dir}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        task.extracted_path = str(target)
        # Syntax-validate when the file looks like Python — catches the
        # common failure mode where the model emits almost-valid code
        # with a stray bracket or import line, which Codex would catch
        # via py_compile and we'd previously discover only at run time.
        # ast.parse is in-process and ~1ms; cheap enough to always run.
        # Validation failure raises so the orchestrator retries this
        # subtask on a different worker (model-quality issue is often
        # stochastic — a re-roll succeeds where the first didn't).
        if target.suffix == ".py":
            import ast
            try:
                tree = ast.parse(body)
            except SyntaxError as exc:
                raise ValueError(
                    f"extract_to wrote {target.name} but it has a "
                    f"SyntaxError on line {exc.lineno}: {exc.msg}"
                ) from exc
            # ast.parse accepts a bare identifier ("write_file") as
            # valid Python — it's a no-op expression statement. Live
            # observation: a coder subtask returned the literal text
            # `write_file` (the tool name) and that passed parsing,
            # producing an 11-byte file. Require at least one
            # substantive top-level construct so empty-or-degenerate
            # bodies trigger a retry. `import` covers stubs that just
            # re-export; `def`/`class`/`Assign`/`AnnAssign` cover the
            # real cases.
            has_substance = any(
                isinstance(
                    node,
                    (
                        ast.FunctionDef
                        | ast.AsyncFunctionDef
                        | ast.ClassDef
                        | ast.Assign
                        | ast.AnnAssign
                        | ast.Import
                        | ast.ImportFrom
                        | ast.If
                        | ast.For
                        | ast.While
                        | ast.Try
                        | ast.With
                    ),
                )
                for node in tree.body
            )
            if not has_substance:
                raise ValueError(
                    f"extract_to wrote {target.name} but it has no "
                    "substantive code (no def/class/assignment/import) — "
                    f"got {body[:80]!r}"
                )

    @staticmethod
    def _workspace_preamble(workspace_dir: str | None) -> str:
        """Prefix subtask prompts with a workspace-directive when set.

        Subtasks share state via files in this directory: a coder writes
        ``game.py`` there, and a downstream tester reads it back. Tool
        execution happens on the coordinator, so a single absolute path
        works for every subtask regardless of which worker runs it.
        """
        if not workspace_dir:
            return ""
        return (
            f"Shared workspace: {workspace_dir}\n"
            "Use the filesystem tools (write_file, read_file, edit_file, "
            "list_directory) against this directory so other subtasks "
            "in this orchestration can see your work. Prefer relative "
            "paths under the workspace; absolute paths outside it should "
            "be avoided unless the goal explicitly requires it.\n\n"
        )

    async def run(
        self,
        goal: str,
        tasks: list[AgentTask],
        *,
        workspace_dir: str | None = None,
    ) -> OrchestratorResult:
        """Execute a sequence of agent tasks, respecting dependencies."""
        start = time.perf_counter()
        result = OrchestratorResult(tasks=tasks)

        log.info(f"Orchestrating {len(tasks)} tasks for: {goal[:80]}")

        workspace_preamble = self._workspace_preamble(workspace_dir)

        for i, task in enumerate(tasks):
            # Short-circuit when a direct dependency didn't succeed.
            # Without this, the dependent runs with the failed dep's
            # error string injected as `Result from <role>` context —
            # the worker either reasons against a misleading "result"
            # or wastes time refusing the prompt. Marking the task
            # `skipped` makes the failure cascade visible in the
            # response and saves the worker turn.
            failed_deps = [
                d for d in task.depends_on
                if 0 <= d < len(tasks) and tasks[d].status in ("failed", "skipped")
            ]
            if failed_deps:
                task.status = "skipped"
                task.result = (
                    f"Skipped: depends on task(s) {failed_deps} which did "
                    "not complete successfully."
                )
                task.elapsed = 0.0
                task.attempts = 0
                log.info(
                    "Task %d (%s): skipped (failed deps %s)",
                    i, task.role, failed_deps,
                )
                continue

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
            full_prompt = workspace_preamble
            if dep_context:
                full_prompt += f"Context from previous tasks:\n{dep_context}\n"
            if task.context:
                full_prompt += f"{task.context}\n\n"
            full_prompt += f"Goal: {goal}\n\nYour task: {task.prompt}"

            # Execute (extract-and-validate happens INSIDE the retry
            # loop when extract_to is set, so a syntax-error in the
            # written file triggers another attempt instead of leaving
            # broken code on disk).
            await self._execute_with_retry(
                task, full_prompt, workspace_dir=workspace_dir,
            )
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

    async def run_parallel(
        self,
        goal: str,
        tasks: list[AgentTask],
        *,
        workspace_dir: str | None = None,
    ) -> OrchestratorResult:
        """Execute independent tasks in parallel."""
        start = time.perf_counter()
        workspace_preamble = self._workspace_preamble(workspace_dir)

        async def _exec(i: int, task: AgentTask) -> None:  # noqa: ARG001
            full_prompt = (
                f"{workspace_preamble}Goal: {goal}\n\nYour task: {task.prompt}"
            )
            await self._execute_with_retry(
                task, full_prompt, workspace_dir=workspace_dir,
            )

        await asyncio.gather(*[_exec(i, t) for i, t in enumerate(tasks)])

        result = OrchestratorResult(tasks=tasks, total_elapsed=time.perf_counter() - start)
        return result

    async def _run_agent(
        self, role: str, prompt: str, *,
        with_tools: bool = False,
        exclude_workers: set[str] | None = None,
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
                task_type=ROLE_TASK_TYPES.get(role),
                exclude_workers=exclude_workers,
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

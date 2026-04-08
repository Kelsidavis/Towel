"""Idle task system — keep workers productive when no user requests are pending.

Workers automatically pick up background tasks when idle. These tasks are
low-priority and get cancelled instantly when a real request arrives.

Idle tasks run as agent jobs on the worker — the coordinator sends a synthetic
prompt and collects results. Results are cached so they're ready when a user
asks about them (e.g. "any lint issues?" can be answered from the cache).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger("towel.gateway.idle")


class IdleTask(Enum):
    """Background tasks workers can run when idle."""

    # ── Code quality ──────────────────────────────────────────────────
    LINT = "lint"                  # Run linter on project files
    TEST = "test"                  # Run test suite
    TYPE_CHECK = "type_check"     # Run type checker
    INDEX = "index"               # Index codebase for search
    SECURITY_SCAN = "security"    # Check for common vulnerabilities
    DEPS_AUDIT = "deps_audit"     # Check for outdated/vulnerable deps
    TODO_SCAN = "todo_scan"       # Find TODOs/FIXMEs in codebase
    DOC_CHECK = "doc_check"       # Check for missing docstrings

    # ── Personal assistant ────────────────────────────────────────────
    EMAIL_TRIAGE = "email_triage"     # Check and summarize unread email
    EMAIL_DRAFTS = "email_drafts"     # Draft replies to important emails
    SOCIAL_DIGEST = "social_digest"   # Summarize social media activity
    NEWS_DIGEST = "news_digest"       # Tech news relevant to user's work
    CALENDAR_PREP = "calendar_prep"   # Summarize upcoming events, prep notes
    PROACTIVE_HELP = "proactive_help" # Analyze recent work, suggest improvements

    def __str__(self) -> str:
        return self.value


# Prompts that the coordinator sends to workers for each idle task.
# These are synthetic user messages — the worker runs them through
# its normal agent loop with tools enabled.
IDLE_TASK_PROMPTS: dict[IdleTask, str] = {
    IdleTask.LINT: (
        "Run the project linter silently and report only errors and warnings. "
        "Use the appropriate linter for the project (ruff, flake8, eslint, clippy, etc). "
        "Be concise — list file:line:message format. If clean, say 'No lint issues.'"
    ),
    IdleTask.TEST: (
        "Run the project test suite and report results. Show only failures and errors. "
        "If all pass, say 'All N tests passed.' with the count."
    ),
    IdleTask.TYPE_CHECK: (
        "Run the type checker (mypy, pyright, tsc, etc) and report only errors. "
        "Be concise — list file:line:message format. If clean, say 'No type errors.'"
    ),
    IdleTask.INDEX: (
        "Scan the project directory structure and list key files: entry points, configs, "
        "test files, and the main source directories. Output a concise tree."
    ),
    IdleTask.SECURITY_SCAN: (
        "Check the project for common security issues: hardcoded secrets, SQL injection risks, "
        "insecure defaults, and exposed credentials. Report only findings. "
        "If clean, say 'No security issues found.'"
    ),
    IdleTask.DEPS_AUDIT: (
        "Check project dependencies for known vulnerabilities or outdated packages. "
        "Use pip-audit, npm audit, cargo audit, or equivalent. Report only issues found."
    ),
    IdleTask.TODO_SCAN: (
        "Find all TODO, FIXME, HACK, and XXX comments in the codebase. "
        "List each as file:line:comment. Be concise."
    ),
    IdleTask.DOC_CHECK: (
        "Check for public functions and classes missing docstrings in the main source directory. "
        "List each as file:line:name. Only report missing ones, skip private/internal."
    ),
    IdleTask.EMAIL_TRIAGE: (
        "Check for unread emails. Summarize each with: sender, subject, urgency (high/medium/low), "
        "and a one-line summary. Flag anything that needs immediate attention. "
        "Group by urgency. If no unread emails, say 'Inbox clear.'"
    ),
    IdleTask.EMAIL_DRAFTS: (
        "Review recent important emails that need replies. For each, draft a concise reply. "
        "Present as: original subject, sender, draft reply. Focus on emails requiring action."
    ),
    IdleTask.SOCIAL_DIGEST: (
        "Check recent social media activity — mentions, DMs, notifications. "
        "Summarize anything relevant or requiring attention. "
        "Focus on professional/tech-related activity. Skip noise."
    ),
    IdleTask.NEWS_DIGEST: (
        "Find recent tech news relevant to the user's projects and interests. "
        "Focus on: AI/ML updates, tools the user works with, security advisories, "
        "and industry trends. List 3-5 items with one-line summaries and links."
    ),
    IdleTask.CALENDAR_PREP: (
        "Check upcoming calendar events for the next 24 hours. For each meeting: "
        "summarize the topic, list attendees, and prepare 2-3 bullet points of "
        "context or talking points based on recent project activity."
    ),
    IdleTask.PROACTIVE_HELP: (
        "Review the user's recent git commits, open issues, and project state. "
        "Identify 2-3 actionable suggestions: things that could be improved, "
        "tasks that might be forgotten, or patterns that suggest a problem. "
        "Be specific and practical, not generic."
    ),
}

# How often each task can re-run (seconds). Prevents redundant work.
IDLE_TASK_COOLDOWNS: dict[IdleTask, float] = {
    IdleTask.LINT: 300,           # 5 minutes
    IdleTask.TEST: 600,           # 10 minutes
    IdleTask.TYPE_CHECK: 300,     # 5 minutes
    IdleTask.INDEX: 1800,         # 30 minutes
    IdleTask.SECURITY_SCAN: 3600, # 1 hour
    IdleTask.DEPS_AUDIT: 3600,    # 1 hour
    IdleTask.TODO_SCAN: 900,      # 15 minutes
    IdleTask.DOC_CHECK: 1800,     # 30 minutes
    IdleTask.EMAIL_TRIAGE: 600,    # 10 minutes
    IdleTask.EMAIL_DRAFTS: 1800,   # 30 minutes
    IdleTask.SOCIAL_DIGEST: 1800,  # 30 minutes
    IdleTask.NEWS_DIGEST: 3600,    # 1 hour
    IdleTask.CALENDAR_PREP: 1800,  # 30 minutes
    IdleTask.PROACTIVE_HELP: 3600, # 1 hour
}

# Priority order — most valuable tasks first
IDLE_TASK_PRIORITY: list[IdleTask] = [
    # Personal assistant — user-facing value first
    IdleTask.EMAIL_TRIAGE,
    IdleTask.CALENDAR_PREP,
    # Code quality
    IdleTask.LINT,
    IdleTask.TEST,
    IdleTask.TYPE_CHECK,
    # More PA
    IdleTask.PROACTIVE_HELP,
    IdleTask.EMAIL_DRAFTS,
    IdleTask.NEWS_DIGEST,
    IdleTask.SOCIAL_DIGEST,
    # Lower priority code tasks
    IdleTask.TODO_SCAN,
    IdleTask.SECURITY_SCAN,
    IdleTask.DEPS_AUDIT,
    IdleTask.INDEX,
    IdleTask.DOC_CHECK,
]


@dataclass
class IdleTaskResult:
    """Cached result of a background idle task."""

    task: IdleTask
    worker_id: str
    output: str
    timestamp: float = field(default_factory=time.time)
    error: bool = False

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": str(self.task),
            "worker_id": self.worker_id,
            "output": self.output,
            "timestamp": self.timestamp,
            "age_seconds": round(self.age_seconds),
            "error": self.error,
        }


class IdleTaskManager:
    """Manages background idle task scheduling and result caching."""

    def __init__(self) -> None:
        self._results: dict[IdleTask, IdleTaskResult] = {}
        self._last_run: dict[IdleTask, float] = {}
        self._active: dict[str, IdleTask] = {}  # worker_id → running idle task
        # Per-worker idle task config. None = use defaults from assigned tasks.
        self._worker_idle_tasks: dict[str, list[IdleTask] | None] = {}

    def set_worker_idle_tasks(self, worker_id: str, tasks: list[IdleTask] | None) -> None:
        """Override which idle tasks a worker should run. None = auto."""
        self._worker_idle_tasks[worker_id] = tasks

    def get_worker_idle_tasks(self, worker_id: str) -> list[IdleTask] | None:
        return self._worker_idle_tasks.get(worker_id)

    def next_task_for_worker(
        self,
        worker_id: str,
        has_tools: bool,
        assigned_tasks: list[Any] | None = None,
    ) -> IdleTask | None:
        """Pick the next idle task for a worker based on priority and cooldown.

        Returns None if no tasks are ready.
        """
        # Determine which idle tasks this worker should run
        override = self._worker_idle_tasks.get(worker_id)
        if override is not None:
            allowed = override
        else:
            # Default: tool-requiring idle tasks need tools on the worker
            allowed = []
            for task in IDLE_TASK_PRIORITY:
                # All idle tasks except INDEX need tools
                if task == IdleTask.INDEX or has_tools:
                    allowed.append(task)

        now = time.time()
        for task in IDLE_TASK_PRIORITY:
            if task not in allowed:
                continue
            cooldown = IDLE_TASK_COOLDOWNS.get(task, 300)
            last = self._last_run.get(task, 0)
            if now - last < cooldown:
                continue
            # Don't run if another worker is already doing this task
            if task in self._active.values():
                continue
            return task
        return None

    def start_task(self, worker_id: str, task: IdleTask) -> None:
        """Mark an idle task as running on a worker."""
        self._active[worker_id] = task
        self._last_run[task] = time.time()

    def cancel_task(self, worker_id: str) -> IdleTask | None:
        """Cancel idle task on a worker. Returns the task that was running."""
        return self._active.pop(worker_id, None)

    def complete_task(self, worker_id: str, output: str, error: bool = False) -> None:
        """Record completed idle task result."""
        task = self._active.pop(worker_id, None)
        if task is None:
            return
        self._results[task] = IdleTaskResult(
            task=task,
            worker_id=worker_id,
            output=output,
            error=error,
        )
        log.info("Idle task %s completed on %s (%d chars)", task, worker_id, len(output))

    def is_idle_task(self, worker_id: str) -> bool:
        """Check if a worker is running an idle task (not a real request)."""
        return worker_id in self._active

    def get_result(self, task: IdleTask) -> IdleTaskResult | None:
        return self._results.get(task)

    def all_results(self) -> dict[str, Any]:
        return {str(t): r.to_dict() for t, r in self._results.items()}

    def active_tasks(self) -> dict[str, str]:
        return {wid: str(task) for wid, task in self._active.items()}

"""Eval — benchmark agent performance on test prompts.

Run a set of test prompts through the agent and measure:
- Response quality (keyword matching)
- Tool usage accuracy
- Speed (tokens/sec)
- Consistency across runs
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    """A single evaluation test case."""
    prompt: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    max_seconds: float = 30.0
    # Results
    response: str = ""
    tools_called: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    passed: bool = False
    score: float = 0.0
    notes: str = ""


@dataclass
class EvalResult:
    """Results from an eval run."""
    cases: list[EvalCase]
    total_elapsed: float = 0.0

    @property
    def pass_rate(self) -> float:
        if not self.cases: return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    @property
    def avg_score(self) -> float:
        if not self.cases: return 0.0
        return sum(c.score for c in self.cases) / len(self.cases)

    def summary(self) -> str:
        lines = [
            f"Eval: {len(self.cases)} cases, {self.pass_rate:.0%} pass rate, "
            f"avg score {self.avg_score:.1f}/1.0, {self.total_elapsed:.1f}s total",
            "",
        ]
        for i, c in enumerate(self.cases):
            icon = "✓" if c.passed else "✗"
            lines.append(f"  [{icon}] {i+1}. {c.prompt[:50]}...")
            lines.append(f"       Score: {c.score:.2f} · {c.elapsed:.1f}s · {c.notes}")
        return "\n".join(lines)


# Built-in eval suite
BUILTIN_EVALS: list[dict] = [
    {"prompt": "What is 2 + 2?", "expected_keywords": ["4"]},
    {"prompt": "What day of the week is it?",
     "expected_keywords": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
    {"prompt": "Generate a UUID for me", "expected_tools": ["generate_uuid"]},
    {"prompt": "What's the SHA256 hash of 'hello'?",
     "expected_keywords": ["2cf24dba"], "expected_tools": ["hash_text"]},
    {"prompt": "Roll 2d6", "expected_tools": ["roll_dice"], "expected_keywords": ["Rolling"]},
    {"prompt": "Convert 100 USD to EUR", "expected_tools": ["currency_convert"]},
    {"prompt": "What is my current working directory?", "expected_keywords": ["/"]},
    {"prompt": "List the files in the current directory", "expected_tools": ["list_directory", "run_command"]},
]


def score_case(case: EvalCase) -> None:
    """Score a completed eval case."""
    points = 0.0
    total = 0.0

    # Keyword matching
    if case.expected_keywords:
        total += 1.0
        matched = sum(1 for k in case.expected_keywords if k.lower() in case.response.lower())
        # Any keyword match counts (they're alternatives for day-of-week etc.)
        if matched > 0:
            points += 1.0

    # Tool usage
    if case.expected_tools:
        total += 1.0
        matched = sum(1 for t in case.expected_tools if t in case.tools_called)
        if matched > 0:
            points += 1.0

    # Speed bonus
    if case.elapsed < case.max_seconds:
        total += 0.5
        points += 0.5

    # Non-empty response
    total += 0.5
    if len(case.response.strip()) > 10:
        points += 0.5

    case.score = points / total if total > 0 else 0.0
    case.passed = case.score >= 0.5

    # Notes
    notes = []
    if case.expected_keywords and not any(k.lower() in case.response.lower() for k in case.expected_keywords):
        notes.append("missing keywords")
    if case.expected_tools and not any(t in case.tools_called for t in case.expected_tools):
        notes.append(f"expected tools: {case.expected_tools}")
    case.notes = ", ".join(notes) if notes else "ok"

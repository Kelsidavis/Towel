"""A/B testing — compare two models or prompts side by side.

Run the same prompt through two configurations and compare
responses, speed, and quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ABResult:
    """Result from one side of an A/B test."""

    label: str
    response: str = ""
    tokens: int = 0
    tps: float = 0.0
    elapsed: float = 0.0
    error: str = ""


@dataclass
class ABTestResult:
    """Complete A/B test comparison."""

    prompt: str
    a: ABResult = field(default_factory=lambda: ABResult(label="A"))
    b: ABResult = field(default_factory=lambda: ABResult(label="B"))

    def summary(self) -> str:
        lines = [f"A/B Test: {self.prompt[:60]}...\n"]
        for side in [self.a, self.b]:
            if side.error:
                lines.append(f"  [{side.label}] ERROR: {side.error}")
            else:
                lines.append(
                    f"  [{side.label}] {side.elapsed:.1f}s, "
                    f"{side.tokens} tokens, {side.tps:.1f} tok/s"
                )
                lines.append(f"       {side.response[:100]}...")
        # Winner
        if self.a.tps > 0 and self.b.tps > 0:
            faster = "A" if self.a.elapsed < self.b.elapsed else "B"
            lines.append(
                f"\n  Faster: {faster} ({abs(self.a.elapsed - self.b.elapsed):.1f}s difference)"
            )
        return "\n".join(lines)

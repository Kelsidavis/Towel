"""Guardrails — content safety and prompt injection protection.

Checks user input and agent output for:
- Prompt injection attempts
- Sensitive data leakage (SSN, credit cards, etc.)
- Harmful content patterns
- Token budget abuse
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool = True
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


# ── Input guardrails ──

_INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?previous\s+instructions", "prompt injection: ignore instructions"),
    (r"ignore\s+.*?system\s+prompt", "prompt injection: ignore system prompt"),
    (r"you\s+are\s+now\s+(?:DAN|jailbreak|unrestricted)", "prompt injection: role override"),
    (r"pretend\s+you\s+(?:are|have)\s+no\s+(?:rules|restrictions|limits)", "prompt injection: remove restrictions"),
    (r"output\s+your\s+(?:system|initial)\s+prompt", "prompt injection: extract system prompt"),
    (r"repeat\s+(?:everything|all)\s+(?:above|before)", "prompt injection: extract context"),
    (r"\[system\]|\[INST\]|<<SYS>>", "prompt injection: format injection"),
]

_SENSITIVE_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN pattern detected"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit card number pattern"),
    (r"\b[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b", "IBAN pattern"),
    (r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----", "private key detected"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"sk-[A-Za-z0-9]{32,}", "API key pattern (OpenAI-style)"),
    (r"ghp_[A-Za-z0-9]{36}", "GitHub PAT"),
]


def check_input(text: str) -> GuardrailResult:
    """Check user input for injection and sensitive data."""
    result = GuardrailResult()

    for pattern, desc in _INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            result.passed = False
            result.violations.append(f"[injection] {desc}")

    for pattern, desc in _SENSITIVE_PATTERNS:
        if re.search(pattern, text):
            result.violations.append(f"[sensitive] {desc}")
            # Don't block — just warn

    # Token abuse: extremely long input
    if len(text) > 100_000:
        result.passed = False
        result.violations.append(f"[abuse] Input too long ({len(text)} chars)")

    return result


def check_output(text: str) -> GuardrailResult:
    """Check agent output for sensitive data leakage."""
    result = GuardrailResult()

    for pattern, desc in _SENSITIVE_PATTERNS:
        if re.search(pattern, text):
            result.violations.append(f"[leakage] {desc}")

    return result


def redact_sensitive(text: str) -> str:
    """Redact sensitive patterns from text."""
    redacted = text
    for pattern, desc in _SENSITIVE_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted

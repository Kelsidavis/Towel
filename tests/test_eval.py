"""Tests for agent evaluation."""
import pytest
from towel.agent.eval import EvalCase, EvalResult, score_case, BUILTIN_EVALS


class TestEvalCase:
    def test_score_keyword_match(self):
        c = EvalCase(prompt="test", expected_keywords=["hello"], response="hello world", elapsed=1.0)
        score_case(c)
        assert c.passed
        assert c.score > 0.5

    def test_score_keyword_miss(self):
        c = EvalCase(prompt="test", expected_keywords=["xyz"], response="no match here", elapsed=1.0)
        score_case(c)
        assert c.score < 1.0

    def test_score_tool_match(self):
        c = EvalCase(prompt="test", expected_tools=["hash_text"],
                     tools_called=["hash_text"], response="sha256: abc", elapsed=1.0)
        score_case(c)
        assert c.passed

    def test_score_empty_response(self):
        c = EvalCase(prompt="test", response="", elapsed=1.0)
        score_case(c)
        assert c.score < 1.0


class TestEvalResult:
    def test_pass_rate(self):
        cases = [
            EvalCase(prompt="a", passed=True),
            EvalCase(prompt="b", passed=True),
            EvalCase(prompt="c", passed=False),
        ]
        r = EvalResult(cases=cases)
        assert abs(r.pass_rate - 0.667) < 0.01

    def test_summary(self):
        r = EvalResult(cases=[EvalCase(prompt="test", passed=True, score=0.8, elapsed=1.0, notes="ok")])
        assert "test" in r.summary()


class TestBuiltinEvals:
    def test_has_cases(self):
        assert len(BUILTIN_EVALS) >= 8

    def test_cases_valid(self):
        for e in BUILTIN_EVALS:
            assert "prompt" in e

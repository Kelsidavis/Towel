"""Tests for A/B testing."""

from towel.agent.ab_test import ABResult, ABTestResult


class TestABResult:
    def test_defaults(self):
        r = ABResult(label="A")
        assert r.response == ""
        assert r.tokens == 0


class TestABTestResult:
    def test_summary(self):
        r = ABTestResult(prompt="test prompt")
        r.a = ABResult(label="model-a", response="hello", elapsed=1.0, tokens=10, tps=10.0)
        r.b = ABResult(label="model-b", response="world", elapsed=2.0, tokens=20, tps=10.0)
        s = r.summary()
        assert "model-a" in s
        assert "model-b" in s
        assert "Faster: A" in s

    def test_summary_with_error(self):
        r = ABTestResult(prompt="test")
        r.a = ABResult(label="A", error="failed")
        s = r.summary()
        assert "ERROR" in s

    def test_cli_registered(self):
        from towel.cli.main import cli

        assert "ab-test" in [c.name for c in cli.commands.values()]

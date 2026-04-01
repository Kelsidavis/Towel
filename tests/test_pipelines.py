"""Tests for tool pipelines."""

from towel.agent.pipelines import PipelineResult, PipeStep, get_pipeline, list_pipelines


class TestPipeStep:
    def test_defaults(self):
        s = PipeStep(tool="test_tool")
        assert s.status == "pending"
        assert s.args == {}


class TestPipelineResult:
    def test_success(self):
        r = PipelineResult(
            name="test",
            steps=[
                PipeStep(tool="a", status="completed"),
                PipeStep(tool="b", status="completed"),
            ],
        )
        assert r.success

    def test_failure(self):
        r = PipelineResult(
            name="test",
            steps=[
                PipeStep(tool="a", status="completed"),
                PipeStep(tool="b", status="failed"),
            ],
        )
        assert not r.success

    def test_summary(self):
        r = PipelineResult(
            name="test",
            steps=[
                PipeStep(tool="a", status="completed", elapsed=1.2, result="ok"),
            ],
            total_elapsed=1.5,
        )
        s = r.summary()
        assert "test" in s
        assert "✓" in s


class TestBuiltinPipelines:
    def test_list(self):
        names = list_pipelines()
        assert "security-audit" in names
        assert "project-health" in names
        assert "morning-briefing" in names

    def test_get(self):
        p = get_pipeline("project-health")
        assert p is not None
        assert len(p.steps) == 3

    def test_get_unknown(self):
        assert get_pipeline("nonexistent") is None

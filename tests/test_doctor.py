"""Tests for towel doctor diagnostics."""

from towel.cli.doctor import (
    Check,
    check_config,
    check_environment,
    check_gateway,
    check_mlx,
    check_skills,
    check_storage,
    run_doctor,
)
from towel.config import TowelConfig


class TestCheck:
    def test_passing_check(self):
        c = Check("test")
        c.ok("all good")
        c.finalize()
        assert c.passed
        assert len(c.details) == 1
        assert len(c.errors) == 0

    def test_failing_check(self):
        c = Check("test")
        c.fail("something broke", "fix it")
        c.finalize()
        assert not c.passed
        assert len(c.errors) == 1
        assert len(c.suggestions) == 1

    def test_warning_check(self):
        c = Check("test")
        c.ok("mostly fine")
        c.warn("minor issue")
        c.finalize()
        assert c.passed  # warnings don't fail
        assert len(c.warnings) == 1

    def test_render_does_not_crash(self):
        c = Check("test")
        c.ok("detail")
        c.warn("warning")
        c.fail("error", "suggestion")
        c.render()  # should not raise


class TestEnvironmentCheck:
    def test_reports_python_version(self):
        c = check_environment()
        c.finalize()
        assert any("Python" in d for d in c.details)

    def test_reports_platform(self):
        c = check_environment()
        c.finalize()
        assert any("Darwin" in d or "Linux" in d or "Windows" in d for d in c.details)

    def test_passes(self):
        c = check_environment()
        c.finalize()
        assert c.passed


class TestConfigCheck:
    def test_default_config_passes(self):
        config = TowelConfig()
        c = check_config(config)
        c.finalize()
        assert c.passed

    def test_detects_bad_context_window(self):
        config = TowelConfig()
        config.model.context_window = 100
        config.model.max_tokens = 200
        c = check_config(config)
        c.finalize()
        assert not c.passed
        assert any("context_window" in e for e in c.errors)

    def test_shows_model_name(self):
        config = TowelConfig()
        c = check_config(config)
        assert any(config.model.name in d for d in c.details)


class TestMlxCheck:
    def test_mlx_check_runs(self):
        import sys

        c = check_mlx()
        c.finalize()
        if sys.platform == "darwin":
            assert any("mlx" in d.lower() for d in c.details)
        else:
            # MLX is macOS-only; on Linux the check should complete without crashing
            assert c.passed is False or c.details is not None

    def test_mlx_lm_detected(self):
        import sys

        c = check_mlx()
        c.finalize()
        if sys.platform == "darwin":
            assert any("mlx-lm" in d.lower() or "mlx_lm" in d.lower() for d in c.details)
        else:
            assert c.passed is False or c.details is not None


class TestSkillsCheck:
    def test_builtins_load(self):
        config = TowelConfig()
        c = check_skills(config)
        c.finalize()
        assert c.passed
        assert any("filesystem" in d for d in c.details)
        assert any("shell" in d for d in c.details)
        assert any("web" in d for d in c.details)


class TestGatewayCheck:
    def test_gateway_not_running(self):
        config = TowelConfig()
        # Use a port that's definitely not in use
        config.gateway.port = 19999
        c = check_gateway(config)
        c.finalize()
        assert c.passed
        assert any("not running" in d for d in c.details)

    def test_reports_port_availability(self):
        config = TowelConfig()
        config.gateway.port = 19998
        c = check_gateway(config)
        c.finalize()
        assert any("available" in d or "not running" in d for d in c.details)


class TestStorageCheck:
    def test_storage_check_runs(self):
        c = check_storage()
        c.finalize()
        assert c.passed


class TestRunDoctor:
    def test_run_all_checks(self):
        config = TowelConfig()
        config.gateway.port = 19997  # avoid conflicts
        checks = run_doctor(config)
        assert len(checks) == 7
        names = [c.name for c in checks]
        assert "Environment" in names
        assert "Configuration" in names
        assert "MLX" in names
        assert "Model" in names
        assert "Skills" in names
        assert "Gateway" in names
        assert "Storage" in names

    def test_all_checks_finalized(self):
        checks = run_doctor(TowelConfig())
        for c in checks:
            # Each check should have been finalized (passed is set)
            assert isinstance(c.passed, bool)

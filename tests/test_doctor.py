"""Tests for towel doctor diagnostics."""

from towel.cli.doctor import (
    Check,
    check_config,
    check_environment,
    check_gateway,
    check_memory_store,
    check_mlx,
    check_persisted_worker_state,
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

    def test_doctor_exits_nonzero_on_failure(self):
        """``towel doctor`` previously always returned exit 0 — a
        broken environment looked identical to a clean one from the
        shell's perspective, defeating the point of running it in a
        CI script or pre-commit hook. Now any failed check causes a
        non-zero exit so scripted callers can detect setup issues."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from towel.cli.main import cli

        # Inject a synthetic failing check by patching run_doctor.
        c_failing = Check("synthetic-failure")
        c_failing.fail("this check intentionally failed", "fix it")
        c_passing = Check("synthetic-pass")
        c_passing.ok("clean")

        runner = CliRunner()
        with patch(
            "towel.cli.doctor.run_doctor",
            return_value=[c_failing, c_passing],
        ):
            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 1
        # Render still happened so the operator sees what failed.
        assert "synthetic-failure" in result.output

    def test_doctor_exits_zero_when_only_warnings(self):
        """Warnings shouldn't fail the script — they're advisories,
        not blockers. Only ``fail()`` calls should drive a non-zero
        exit so a fleet with one flaky worker doesn't break every
        CI run."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from towel.cli.main import cli

        c_warned = Check("with-warning")
        c_warned.ok("mostly fine")
        c_warned.warn("but here's a thing")

        runner = CliRunner()
        with patch(
            "towel.cli.doctor.run_doctor",
            return_value=[c_warned],
        ):
            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0

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

    def test_stuck_worker_count_surfaces_as_warning(self, monkeypatch):
        """The /health response carries `workers.stuck` — busy workers
        wedged ≥5min on a request that won't return. The fleet panel
        renders this with a red border, but operators running
        `towel doctor` from the CLI had no equivalent signal until
        now. Translate the count into an actionable warn + suggestion
        so doctor matches the panel's visibility."""
        from unittest.mock import MagicMock, patch

        from towel.cli.doctor import check_gateway

        config = TowelConfig()

        resp = MagicMock()
        resp.json.return_value = {
            "status": "hoopy",
            "connections": 1,
            "sessions": 3,
            "workers": {
                "total": 2, "busy": 2, "idle": 0,
                "enabled": 2, "draining": 0, "disabled": 0,
                "stuck": 2,
            },
        }
        with patch("httpx.get", return_value=resp):
            c = check_gateway(config)

        joined_warnings = " | ".join(c.warnings)
        joined_suggestions = " | ".join(c.suggestions)
        assert "2 worker(s) stuck" in joined_warnings
        assert "wedged on a request" in joined_warnings
        # Operator guidance: drain the stuck worker so doctor is
        # actionable, not just informative.
        assert "drain" in joined_suggestions.lower()

    def test_no_stuck_workers_emits_no_warning(self, monkeypatch):
        """Silent when workers.stuck == 0 — otherwise doctor's "WARN"
        signal loses meaning."""
        from unittest.mock import MagicMock, patch

        from towel.cli.doctor import check_gateway

        config = TowelConfig()

        resp = MagicMock()
        resp.json.return_value = {
            "status": "hoopy",
            "connections": 1,
            "sessions": 3,
            "workers": {
                "total": 2, "busy": 0, "idle": 2,
                "enabled": 2, "draining": 0, "disabled": 0,
                "stuck": 0,
            },
        }
        with patch("httpx.get", return_value=resp):
            c = check_gateway(config)

        joined_warnings = " | ".join(c.warnings)
        assert "stuck" not in joined_warnings


class TestStorageCheck:
    def test_storage_check_runs(self):
        c = check_storage()
        c.finalize()
        assert c.passed


class TestPersistedWorkerStateCheck:
    def test_clean_slate_when_file_absent(self, tmp_path, monkeypatch):
        # Point the default store at an empty tmp dir.
        from towel.persistence import worker_state

        monkeypatch.setattr(
            worker_state, "DEFAULT_WORKER_STATE_PATH", tmp_path / "absent.json"
        )
        c = check_persisted_worker_state()
        c.finalize()
        assert c.passed
        assert any("clean slate" in d for d in c.details)

    def test_reports_disabled_draining_and_overrides(self, tmp_path, monkeypatch):
        import json

        from towel.persistence import worker_state

        state_path = tmp_path / "worker_state.json"
        state_path.write_text(
            json.dumps(
                {
                    "gpu-host": {
                        "enabled": False,
                        "draining": False,
                        "tasks": ["code_review"],
                    },
                    "pi-host": {
                        "enabled": True,
                        "draining": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            worker_state, "DEFAULT_WORKER_STATE_PATH", state_path
        )
        c = check_persisted_worker_state()
        c.finalize()
        assert c.passed
        # Disabled and draining warnings surfaced.
        assert any("gpu-host" in w for w in c.warnings)
        assert any("pi-host" in w for w in c.warnings)
        # Manual override is in details (it's informational, not a warning).
        assert any("code_review" in d for d in c.details)


class TestMemoryStoreCheck:
    def test_no_memory_file_is_ok(self, tmp_path, monkeypatch):
        from towel.memory import store as memory_store

        monkeypatch.setattr(memory_store, "DEFAULT_MEMORY_DIR", tmp_path / "memory")
        c = check_memory_store()
        c.finalize()
        assert c.passed
        # Either "no memories yet" or the entry count line — both are
        # the OK state.
        joined = " ".join(c.details)
        assert "No memories stored yet" in joined or "Memory store:" in joined

    def test_migration_archive_is_surfaced(self, tmp_path, monkeypatch):
        from towel.memory import store as memory_store

        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        # Simulate a previous open that migrated from JSON.
        (mem_dir / "memories.json.migrated-20260101T120000").write_text(
            "old json payload", encoding="utf-8"
        )
        monkeypatch.setattr(memory_store, "DEFAULT_MEMORY_DIR", mem_dir)
        c = check_memory_store()
        c.finalize()
        # Informational OK line so operators see what happened.
        assert any("Migrated from JSON store" in d for d in c.details)
        assert c.passed


class TestRunDoctor:
    def test_run_all_checks(self):
        config = TowelConfig()
        config.gateway.port = 19997  # avoid conflicts
        checks = run_doctor(config)
        assert len(checks) == 12
        names = [c.name for c in checks]
        assert "Environment" in names
        assert "Configuration" in names
        assert "GPU" in names
        assert "MLX" in names
        assert "Model" in names
        assert "Skills" in names
        assert "Gateway" in names
        assert "Storage" in names
        assert "Persisted worker state" in names
        assert "SQLite FTS5" in names
        assert "Memory embeddings" in names
        assert "Memory store" in names

    def test_all_checks_finalized(self):
        checks = run_doctor(TowelConfig())
        for c in checks:
            # Each check should have been finalized (passed is set)
            assert isinstance(c.passed, bool)

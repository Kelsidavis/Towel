"""Tests for model management."""

from pathlib import Path

from towel.cli.models import (
    RECOMMENDED_MODELS,
    CachedModel,
    get_model_usage,
    is_model_cached,
    list_cached_models,
)
from towel.config import TowelConfig


class TestCachedModel:
    def test_size_display_gb(self):
        m = CachedModel(name="test", path=Path("/tmp"), size_bytes=5 * 1024**3)
        assert "5.0 GB" == m.size_display

    def test_size_display_mb(self):
        m = CachedModel(name="test", path=Path("/tmp"), size_bytes=500 * 1024**2)
        assert "500 MB" == m.size_display


class TestListCachedModels:
    def test_empty_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.models.get_hf_cache_dir", lambda: tmp_path)
        assert list_cached_models() == []

    def test_finds_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.models.get_hf_cache_dir", lambda: tmp_path)
        (tmp_path / "models--org--model-name").mkdir()
        (tmp_path / "models--org--model-name" / "weights.safetensors").write_bytes(b"x" * 1000)

        cached = list_cached_models()
        assert len(cached) == 1
        assert cached[0].name == "org/model-name"
        assert cached[0].size_bytes == 1000

    def test_skips_non_model_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.models.get_hf_cache_dir", lambda: tmp_path)
        (tmp_path / ".locks").mkdir()
        (tmp_path / "version.txt").touch()

        assert list_cached_models() == []

    def test_nonexistent_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.models.get_hf_cache_dir", lambda: tmp_path / "nope")
        assert list_cached_models() == []


class TestIsModelCached:
    def test_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.models.get_hf_cache_dir", lambda: tmp_path)
        (tmp_path / "models--org--mymodel").mkdir()
        assert is_model_cached("org/mymodel")

    def test_not_cached(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.cli.models.get_hf_cache_dir", lambda: tmp_path)
        assert not is_model_cached("org/missing")


class TestGetModelUsage:
    def test_default_model(self):
        config = TowelConfig()
        usage = get_model_usage(config)
        assert "default" in usage[config.model.name]

    def test_agent_models(self):
        config = TowelConfig()
        usage = get_model_usage(config)
        # Built-in agents should appear
        all_agents = []
        for agents in usage.values():
            all_agents.extend(agents)
        assert "coder" in all_agents


class TestRecommendedModels:
    def test_has_entries(self):
        assert len(RECOMMENDED_MODELS) > 0

    def test_all_have_required_fields(self):
        for m in RECOMMENDED_MODELS:
            assert "name" in m
            assert "params" in m
            assert "ram" in m
            assert "use" in m

    def test_includes_variety(self):
        params = [m["params"] for m in RECOMMENDED_MODELS]
        assert "70B" in params
        assert "7B" in params or "8B" in params


class TestModelsCLI:
    def test_list_command_exists(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "list"])
        assert result.exit_code == 0

    def test_recommended_command(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "recommended"])
        assert result.exit_code == 0
        assert "Recommended" in result.output

    def test_active_command(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "active"])
        assert result.exit_code == 0
        assert "default" in result.output

    def test_models_help(self):
        from click.testing import CliRunner

        from towel.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["models", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "pull" in result.output
        assert "recommended" in result.output

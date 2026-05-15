"""Tests for configuration management."""

from towel.config import GatewayConfig, ModelConfig, TowelConfig


def test_default_config():
    config = TowelConfig()
    assert config.gateway.port == 18742
    assert config.model.max_tokens == 4096
    assert "Don't Panic" in config.identity


def test_model_config():
    mc = ModelConfig(name="mlx-community/test-model", temperature=0.5)
    assert mc.name == "mlx-community/test-model"
    assert mc.temperature == 0.5


def test_gateway_config():
    gc = GatewayConfig(port=9999)
    assert gc.host == "127.0.0.1"
    assert gc.port == 9999


def test_save_load_roundtrip(tmp_path):
    """Saved config must load back identically — atomic-write
    pattern still produces a valid TOML the loader accepts."""
    path = tmp_path / "config.toml"
    original = TowelConfig(identity="custom identity here")
    original.save(path)

    reloaded = TowelConfig.load(path)
    assert reloaded.identity == "custom identity here"


def test_load_corrupt_config_backs_up_and_returns_defaults(tmp_path):
    """A corrupt config.toml previously made TowelConfig.load() raise
    on startup — the coordinator crashed before reaching the
    operator. Now load() backs up the bad file aside and falls back
    to defaults; doctor / the next save will then write a fresh
    config without clobbering the corrupted bytes."""
    path = tmp_path / "config.toml"
    path.write_text("[invalid toml syntax")

    config = TowelConfig.load(path)
    # Defaults returned.
    assert config.gateway.port == 18742

    # Original file renamed aside.
    assert not path.exists()
    backups = list(tmp_path.glob("config.toml.corrupted-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "[invalid toml syntax"


def test_load_invalid_schema_backs_up_and_returns_defaults(tmp_path):
    """Valid TOML that doesn't match the schema (e.g. a stale config
    from an old version with renamed fields) also tripped model
    validation — same data-loss path. Treat it as corruption."""
    path = tmp_path / "config.toml"
    # Valid TOML but wrong shape — gateway should be a table, not int.
    path.write_text("gateway = 42\n")

    config = TowelConfig.load(path)
    assert config.gateway.port == 18742

    assert not path.exists()
    assert len(list(tmp_path.glob("config.toml.corrupted-*"))) == 1


def test_save_is_atomic(tmp_path, monkeypatch):
    """A crash mid-write must not corrupt the existing config so the
    operator's tuned values survive. Same pattern as the JSON
    persistence stores."""
    path = tmp_path / "config.toml"
    TowelConfig(identity="original").save(path)
    assert TowelConfig.load(path).identity == "original"

    from pathlib import Path
    original_replace = Path.replace

    def failing_replace(self, target):
        raise OSError("simulated disk-full at rename time")

    monkeypatch.setattr(Path, "replace", failing_replace)
    try:
        TowelConfig(identity="new value").save(path)
    except OSError:
        pass
    finally:
        monkeypatch.setattr(Path, "replace", original_replace)

    # Original file survives intact.
    assert TowelConfig.load(path).identity == "original"

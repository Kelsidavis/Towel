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

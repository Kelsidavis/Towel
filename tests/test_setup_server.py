"""Tests for the setup-wizard backend."""

from __future__ import annotations

from pathlib import Path

from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from towel.config import TowelConfig
from towel.setup_server import (
    _apply_form_to_config,
    _list_mlx_cached_models,
    build_app,
    setup_routes,
)


class TestApplyFormToConfig:
    def test_valid_form_persists_all_fields(self):
        cfg = TowelConfig()
        ok, err = _apply_form_to_config(
            cfg,
            {
                "backend": "ollama",
                "identity": "You are Towel testing.",
                "ollama_url": "http://192.168.1.10:11434",
                "llama_url": "",
                "claude_model": "",
                "model_name": "qwen3.6:27b",
            },
        )
        assert ok and err is None
        assert cfg.backend == "ollama"
        assert cfg.identity == "You are Towel testing."
        assert cfg.ollama_url == "http://192.168.1.10:11434"
        assert cfg.model.name == "qwen3.6:27b"

    def test_unknown_backend_rejected(self):
        cfg = TowelConfig()
        ok, err = _apply_form_to_config(cfg, {"backend": "bogus"})
        assert not ok
        assert "Unknown backend" in (err or "")

    def test_empty_backend_means_auto_detect(self):
        # An empty backend is valid — it means "let the CLI auto-detect".
        cfg = TowelConfig()
        ok, _err = _apply_form_to_config(cfg, {"backend": "", "identity": "x"})
        assert ok
        assert cfg.backend == ""

    def test_llama_and_claude_dont_overwrite_model_name(self):
        # Both backends drive their own model selection through other fields,
        # so a stray model_name in the form must not clobber config.model.name.
        cfg = TowelConfig()
        original = cfg.model.name
        _apply_form_to_config(
            cfg,
            {"backend": "llama", "model_name": "should-be-ignored"},
        )
        assert cfg.model.name == original
        _apply_form_to_config(
            cfg,
            {"backend": "claude", "model_name": "also-ignored", "claude_model": "opus"},
        )
        assert cfg.model.name == original
        assert cfg.claude_model == "opus"

    def test_empty_identity_keeps_existing(self):
        cfg = TowelConfig(identity="existing identity")
        _apply_form_to_config(cfg, {"backend": "mlx", "identity": ""})
        assert cfg.identity == "existing identity"


class TestListMlxCachedModels:
    def test_returns_a_list_without_crashing(self):
        # The function reads $HF_HOME and ~/.cache/huggingface/hub. Either
        # may not exist on every test machine; either way the return is a
        # list (possibly empty).
        result = _list_mlx_cached_models()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)


class TestSetupRoutes:
    def test_setup_routes_registers_expected_paths(self):
        routes = setup_routes()
        paths = [r.path if hasattr(r, "path") else None for r in routes]
        assert "/setup" in paths
        assert "/api/setup/state" in paths
        assert "/api/setup/save" in paths
        assert any(isinstance(r, Route) and "/api/setup/backends/" in r.path for r in routes)
        # Static mount should be present when the web dir exists.
        web_dir = Path(__file__).resolve().parent.parent / "src" / "towel" / "web"
        if web_dir.is_dir():
            assert any(isinstance(r, Mount) for r in routes)


class TestStandaloneApp:
    def test_state_endpoint_returns_config_and_backends(self):
        with TestClient(build_app()) as client:
            resp = client.get("/api/setup/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "backends" in data
        assert "config_path" in data
        # All four backends must show up with an availability decision.
        for name in ("mlx", "ollama", "llama", "claude"):
            entry = data["backends"][name]
            assert "available" in entry
            assert isinstance(entry["available"], bool)
            assert "reason" in entry

    def test_save_rejects_invalid_json(self):
        with TestClient(build_app()) as client:
            resp = client.post(
                "/api/setup/save",
                content="not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 400
        assert "Invalid JSON" in resp.json().get("error", "")

    def test_save_rejects_unknown_backend(self):
        with TestClient(build_app()) as client:
            resp = client.post(
                "/api/setup/save",
                json={"backend": "bogus", "identity": "x"},
            )
        assert resp.status_code == 400

    def test_mlx_models_endpoint_returns_list(self):
        with TestClient(build_app()) as client:
            resp = client.get("/api/setup/backends/mlx/models")
        assert resp.status_code == 200
        assert isinstance(resp.json().get("models"), list)

    def test_setup_page_served_at_root(self):
        with TestClient(build_app()) as client:
            resp = client.get("/")
        assert resp.status_code == 200
        # The HTML should contain the wizard title and form sections.
        assert "TOWEL SETUP" in resp.text or "Towel Setup" in resp.text

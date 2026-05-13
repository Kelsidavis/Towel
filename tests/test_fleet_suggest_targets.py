"""Tests for /fleet/suggest-targets and the model-name → param-count guesser."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from towel.config import TowelConfig
from towel.gateway.server import GatewayServer, _guess_model_param_b
from towel.gateway.sessions import SessionManager
from towel.persistence.session_pins import SessionPinStore
from towel.persistence.store import ConversationStore
from towel.persistence.worker_state import WorkerStateStore


class _FakeAgent:
    pass


@pytest.fixture
def store(tmp_path):
    return ConversationStore(store_dir=tmp_path)


@pytest.fixture
def gateway(store):
    sessions = SessionManager(store=store)
    pin_store = SessionPinStore(path=store.store_dir / "session_pins.json")
    worker_state_store = WorkerStateStore(path=store.store_dir / "worker_state.json")
    return GatewayServer(
        config=TowelConfig(),
        agent=_FakeAgent(),
        sessions=sessions,
        pin_store=pin_store,
        worker_state_store=worker_state_store,
    )


def _register(gateway: GatewayServer, worker_id: str, **caps) -> None:
    gateway._workers.register(worker_id, ws=MagicMock(), capabilities=caps)


class TestGuessModelParamB:
    def test_recognises_common_param_count_patterns(self):
        assert _guess_model_param_b("Llama-3.3-70B-Instruct-4bit") == 70.0
        assert _guess_model_param_b("qwen3.6:27b") == 27.0
        assert _guess_model_param_b("Phi-3.5-mini-3.8B") == 3.8
        assert _guess_model_param_b("mlx-community/Qwen2.5-Coder-32B") == 32.0

    def test_ignores_quantisation_tags(self):
        # "4bit" and "8bit" must not be read as param counts.
        assert _guess_model_param_b("mlx-community/Llama-3.3-70B-Instruct-4bit") == 70.0
        # A model name with ONLY a quant tag and no param count returns None.
        assert _guess_model_param_b("some-model-8bit") is None

    def test_unrecognised_name_returns_none(self):
        assert _guess_model_param_b("") is None
        assert _guess_model_param_b("opus") is None
        assert _guess_model_param_b("claude-sonnet") is None


class TestSuggestTargets:
    def test_classifies_workers_by_cached_and_fit(self, gateway):
        _register(
            gateway, "beefy-cached",
            backend="mlx",
            max_param_b_est=80.0,
            available_models=["mlx-community/Llama-3.3-70B-Instruct-4bit"],
            total_vram_mb=24000, context_window=131072,
        )
        _register(
            gateway, "beefy-not-cached",
            backend="mlx",
            max_param_b_est=80.0,
            available_models=["other-model"],
            total_vram_mb=24000, context_window=131072,
        )
        _register(
            gateway, "pi-too-small",
            backend="ollama",
            max_param_b_est=3.0,
            available_models=[],
        )

        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/suggest-targets",
            json={"model": "mlx-community/Llama-3.3-70B-Instruct-4bit"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "mlx-community/Llama-3.3-70B-Instruct-4bit"
        assert data["estimated_param_b"] == 70.0

        analyses = {a["worker_id"]: a for a in data["workers"]}
        assert analyses["beefy-cached"]["has_model_cached"] is True
        assert analyses["beefy-cached"]["fits"] is True
        assert analyses["beefy-not-cached"]["has_model_cached"] is False
        assert analyses["beefy-not-cached"]["fits"] is True
        assert analyses["pi-too-small"]["fits"] is False

        # Recommended = cached AND fits. Only beefy-cached qualifies.
        assert data["recommended"] == ["beefy-cached"]

    def test_orders_cached_and_high_tier_first(self, gateway):
        _register(
            gateway, "low-cached",
            backend="ollama",
            max_param_b_est=10.0,
            available_models=["model:x"],
            context_window=8192,  # low tier
        )
        _register(
            gateway, "high-cached",
            backend="mlx",
            max_param_b_est=80.0,
            available_models=["model:x"],
            total_vram_mb=24000, context_window=131072,  # high tier
        )
        _register(
            gateway, "high-not-cached",
            backend="mlx",
            max_param_b_est=80.0,
            available_models=[],
            total_vram_mb=24000, context_window=131072,
        )

        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/suggest-targets", json={"model": "model:x"}
        )
        ids = [w["worker_id"] for w in resp.json()["workers"]]
        # Cached-and-fits high tier first, then cached-and-fits low tier,
        # then the non-cached high tier.
        assert ids == ["high-cached", "low-cached", "high-not-cached"]

    def test_min_param_b_filter_excludes_smaller_workers(self, gateway):
        _register(gateway, "big", max_param_b_est=70.0, available_models=["m"])
        _register(gateway, "small", max_param_b_est=3.0, available_models=["m"])
        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/suggest-targets",
            json={"model": "m", "min_param_b": 8.0},
        )
        ids = [w["worker_id"] for w in resp.json()["workers"]]
        assert ids == ["big"]

    def test_unknown_size_means_every_worker_fits(self, gateway):
        # "haiku" doesn't parse to a number — the endpoint treats it as
        # "size unknown, don't pre-reject anyone."
        _register(gateway, "pi", max_param_b_est=2.0, available_models=["haiku"])
        client = TestClient(gateway._build_http_app())
        resp = client.post("/fleet/suggest-targets", json={"model": "haiku"})
        data = resp.json()
        assert data["estimated_param_b"] is None
        assert data["workers"][0]["fits"] is True

    def test_missing_model_returns_400(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post("/fleet/suggest-targets", json={})
        assert resp.status_code == 400

    def test_no_workers_means_empty_lists(self, gateway):
        client = TestClient(gateway._build_http_app())
        resp = client.post("/fleet/suggest-targets", json={"model": "any"})
        data = resp.json()
        assert data["workers"] == []
        assert data["recommended"] == []

    def test_download_size_estimate_and_disk_fit(self, gateway):
        """A 70B model needs ~42 GB download. A worker with 100 GB free can
        fit it; a worker with 30 GB can't. Cached workers don't pay the
        cost so their disk_fits is left as None."""
        _register(
            gateway, "spacious",
            backend="ollama",
            max_param_b_est=80.0,
            available_models=[],
            live_resources={"disk_free_gb": 100.0},
        )
        _register(
            gateway, "cramped",
            backend="ollama",
            max_param_b_est=80.0,
            available_models=[],
            live_resources={"disk_free_gb": 30.0},
        )
        _register(
            gateway, "cached-but-tiny-disk",
            backend="ollama",
            max_param_b_est=80.0,
            available_models=["mlx-community/Llama-3.3-70B-Instruct-4bit"],
            live_resources={"disk_free_gb": 1.0},  # full disk doesn't matter
        )

        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/suggest-targets",
            json={"model": "mlx-community/Llama-3.3-70B-Instruct-4bit"},
        )
        data = resp.json()
        assert data["estimated_param_b"] == 70.0
        # 70 * 0.6 = 42 GB
        assert data["estimated_download_gb"] == 42.0
        per_worker = {w["worker_id"]: w for w in data["workers"]}
        assert per_worker["spacious"]["disk_fits"] is True
        assert per_worker["cramped"]["disk_fits"] is False
        # Cached worker: disk_fits is None — can't fail what won't happen.
        assert per_worker["cached-but-tiny-disk"]["disk_fits"] is None

    def test_disk_fits_is_none_when_size_is_unknown(self, gateway):
        # "haiku" can't be parsed to a size, so we have no download estimate
        # and therefore can't say whether disk fits.
        _register(
            gateway, "any",
            backend="claude",
            max_param_b_est=50.0,
            available_models=["sonnet"],
            live_resources={"disk_free_gb": 100.0},
        )
        client = TestClient(gateway._build_http_app())
        resp = client.post("/fleet/suggest-targets", json={"model": "haiku"})
        data = resp.json()
        assert data["estimated_download_gb"] is None
        assert data["workers"][0]["disk_fits"] is None

    def test_disk_fits_is_none_when_worker_does_not_report_disk(self, gateway):
        # Older workers won't have disk_free_gb in live_resources — must
        # treat as "unknown" rather than "definitely won't fit".
        _register(
            gateway, "old-worker",
            backend="ollama",
            max_param_b_est=80.0,
            available_models=[],
            live_resources={},  # no disk telemetry
        )
        client = TestClient(gateway._build_http_app())
        resp = client.post(
            "/fleet/suggest-targets", json={"model": "Llama-3.3-70B"}
        )
        data = resp.json()
        assert data["workers"][0]["disk_fits"] is None

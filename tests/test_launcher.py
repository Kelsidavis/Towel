"""Tests for the Towel launcher daemon.

The launcher exposes ``POST /launch`` which spawns ``towel worker`` as a
subprocess. We stub :func:`towel.launcher._spawn_worker` so the tests can
exercise the routing/auth/validation logic without actually starting new
processes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from towel import launcher
from towel.launcher import _build_worker_argv, build_app


class TestArgvBuilder:
    def test_minimum_required_is_controller(self):
        argv, err = _build_worker_argv({})
        assert err is not None
        assert "controller" in err

    def test_accepts_master_as_alias_for_controller(self):
        argv, err = _build_worker_argv({"master": "ws://10.0.0.1:18742"})
        assert err is None
        assert "--master" in argv
        assert "ws://10.0.0.1:18742" in argv

    def test_rejects_unknown_backend(self):
        argv, err = _build_worker_argv(
            {"controller": "ws://x/y", "backend": "bogus"}
        )
        assert err is not None
        assert "backend" in err

    def test_full_payload_produces_expected_flags(self):
        argv, err = _build_worker_argv(
            {
                "controller": "ws://controller:18742",
                "backend": "ollama",
                "ollama_url": "http://gpu-box:11434",
                "worker_id": "w-gpu-1",
                "allow_tools": False,
            }
        )
        assert err is None
        assert argv[1] == "worker" or argv[2] == "worker"  # towel worker / python -m towel worker
        assert "--master" in argv
        assert "ws://controller:18742" in argv
        assert "--backend" in argv
        assert "ollama" in argv
        assert "--ollama-url" in argv
        assert "http://gpu-box:11434" in argv
        assert "--worker-id" in argv
        assert "w-gpu-1" in argv
        assert "--no-tools" in argv

    def test_empty_optionals_are_omitted(self):
        argv, _ = _build_worker_argv({"controller": "ws://x"})
        # No --backend, --ollama-url, --worker-id, --no-tools when not specified.
        assert "--backend" not in argv
        assert "--ollama-url" not in argv
        assert "--worker-id" not in argv
        assert "--no-tools" not in argv


class TestAuth:
    TOKEN = "test-secret-token-do-not-leak"

    def test_health_does_not_require_auth(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "towel-launcher"

    def test_missing_auth_header_rejected(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post("/launch", json={"controller": "ws://x"})
        assert resp.status_code == 401
        assert "missing" in resp.json()["error"].lower()

    def test_wrong_token_rejected(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post(
            "/launch",
            json={"controller": "ws://x"},
            headers={"authorization": "Bearer different-token"},
        )
        assert resp.status_code == 401
        assert "invalid" in resp.json()["error"].lower()

    def test_wrong_scheme_rejected(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post(
            "/launch",
            json={"controller": "ws://x"},
            headers={"authorization": f"Basic {self.TOKEN}"},
        )
        assert resp.status_code == 401


class TestLaunch:
    TOKEN = "test-secret"

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.TOKEN}"}

    def _patched_spawn(self):
        """Patch _spawn_worker to return a fake Popen so we don't fork."""
        fake_proc = MagicMock()
        fake_proc.pid = 12345
        return patch.object(launcher, "_spawn_worker", return_value=fake_proc)

    def test_happy_path_returns_pid_and_argv(self):
        client = TestClient(build_app(self.TOKEN))
        with self._patched_spawn():
            resp = client.post(
                "/launch",
                json={"controller": "ws://ctrl:18742", "backend": "mlx"},
                headers=self._headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["pid"] == 12345
        assert "--master" in data["argv"]
        assert "ws://ctrl:18742" in data["argv"]

    def test_missing_controller_returns_400(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post("/launch", json={}, headers=self._headers())
        assert resp.status_code == 400

    def test_invalid_json_returns_400(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post(
            "/launch",
            content="not-json",
            headers={**self._headers(), "content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_non_object_payload_returns_400(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post(
            "/launch",
            json=["controller", "ws://x"],
            headers=self._headers(),
        )
        assert resp.status_code == 400
        assert "object" in resp.json()["error"]

    def test_env_overrides_must_be_dict(self):
        client = TestClient(build_app(self.TOKEN))
        with self._patched_spawn():
            resp = client.post(
                "/launch",
                json={"controller": "ws://x", "env": "not-a-dict"},
                headers=self._headers(),
            )
        assert resp.status_code == 400
        assert "env" in resp.json()["error"]

    def test_spawn_failure_becomes_500(self):
        client = TestClient(build_app(self.TOKEN))
        with patch.object(launcher, "_spawn_worker", side_effect=OSError("nope")):
            resp = client.post(
                "/launch",
                json={"controller": "ws://x"},
                headers=self._headers(),
            )
        assert resp.status_code == 500
        assert "nope" in resp.json()["error"]

    def test_missing_binary_returns_500(self):
        client = TestClient(build_app(self.TOKEN))
        with patch.object(
            launcher, "_spawn_worker", side_effect=FileNotFoundError("towel")
        ):
            resp = client.post(
                "/launch",
                json={"controller": "ws://x"},
                headers=self._headers(),
            )
        assert resp.status_code == 500
        assert "towel" in resp.json()["error"]


class TestRunStartupValidation:
    def test_refuses_to_start_without_token_env(self, monkeypatch):
        # Unset the env var so the launcher's fail-secure check fires.
        monkeypatch.delenv(launcher.TOKEN_ENV, raising=False)
        try:
            launcher.run(host="127.0.0.1", port=0)
        except RuntimeError as exc:
            assert launcher.TOKEN_ENV in str(exc)
            return
        raise AssertionError("run() should have raised when token env was unset")

"""Tests for the Towel launcher daemon.

The launcher exposes ``POST /launch`` which spawns ``towel worker`` as a
subprocess. We stub :func:`towel.launcher._spawn_worker` so the tests can
exercise the routing/auth/validation logic without actually starting new
processes.
"""

from __future__ import annotations

from pathlib import Path
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

    def test_rejects_non_string_controller(self):
        """A non-string controller (list / dict / number) used to be
        Python-repr'd into argv, producing a worker that connected to
        a bogus URL like "['ws://x']" and silently failed. Reject at
        the boundary."""
        for bad in ([1, 2], {"x": 1}, 42, True):
            argv, err = _build_worker_argv({"controller": bad})
            assert err is not None, f"accepted {bad!r}"
            assert "string" in err.lower(), f"unexpected error for {bad!r}: {err}"

    def test_rejects_non_ws_scheme(self):
        """A typo like 'http://controller:18742' (HTTP coordinator URL,
        not the WS endpoint) used to spawn a worker that hung on
        websockets.connect — opaque to the operator. Fail loud."""
        for bad in (
            "http://x:18742",
            "https://x:18742",
            "controller:18742",
            "192.168.1.1:18742",
        ):
            argv, err = _build_worker_argv({"controller": bad})
            assert err is not None, f"accepted {bad!r}"
            assert "ws://" in err

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
        assert "--no-allow-tools" in argv

    def test_empty_optionals_are_omitted(self):
        argv, _ = _build_worker_argv({"controller": "ws://x"})
        # No --backend, --ollama-url, --worker-id, --allow-tools flags when
        # not specified — the worker CLI's default applies.
        assert "--backend" not in argv
        assert "--ollama-url" not in argv
        assert "--worker-id" not in argv
        assert "--no-allow-tools" not in argv
        assert "--allow-tools" not in argv
        # Most importantly: no --model when not specified, so the worker
        # uses its config.toml default rather than being forced onto a
        # stale value.
        assert "--model" not in argv

    def test_model_field_forwarded_as_dash_dash_model(self):
        """The coordinator distributes different models by passing the
        ``model`` field in the launch payload — the launcher must forward
        it as ``--model`` to ``towel worker``."""
        argv, err = _build_worker_argv(
            {
                "controller": "ws://x",
                "backend": "ollama",
                "model": "qwen3.6:35b-a3b",
            }
        )
        assert err is None
        assert "--model" in argv
        idx = argv.index("--model")
        assert argv[idx + 1] == "qwen3.6:35b-a3b"

    def test_mlx_huggingface_model_id_passes_through(self):
        # MLX models are HF identifiers like "mlx-community/Foo-4bit". The
        # launcher must not mangle or reject them.
        argv, _ = _build_worker_argv(
            {
                "controller": "ws://x",
                "model": "mlx-community/Llama-3.3-70B-Instruct-4bit",
            }
        )
        idx = argv.index("--model")
        assert argv[idx + 1] == "mlx-community/Llama-3.3-70B-Instruct-4bit"


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

    def _patched_spawn(self, exit_code=None, log_tail="ok"):
        """Patch _spawn_worker to return (fake_proc, fake_log_path).

        ``exit_code`` controls what ``proc.poll()`` returns after the
        boot-grace sleep: ``None`` means "still running, looks healthy",
        an int means "crashed at boot, here's the exit code".
        """
        fake_proc = MagicMock()
        fake_proc.pid = 12345
        fake_proc.poll.return_value = exit_code
        # Use a non-existent path so _tail_bytes returns "". Tests that
        # care about log_tail patch _tail_bytes separately.
        fake_log_path = Path("/tmp/towel-test-no-such-log")

        spawn_patch = patch.object(
            launcher, "_spawn_worker", return_value=(fake_proc, fake_log_path)
        )
        # Skip the real boot-grace sleep so the suite stays fast.
        sleep_patch = patch.object(launcher.time, "sleep", new=lambda _s: None)

        class _BothPatches:
            def __enter__(self):
                spawn_patch.__enter__()
                sleep_patch.__enter__()
                return fake_proc

            def __exit__(self, *exc):
                sleep_patch.__exit__(*exc)
                spawn_patch.__exit__(*exc)

        return _BothPatches()

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

    def test_worker_crashes_in_boot_grace_returns_500_with_tail(self):
        """A worker that exits inside the boot-grace window should turn
        the optimistic 200 into a 500 with the log tail attached so the
        operator can see *why* — the previous behaviour silently masked
        instant crashes."""
        client = TestClient(build_app(self.TOKEN))
        with self._patched_spawn(exit_code=2), \
             patch.object(launcher, "_tail_bytes", return_value="ImportError: no module foo"):
            resp = client.post(
                "/launch",
                json={"controller": "ws://x", "backend": "mlx"},
                headers=self._headers(),
            )
        assert resp.status_code == 500
        data = resp.json()
        assert data["ok"] is False
        assert data["exit_code"] == 2
        assert "ImportError" in data["log_tail"]
        # Operator-friendly error message points at the log tail.
        assert "log_tail" in data["error"]

    def test_worker_survives_boot_grace_returns_200_with_tail(self):
        """When the worker is still alive after the grace window the
        response is still 200 but it also carries the (possibly empty)
        log tail so the caller can sanity-check startup output."""
        client = TestClient(build_app(self.TOKEN))
        with self._patched_spawn(exit_code=None), \
             patch.object(launcher, "_tail_bytes", return_value="Listening on..."):
            resp = client.post(
                "/launch",
                json={"controller": "ws://x"},
                headers=self._headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["pid"] == 12345
        assert data["log_tail"] == "Listening on..."


class TestGetLaunchLog:
    TOKEN = "test-secret"

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.TOKEN}"}

    def test_requires_token(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.get("/launches/12345")
        assert resp.status_code == 401

    def test_returns_404_when_no_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launcher, "LAUNCHER_LOG_DIR", tmp_path / "logs")
        client = TestClient(build_app(self.TOKEN))
        resp = client.get("/launches/99999", headers=self._headers())
        assert resp.status_code == 404

    def test_returns_log_when_present(self, tmp_path, monkeypatch):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "worker-12345.log").write_text("the worker said hi", encoding="utf-8")
        monkeypatch.setattr(launcher, "LAUNCHER_LOG_DIR", log_dir)
        client = TestClient(build_app(self.TOKEN))
        resp = client.get("/launches/12345", headers=self._headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["pid"] == 12345
        assert data["log_tail"] == "the worker said hi"

    def test_rejects_non_integer_pid(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.get("/launches/not-a-pid", headers=self._headers())
        assert resp.status_code == 400


class TestUpgrade:
    TOKEN = "test-secret"

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.TOKEN}"}

    def _fake_run(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = stderr
        return result

    def test_pip_strategy_runs_pip_install_upgrade(self):
        client = TestClient(build_app(self.TOKEN))
        with patch("towel.launcher.subprocess.run") as run:
            run.return_value = self._fake_run(stdout="Successfully installed towel-0.42.0")
            resp = client.post(
                "/upgrade",
                json={"strategy": "pip"},
                headers=self._headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["returncode"] == 0
        assert data["strategy"] == "pip"
        assert data["command"] == ["pip", "install", "--upgrade", "towel"]
        assert "towel-0.42.0" in data["stdout"]

    def test_unknown_strategy_returns_400(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post(
            "/upgrade",
            json={"strategy": "yolo"},
            headers=self._headers(),
        )
        assert resp.status_code == 400
        assert "yolo" in resp.json()["error"]

    def test_custom_command_runs_verbatim(self):
        client = TestClient(build_app(self.TOKEN))
        with patch("towel.launcher.subprocess.run") as run:
            run.return_value = self._fake_run(stdout="ok")
            resp = client.post(
                "/upgrade",
                json={"command": ["sh", "-c", "echo hi"]},
                headers=self._headers(),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["command"] == ["sh", "-c", "echo hi"]
        assert data["strategy"] == "custom"

    def test_custom_command_must_be_list_of_strings(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post(
            "/upgrade",
            json={"command": "not-a-list"},
            headers=self._headers(),
        )
        assert resp.status_code == 400

    def test_nonzero_exit_marks_response_as_not_ok_with_500(self):
        client = TestClient(build_app(self.TOKEN))
        with patch("towel.launcher.subprocess.run") as run:
            run.return_value = self._fake_run(
                returncode=1, stderr="pip: command not found"
            )
            resp = client.post(
                "/upgrade",
                json={"strategy": "pip"},
                headers=self._headers(),
            )
        assert resp.status_code == 500
        data = resp.json()
        assert data["ok"] is False
        assert data["returncode"] == 1
        assert "pip" in data["stderr"]

    def test_timeout_returns_504(self):
        client = TestClient(build_app(self.TOKEN))
        with patch("towel.launcher.subprocess.run") as run:
            run.side_effect = launcher.subprocess.TimeoutExpired(cmd=["pip"], timeout=300)
            resp = client.post(
                "/upgrade",
                json={"strategy": "pip"},
                headers=self._headers(),
            )
        assert resp.status_code == 504
        assert "timed out" in resp.json()["error"]

    def test_upgrade_requires_auth(self):
        client = TestClient(build_app(self.TOKEN))
        resp = client.post("/upgrade", json={"strategy": "pip"})
        assert resp.status_code == 401


class TestSelfUpgradeFailureRecord:
    """self_upgrade_and_reexec must stash failure details so the
    next capability heartbeat surfaces them to the operator."""

    def _reset(self):
        from towel import launcher as _l
        _l._last_upgrade_attempt = None

    def test_unknown_strategy_records_failure(self):
        from towel.launcher import get_last_upgrade_attempt, self_upgrade_and_reexec
        self._reset()
        assert self_upgrade_and_reexec("bogus") is False
        attempt = get_last_upgrade_attempt()
        assert attempt is not None
        assert attempt["strategy"] == "bogus"
        assert attempt["status"] == "unknown_strategy"
        assert "ts" in attempt

    def test_command_failure_captures_exit_and_tail(self):
        from towel.launcher import get_last_upgrade_attempt, self_upgrade_and_reexec
        self._reset()
        with patch("towel.launcher.subprocess.run") as run:
            fake = MagicMock()
            fake.returncode = 1
            fake.stderr = "error line\nfinal explanation"
            fake.stdout = ""
            run.return_value = fake
            assert self_upgrade_and_reexec("pip") is False
        attempt = get_last_upgrade_attempt()
        assert attempt is not None
        assert attempt["status"] == "failed_exit"
        assert attempt["returncode"] == 1
        assert "final explanation" in attempt["tail"]

    def test_timeout_records_status(self):
        import subprocess as _sp

        from towel.launcher import get_last_upgrade_attempt, self_upgrade_and_reexec
        self._reset()
        with patch(
            "towel.launcher.subprocess.run",
            side_effect=_sp.TimeoutExpired(cmd="pip", timeout=300),
        ):
            assert self_upgrade_and_reexec("pip") is False
        attempt = get_last_upgrade_attempt()
        assert attempt is not None
        assert attempt["status"] == "timeout"

    def test_command_not_found_records(self):
        from towel.launcher import get_last_upgrade_attempt, self_upgrade_and_reexec
        self._reset()
        with patch(
            "towel.launcher.subprocess.run",
            side_effect=FileNotFoundError("pip"),
        ):
            assert self_upgrade_and_reexec("pip") is False
        attempt = get_last_upgrade_attempt()
        assert attempt is not None
        assert attempt["status"] == "command_not_found"


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

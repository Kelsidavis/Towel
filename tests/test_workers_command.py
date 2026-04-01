"""Tests for the `towel workers` CLI command."""

from click.testing import CliRunner

from towel.cli.main import cli


class _Resp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class TestWorkersCommand:
    def test_workers_human_output(self, monkeypatch):
        payload = {
            "requirements": {
                "backend": "mlx",
                "mode": "mlx_prompt",
                "model": "repo/model-a",
                "tools": False,
            },
            "pins": {"chat-2": "desktop-1"},
            "workers": [
                {
                    "id": "desktop-1",
                    "busy": True,
                    "enabled": True,
                    "draining": False,
                    "current_session_id": "chat-1",
                    "capabilities": {
                        "backend": "mlx",
                        "modes": ["mlx_prompt"],
                        "model": "repo/model-a",
                        "tools": False,
                    },
                }
            ],
        }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: _Resp(payload))

        result = CliRunner().invoke(cli, ["workers"])

        assert result.exit_code == 0
        assert "desktop-1" in result.output
        assert "backend=mlx" in result.output
        assert "session=chat-1" in result.output
        assert "chat-2 -> desktop-1" in result.output
        assert "busy/enabled/ready" in result.output

    def test_workers_json_output(self, monkeypatch):
        payload = {
            "requirements": {"backend": "mlx", "mode": "mlx_prompt", "model": "repo/model-a"},
            "pins": {"chat-9": "desktop-2"},
            "workers": [],
        }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: _Resp(payload))

        result = CliRunner().invoke(cli, ["workers", "--json"])

        assert result.exit_code == 0
        assert '"workers": []' in result.output
        assert '"backend": "mlx"' in result.output
        assert '"chat-9": "desktop-2"' in result.output

    def test_status_shows_pinned_sessions(self, monkeypatch):
        health_payload = {
            "status": "hoopy",
            "version": "0.1.0",
            "motto": "Don't Panic.",
            "connections": 1,
            "sessions": 2,
            "workers": {
                "total": 1,
                "idle": 0,
                "busy": 0,
                "enabled": 1,
                "draining": 1,
                "disabled": 0,
            },
        }
        workers_payload = {
            "requirements": {
                "backend": "mlx",
                "mode": "mlx_prompt",
                "model": "repo/model-a",
                "tools": False,
            },
            "pins": {"chat-7": "desktop-1"},
            "workers": [
                {
                    "id": "desktop-1",
                    "busy": False,
                    "enabled": True,
                    "draining": True,
                    "current_session_id": None,
                    "capabilities": {
                        "backend": "mlx",
                        "modes": ["mlx_prompt"],
                        "model": "repo/model-a",
                        "tools": False,
                    },
                }
            ],
        }

        responses = [_Resp(health_payload), _Resp(workers_payload)]
        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: responses.pop(0))

        result = CliRunner().invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "Pinned Sessions" in result.output
        assert "chat-7 -> desktop-1" in result.output
        assert "draining 1" in result.output

    def test_workers_gateway_not_running(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("down")

        monkeypatch.setattr("httpx.get", _boom)

        result = CliRunner().invoke(cli, ["workers"])

        assert result.exit_code == 1
        assert "Gateway not running" in result.output

    def test_pin_worker_command(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.post",
            lambda *args, **kwargs: _Resp(
                {"session_id": "chat-1", "worker_id": "desktop-1", "pinned": True}
            ),
        )

        result = CliRunner().invoke(cli, ["pin-worker", "chat-1", "desktop-1"])

        assert result.exit_code == 0
        assert "Pinned" in result.output
        assert "desktop-1" in result.output

    def test_unpin_worker_command(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.delete",
            lambda *args, **kwargs: _Resp(
                {"session_id": "chat-1", "pinned": False, "removed": True}
            ),
        )

        result = CliRunner().invoke(cli, ["unpin-worker", "chat-1"])

        assert result.exit_code == 0
        assert "Unpinned" in result.output

    def test_pin_worker_command_surfaces_api_error(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.post",
            lambda *args, **kwargs: _Resp({"error": "Worker not found"}, status_code=404),
        )

        result = CliRunner().invoke(cli, ["pin-worker", "chat-1", "missing"])

        assert result.exit_code == 1
        assert "Worker not found" in result.output

    def test_routes_human_output(self, monkeypatch):
        payload = {
            "sessions": [
                {
                    "id": "chat-1",
                    "channel": "cli",
                    "messages": 4,
                    "worker_id": "desktop-1",
                    "pinned_worker_id": "desktop-2",
                }
            ]
        }

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: _Resp(payload))

        result = CliRunner().invoke(cli, ["routes"])

        assert result.exit_code == 0
        assert "chat-1" in result.output
        assert "current=desktop-1" in result.output
        assert "pinned=desktop-2" in result.output

    def test_routes_json_output(self, monkeypatch):
        payload = {"sessions": [{"id": "chat-3", "worker_id": "desktop-1"}]}

        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: _Resp(payload))

        result = CliRunner().invoke(cli, ["routes", "--json"])

        assert result.exit_code == 0
        assert '"sessions"' in result.output
        assert '"chat-3"' in result.output

    def test_routes_empty_output(self, monkeypatch):
        monkeypatch.setattr("httpx.get", lambda *args, **kwargs: _Resp({"sessions": []}))

        result = CliRunner().invoke(cli, ["routes"])

        assert result.exit_code == 0
        assert "No active sessions" in result.output

    def test_drain_worker_command(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.post",
            lambda *args, **kwargs: _Resp({"id": "desktop-1", "draining": True}),
        )

        result = CliRunner().invoke(cli, ["drain-worker", "desktop-1"])

        assert result.exit_code == 0
        assert "Draining" in result.output

    def test_disable_worker_command(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.post",
            lambda *args, **kwargs: _Resp({"id": "desktop-1", "enabled": False}),
        )

        result = CliRunner().invoke(cli, ["disable-worker", "desktop-1"])

        assert result.exit_code == 0
        assert "Disabled" in result.output

    def test_worker_state_command_surfaces_api_error(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.post",
            lambda *args, **kwargs: _Resp({"error": "Worker not found"}, status_code=404),
        )

        result = CliRunner().invoke(cli, ["drain-worker", "missing"])

        assert result.exit_code == 1
        assert "Worker not found" in result.output

"""Tests for the `towel orchestrate` CLI command.

Mocks the gateway HTTP call so we exercise the spec parser, the
plan-file loader, and the response renderer without needing a live
coordinator. The live-cluster behavior is covered by the server-side
tests in test_gateway_http.py::TestOrchestrateEndpoint.
"""

from __future__ import annotations

import json as json_mod
from unittest.mock import patch

from click.testing import CliRunner

from towel.cli.main import cli


def _mock_response(payload: dict, status_code: int = 200) -> object:
    class _Resp:
        def __init__(self) -> None:
            self.status_code = status_code
            self.text = json_mod.dumps(payload)

        def json(self) -> dict:
            return payload

    return _Resp()


class TestOrchestrateCLI:
    def test_task_spec_basic(self) -> None:
        runner = CliRunner()
        captured: dict = {}

        def fake_post(url: str, json: dict, timeout=None) -> object:  # noqa: ARG001
            captured["url"] = url
            captured["json"] = json
            return _mock_response({
                "goal": json["goal"],
                "success": True,
                "total_elapsed_ms": 12.3,
                "synthesis": "",
                "tasks": [
                    {"role": "coder", "prompt": "write x.py",
                     "depends_on": [], "with_tools": False, "status": "completed",
                     "elapsed_ms": 12.3, "attempts": 1, "result": "ok"},
                ],
            })

        with patch("httpx.post", side_effect=fake_post):
            result = runner.invoke(cli, [
                "orchestrate",
                "--goal", "g",
                "--task", "coder:write x.py",
            ])
        assert result.exit_code == 0, result.output
        assert captured["json"]["goal"] == "g"
        assert captured["json"]["tasks"][0]["role"] == "coder"
        assert captured["json"]["tasks"][0]["prompt"] == "write x.py"
        assert captured["json"]["tasks"][0]["depends_on"] == []
        assert captured["json"]["tasks"][0]["with_tools"] is False

    def test_task_spec_with_tools_suffix(self) -> None:
        runner = CliRunner()
        captured: dict = {}

        def fake_post(url: str, json: dict, timeout=None) -> object:  # noqa: ARG001
            captured["json"] = json
            return _mock_response({
                "goal": "g", "success": True, "total_elapsed_ms": 0,
                "synthesis": "", "tasks": [],
            })

        with patch("httpx.post", side_effect=fake_post):
            result = runner.invoke(cli, [
                "orchestrate",
                "--goal", "g",
                "--task", "coder:write x.py+tools",
            ])
        assert result.exit_code == 0, result.output
        assert captured["json"]["tasks"][0]["with_tools"] is True
        # `+tools` must not leak into the prompt itself.
        assert captured["json"]["tasks"][0]["prompt"] == "write x.py"

    def test_task_spec_with_deps(self) -> None:
        runner = CliRunner()
        captured: dict = {}

        def fake_post(url: str, json: dict, timeout=None) -> object:  # noqa: ARG001
            captured["json"] = json
            return _mock_response({
                "goal": "g", "success": True, "total_elapsed_ms": 0,
                "synthesis": "", "tasks": [],
            })

        with patch("httpx.post", side_effect=fake_post):
            result = runner.invoke(cli, [
                "orchestrate",
                "--goal", "g",
                "--task", "architect:plan it",
                "--task", "coder:write it@0+tools",
                "--task", "reviewer:check it@0,1",
            ])
        assert result.exit_code == 0, result.output
        assert captured["json"]["tasks"][1]["depends_on"] == [0]
        assert captured["json"]["tasks"][2]["depends_on"] == [0, 1]
        assert captured["json"]["tasks"][1]["with_tools"] is True

    def test_task_spec_missing_colon_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [
            "orchestrate",
            "--goal", "g",
            "--task", "no-colon-here",
        ])
        assert result.exit_code != 0
        assert "role:prompt" in result.output

    def test_task_spec_bad_deps_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, [
            "orchestrate",
            "--goal", "g",
            "--task", "coder:x@abc",
        ])
        assert result.exit_code != 0
        assert "deps" in result.output.lower()

    def test_missing_goal_and_no_plan_file_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["orchestrate"])
        assert result.exit_code != 0
        assert "goal" in result.output.lower()

    def test_no_tasks_without_plan_file_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["orchestrate", "--goal", "g"])
        assert result.exit_code != 0
        assert "task" in result.output.lower()

    def test_plan_file_mode(self, tmp_path) -> None:
        runner = CliRunner()
        plan = {
            "goal": "from file",
            "tasks": [
                {"role": "coder", "prompt": "from file", "with_tools": True},
            ],
        }
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json_mod.dumps(plan))

        captured: dict = {}

        def fake_post(url: str, json: dict, timeout=None) -> object:  # noqa: ARG001
            captured["json"] = json
            return _mock_response({
                "goal": "from file", "success": True, "total_elapsed_ms": 0,
                "synthesis": "", "tasks": [],
            })

        with patch("httpx.post", side_effect=fake_post):
            result = runner.invoke(cli, ["orchestrate", str(plan_path)])
        assert result.exit_code == 0, result.output
        assert captured["json"]["goal"] == "from file"
        assert captured["json"]["tasks"][0]["with_tools"] is True

    def test_plan_file_overridden_by_cli_flags(self, tmp_path) -> None:
        runner = CliRunner()
        plan = {
            "goal": "from file",
            "parallel": False,
            "tasks": [{"role": "coder", "prompt": "x"}],
        }
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json_mod.dumps(plan))

        captured: dict = {}

        def fake_post(url: str, json: dict, timeout=None) -> object:  # noqa: ARG001
            captured["json"] = json
            return _mock_response({
                "goal": "g", "success": True, "total_elapsed_ms": 0,
                "synthesis": "", "tasks": [],
            })

        with patch("httpx.post", side_effect=fake_post):
            result = runner.invoke(cli, [
                "orchestrate", str(plan_path),
                "--parallel",
                "--workspace", "/tmp/cli-ws",
                "--max-attempts", "3",
            ])
        assert result.exit_code == 0, result.output
        # CLI flags must win over plan-file values so saved plans can
        # be re-run with different runtime knobs.
        assert captured["json"]["parallel"] is True
        assert captured["json"]["workspace_dir"] == "/tmp/cli-ws"
        assert captured["json"]["max_attempts"] == 3

    def test_gateway_error_propagates_exit_code(self) -> None:
        runner = CliRunner()

        def fake_post(url: str, json: dict, timeout=None) -> object:  # noqa: ARG001
            return _mock_response({"error": "bad request"}, status_code=400)

        with patch("httpx.post", side_effect=fake_post):
            result = runner.invoke(cli, [
                "orchestrate",
                "--goal", "g",
                "--task", "coder:x",
            ])
        assert result.exit_code != 0
        assert "400" in result.output

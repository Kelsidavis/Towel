"""Workers advertise the URL of a co-located ``towel launcher`` daemon so
the coordinator's replace/upgrade UI doesn't need the operator to retype
the URL for every worker. The advertisement is best-effort — capability
key stays absent when no launcher is reachable, rather than carrying a
misleading value.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

from towel.gateway.worker_client import _detect_local_launcher


class TestLauncherDetection:
    def test_no_launcher_returns_none(self):
        # Probe fails (connection refused) — capability stays absent.
        with patch.dict("os.environ", {}, clear=False), \
             patch("httpx.get", side_effect=ConnectionError("refused")):
            # Make sure TOWEL_LAUNCHER_URL isn't set in the live env.
            import os

            old = os.environ.pop("TOWEL_LAUNCHER_URL", None)
            try:
                assert _detect_local_launcher() is None
            finally:
                if old is not None:
                    os.environ["TOWEL_LAUNCHER_URL"] = old

    def test_env_override_trusted_verbatim(self):
        with patch.dict(
            "os.environ",
            {"TOWEL_LAUNCHER_URL": "http://special-host:9999"},
        ):
            # Even with no live launcher, the env var wins — operator knows
            # their topology better than we can guess.
            assert (
                _detect_local_launcher() == "http://special-host:9999"
            )

    def test_successful_probe_returns_hostname_form(self):
        import os

        old = os.environ.pop("TOWEL_LAUNCHER_URL", None)
        try:
            resp = MagicMock()
            resp.status_code = 200
            with patch("httpx.get", return_value=resp):
                url = _detect_local_launcher()
            # Hostname-form (not 127.0.0.1) because the coordinator process
            # that consumes this URL runs on a different host.
            assert url is not None
            assert url.startswith("http://")
            assert "127.0.0.1" not in url
            assert socket.gethostname() in url
            assert url.endswith(":18751")
        finally:
            if old is not None:
                os.environ["TOWEL_LAUNCHER_URL"] = old

    def test_non_200_response_treated_as_absent(self):
        import os

        old = os.environ.pop("TOWEL_LAUNCHER_URL", None)
        try:
            resp = MagicMock()
            resp.status_code = 404
            with patch("httpx.get", return_value=resp):
                assert _detect_local_launcher() is None
        finally:
            if old is not None:
                os.environ["TOWEL_LAUNCHER_URL"] = old

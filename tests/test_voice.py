"""Tests for voice input module."""

import pytest


class TestVoiceDeps:
    def test_check_deps(self):
        from towel.cli.voice import check_voice_deps
        result = check_voice_deps()
        # Either None (installed) or error string
        assert result is None or "not installed" in result


class TestVoiceTranscribe:
    def test_transcribe_nonexistent(self):
        from towel.cli.voice import transcribe_audio
        result = transcribe_audio("/nonexistent/audio.wav")
        assert "failed" in result.lower() or "not installed" in result.lower()


class TestVoiceCLI:
    def test_command_registered(self):
        from towel.cli.main import cli
        assert "voice" in [c.name for c in cli.commands.values()]

    def test_help(self):
        from click.testing import CliRunner
        from towel.cli.main import cli
        runner = CliRunner()
        result = runner.invoke(cli, ["voice", "--help"])
        assert result.exit_code == 0
        assert "voice" in result.output.lower()
        assert "chat" in result.output.lower()

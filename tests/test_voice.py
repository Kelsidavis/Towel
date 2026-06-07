"""Tests for voice input module."""


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


class TestVoiceOutput:
    def test_check_tts_deps_reports_availability(self):
        from towel.cli.voice import check_tts_deps

        result = check_tts_deps()
        assert result is None or "speech output" in result

    def test_speak_text_uses_macos_say(self, monkeypatch):
        from towel.cli import voice as voice_mod

        calls = []
        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda name: "/usr/bin/say")
        monkeypatch.setattr(
            voice_mod.subprocess,
            "run",
            lambda cmd, check: calls.append((cmd, check)),
        )

        err = voice_mod.speak_text(" hello   there ", voice="Samantha", rate=180)

        assert err is None
        assert calls == [(["say", "-v", "Samantha", "-r", "180", "hello there"], True)]

    def test_speak_text_reports_missing_tts(self, monkeypatch):
        from towel.cli import voice as voice_mod

        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Linux")

        err = voice_mod.speak_text("hello")

        assert err is not None
        assert "speech output" in err


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
        assert "speak" in result.output.lower()

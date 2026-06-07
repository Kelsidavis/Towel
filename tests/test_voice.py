"""Tests for voice input module."""


class _FakeProc:
    def __init__(self, *, raise_on_wait=False):
        self.terminated = False
        self._raise = raise_on_wait

    def wait(self):
        if self._raise and not self.terminated:
            raise KeyboardInterrupt

    def terminate(self):
        self.terminated = True


class TestVoiceDeps:
    def test_check_deps(self):
        from towel.cli.voice import check_voice_deps

        result = check_voice_deps()
        assert result is None or "not installed" in result


class TestVoiceTranscribe:
    def test_transcribe_nonexistent(self):
        from towel.cli.voice import transcribe_audio

        result = transcribe_audio("/nonexistent/audio.wav")
        assert "failed" in result.lower() or "not installed" in result.lower()

    def test_transcribe_passes_model(self, monkeypatch):
        import mlx_whisper

        from towel.cli import voice as voice_mod

        seen = {}

        def fake_transcribe(path, *, path_or_hf_repo):
            seen["model"] = path_or_hf_repo
            return {"text": "hello"}

        monkeypatch.setattr(mlx_whisper, "transcribe", fake_transcribe)
        result = voice_mod.transcribe_audio("/fake.wav", model="mlx-community/whisper-large-v3")
        assert seen["model"] == "mlx-community/whisper-large-v3"
        assert result == "hello"


class TestStripForSpeech:
    def test_removes_code_fences(self):
        from towel.cli.voice import strip_for_speech

        assert strip_for_speech("here:\n```python\nx = 1\n```\nok") == "here: code block ok"

    def test_removes_inline_code(self):
        from towel.cli.voice import strip_for_speech

        assert strip_for_speech("run `make test` now") == "run make test now"

    def test_removes_headers(self):
        from towel.cli.voice import strip_for_speech

        assert strip_for_speech("## Step one") == "Step one"

    def test_removes_bold_italic(self):
        from towel.cli.voice import strip_for_speech

        assert strip_for_speech("**bold** and *italic*") == "bold and italic"

    def test_removes_list_markers(self):
        from towel.cli.voice import strip_for_speech

        text = "- first\n- second\n1. third"
        result = strip_for_speech(text)
        assert "first" in result and "second" in result and "third" in result
        assert "-" not in result and "1." not in result

    def test_removes_urls(self):
        from towel.cli.voice import strip_for_speech

        assert strip_for_speech("see https://example.com for details") == "see link for details"

    def test_collapses_whitespace(self):
        from towel.cli.voice import strip_for_speech

        assert strip_for_speech("a\n\n\nb") == "a. b"


class TestVoiceOutput:
    def test_check_tts_deps_reports_availability(self):
        from towel.cli.voice import check_tts_deps

        result = check_tts_deps()
        assert result is None or "speech output" in result

    def test_speak_text_macos_uses_say(self, monkeypatch):
        from towel.cli import voice as voice_mod

        procs = []
        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda _: "/usr/bin/say")
        monkeypatch.setattr(voice_mod.subprocess, "Popen", lambda cmd: (procs.append(cmd), _FakeProc())[1])

        err = voice_mod.speak_text(" hello   there ", voice="Samantha", rate=180)

        assert err is None
        assert procs == [["say", "-v", "Samantha", "-r", "180", "hello there"]]

    def test_speak_text_linux_uses_espeak_ng(self, monkeypatch):
        from towel.cli import voice as voice_mod

        procs = []
        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda b: "/usr/bin/espeak-ng" if b == "espeak-ng" else None)
        monkeypatch.setattr(voice_mod.subprocess, "Popen", lambda cmd: (procs.append(cmd), _FakeProc())[1])

        err = voice_mod.speak_text("hello there", voice="en-us", rate=160)

        assert err is None
        assert procs == [["espeak-ng", "-s", "160", "-v", "en-us", "hello there"]]

    def test_speak_text_linux_falls_back_to_espeak(self, monkeypatch):
        from towel.cli import voice as voice_mod

        procs = []
        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda b: "/usr/bin/espeak" if b == "espeak" else None)
        monkeypatch.setattr(voice_mod.subprocess, "Popen", lambda cmd: (procs.append(cmd), _FakeProc())[1])

        err = voice_mod.speak_text("hello")

        assert err is None
        assert procs[0][0] == "espeak"

    def test_speak_text_ctrl_c_terminates_proc(self, monkeypatch):
        from towel.cli import voice as voice_mod

        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda _: "/usr/bin/say")

        proc = _FakeProc(raise_on_wait=True)
        monkeypatch.setattr(voice_mod.subprocess, "Popen", lambda _cmd: proc)

        err = voice_mod.speak_text("hello")

        assert err is None
        assert proc.terminated

    def test_speak_text_strips_markdown(self, monkeypatch):
        from towel.cli import voice as voice_mod

        spoken = []
        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda _: "/usr/bin/say")
        monkeypatch.setattr(voice_mod.subprocess, "Popen", lambda cmd: (spoken.append(cmd[-1]), _FakeProc())[1])

        voice_mod.speak_text("## Header\n**bold** and `code`")

        assert spoken and "##" not in spoken[0] and "**" not in spoken[0] and "`" not in spoken[0]

    def test_speak_text_reports_missing_tts_on_unsupported_platform(self, monkeypatch):
        from towel.cli import voice as voice_mod

        monkeypatch.setattr(voice_mod.platform, "system", lambda: "SunOS")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda _: None)

        err = voice_mod.speak_text("hello")

        assert err is not None
        assert "speech output" in err

    def test_speak_text_reports_missing_espeak_on_linux(self, monkeypatch):
        from towel.cli import voice as voice_mod

        monkeypatch.setattr(voice_mod.platform, "system", lambda: "Linux")
        monkeypatch.setattr(voice_mod.shutil, "which", lambda _: None)

        err = voice_mod.speak_text("hello")

        assert err is not None
        assert "espeak-ng" in err


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
        assert "chat" in result.output.lower()
        assert "speak" in result.output.lower()
        assert "model" in result.output.lower()

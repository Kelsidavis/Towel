"""Voice input — speak to Towel using your microphone.

Uses mlx-whisper for on-device speech-to-text on Apple Silicon.
No audio leaves your machine.

Usage:
    towel voice              listen and transcribe once
    towel voice --chat       continuous voice chat mode
    towel voice --file a.wav transcribe an audio file
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from rich.console import Console

log = logging.getLogger("towel.cli.voice")
console = Console()

DEFAULT_WHISPER_MODEL = "mlx-community/whisper-small"


def check_voice_deps() -> str | None:
    """Check if voice dependencies are available. Returns error message or None."""
    try:
        import mlx_whisper  # noqa: F401

        return None
    except ImportError:
        return "mlx-whisper not installed. Run: pip install towel-ai[voice]"


def record_audio(duration: float = 10.0, sample_rate: int = 16000) -> bytes | str:
    """Record audio from the default microphone. Returns WAV bytes or error string."""
    try:
        import io
        import wave

        import numpy as np
        import sounddevice as sd

        console.print(f"[green]Listening...[/green] (up to {duration:.0f}s, press Ctrl+C to stop early)")

        # Start non-blocking so KeyboardInterrupt during wait() preserves captured audio.
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        try:
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            console.print("[dim]Stopped.[/dim]")

        # Trim trailing silence
        threshold = np.max(np.abs(audio)) * 0.01
        nonsilent = np.where(np.abs(audio.flatten()) > threshold)[0]
        if len(nonsilent) > 0:
            audio = audio[: nonsilent[-1] + sample_rate]  # keep 1s after last sound

        # Convert to WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    except ImportError:
        return "sounddevice not installed. Run: pip install towel-ai[voice]"
    except Exception as e:
        return f"Recording failed: {e}"


def transcribe_audio(audio_path: str, *, model: str = DEFAULT_WHISPER_MODEL) -> str:
    """Transcribe an audio file using mlx-whisper."""
    try:
        import mlx_whisper

        console.print("[dim]Transcribing...[/dim]")
        start = time.perf_counter()

        result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model)

        elapsed = time.perf_counter() - start
        text = result.get("text", "").strip()

        if not text:
            return "(no speech detected)"

        console.print(f"[dim]({elapsed:.1f}s)[/dim]")
        return text

    except ImportError:
        return "mlx-whisper not installed."
    except Exception as e:
        return f"Transcription failed: {e}"


def transcribe_bytes(wav_bytes: bytes, *, model: str = DEFAULT_WHISPER_MODEL) -> str:
    """Transcribe WAV bytes by writing to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        return transcribe_audio(tmp, model=model)
    finally:
        Path(tmp).unlink(missing_ok=True)


def strip_for_speech(text: str) -> str:
    """Remove markdown formatting that sounds bad when spoken by TTS."""
    # Fenced code blocks → "code block"
    text = re.sub(r"```[^`]*```", "code block", text, flags=re.DOTALL)
    # Inline code → bare text
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold / italic
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_\n]+)_{1,3}", r"\1", text)
    # List markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # URLs
    text = re.sub(r"https?://\S+", "link", text)
    # Consolidate whitespace
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    return " ".join(text.split())


def check_tts_deps() -> str | None:
    """Check if local speech output is available."""
    if platform.system() == "Darwin" and shutil.which("say"):
        return None
    return "speech output requires macOS 'say' on this build"


def speak_text(text: str, *, voice: str | None = None, rate: int | None = None) -> str | None:
    """Speak text with the local OS TTS engine. Ctrl+C skips to next turn."""
    err = check_tts_deps()
    if err:
        return err
    clean = strip_for_speech(text)
    if not clean:
        return None
    cmd = ["say"]
    if voice:
        cmd.extend(["-v", voice])
    if rate:
        cmd.extend(["-r", str(rate)])
    cmd.append(clean)
    try:
        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
    except Exception as exc:
        return f"Speech output failed: {exc}"
    return None

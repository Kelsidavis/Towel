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
import tempfile
import time
from pathlib import Path

from rich.console import Console

log = logging.getLogger("towel.cli.voice")
console = Console()


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

        console.print(f"[green]Listening...[/green] (up to {duration:.0f}s, press Ctrl+C to stop)")

        try:
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocking=True,
            )
        except KeyboardInterrupt:
            sd.stop()
            _frames = sd.get_status().input_underflow
            console.print("[dim]Stopped.[/dim]")
            audio = sd.rec(0, samplerate=sample_rate, channels=1, dtype="int16")

        # Trim silence from end
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


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an audio file using mlx-whisper."""
    try:
        import mlx_whisper

        console.print("[dim]Transcribing...[/dim]")
        start = time.perf_counter()

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo="mlx-community/whisper-small",
        )

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


def transcribe_bytes(wav_bytes: bytes) -> str:
    """Transcribe WAV bytes by writing to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp = f.name
    try:
        return transcribe_audio(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)

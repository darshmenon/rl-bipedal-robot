"""Text-to-speech via Kokoro (ONNX)."""

from __future__ import annotations

import io
import os

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
VOICE = "af_sarah"

_kokoro: Kokoro | None = None


def _load() -> Kokoro:
    global _kokoro
    if _kokoro is None:
        _kokoro = Kokoro(
            os.path.join(MODELS_DIR, "kokoro-v1.0.onnx"),
            os.path.join(MODELS_DIR, "voices-v1.0.bin"),
        )
    return _kokoro


def synthesize(text: str) -> bytes:
    kokoro = _load()
    samples, sample_rate = kokoro.create(text, voice=VOICE, speed=1.0, lang="en-us")
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples), sample_rate, format="WAV")
    return buf.getvalue()

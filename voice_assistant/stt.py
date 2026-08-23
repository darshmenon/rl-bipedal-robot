"""Speech-to-text via faster-whisper, with automatic GPU->CPU fallback."""

from __future__ import annotations

import io

from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def _load_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model
    try:
        _model = WhisperModel("small.en", device="cuda", compute_type="float16")
    except Exception:
        _model = WhisperModel("small.en", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_bytes: bytes) -> str:
    model = _load_model()
    segments, _info = model.transcribe(
        io.BytesIO(audio_bytes),
        language="en",
        vad_filter=True,
        beam_size=1,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()

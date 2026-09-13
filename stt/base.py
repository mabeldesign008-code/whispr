"""
Transcription result type.

One rule: an engine never returns a bare "" to mean "something went wrong".
Empty text with ok=True means the user was genuinely silent. ok=False means
we failed, and `error` says why so the UI can show it and offer a retry.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class WordInfo:
    """A recognised word with its confidence and timing (ms)."""
    text: str
    confidence: float = 1.0
    start: int = 0
    end: int = 0


@dataclass
class TranscriptionResult:
    text: str = ""
    confidence: float = 1.0
    words: List[WordInfo] = field(default_factory=list)
    language: Optional[str] = None

    model: str = ""
    latency_ms: int = 0
    audio_duration_s: float = 0.0

    ok: bool = True
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def low_confidence_words(self, threshold: float = 0.55) -> List[WordInfo]:
        """Words the model was unsure about. Passed to the LLM corrector so it
        knows *where* to look instead of rewriting the whole sentence."""
        return [w for w in self.words if w.confidence < threshold]

    @classmethod
    def failure(cls, error: str, model: str = "") -> "TranscriptionResult":
        return cls(ok=False, error=error, model=model)


def float_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono [-1, 1] to in-memory 16-bit PCM WAV.

    In-memory on purpose: the old code wrote NamedTemporaryFile(delete=False)
    and never cleaned it up, leaking a WAV per dictation into %TEMP%.
    """
    samples = np.asarray(samples, dtype=np.float32).ravel()

    # Clip before scaling. np.int16(x * 32767) wraps on out-of-range input
    # and turns a loud moment into a burst of noise.
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

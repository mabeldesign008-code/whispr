"""
Speech detection and levelling.

Replaces two things the old recorder did that actively hurt accuracy:

1.8 Peak normalisation to 0.9 on every clip (`audio * (0.9 / peak)`).
    If the peak was a keyboard click, the whole utterance was scaled around
    it. A quiet utterance got boosted 20 dB along with its noise floor. ASR
    models are trained on naturally-levelled audio; forcing every clip to
    identical peak amplitude is a domain shift. We now target RMS with a
    capped gain and a limiter, computed over voiced frames only.

1.9 An RMS silence gate at 0.01 (about -40 dBFS). That is *loud* for a
    laptop mic at arm's length and far above whispered speech, so valid
    recordings came back as "No speech detected" and whisper-mode was
    impossible by construction. We now use energy-based VAD with a noise
    floor estimated from the quietest frames, and generous padding so
    plosive onsets and trailing fricatives survive.

Silero VAD (bundled in sherpa-onnx) would be marginally better, but it
would reintroduce a heavyweight dependency we just removed for a
push-to-talk app where the user already tells us when speech starts and
stops. The energy VAD here only has to trim leading/trailing silence, not
find speech in an open stream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000   # 320 samples

# Keep this much audio either side of detected speech. Generous on purpose:
# clipping a /s/ or /f/ costs a word, keeping 250 ms of silence costs nothing.
PAD_SECONDS = 0.25

TARGET_RMS = 0.06          # about -24 dBFS, a natural speaking level
MAX_GAIN = 4.0             # +12 dB ceiling; never amplify a whisper to a shout
LIMIT_PEAK = 0.97

MIN_SPEECH_SECONDS = 0.15
HIGHPASS_HZ = 70           # rumble, HVAC, desk thumps


@dataclass
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_s: float
    speech_detected: bool
    peak: float
    rms: float
    gain_applied: float
    clipped: bool

    @property
    def is_usable(self) -> bool:
        return self.speech_detected and self.duration_s >= MIN_SPEECH_SECONDS


def process(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> ProcessedAudio:
    """Clean a recorded take for the ASR engine.

    Never returns None. If no speech is found we say so in the result and
    let the caller decide -- the old code returned None from three separate
    places, which is how valid audio ended up discarded.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0:
        return ProcessedAudio(audio, sample_rate, 0.0, False, 0.0, 0.0, 1.0, False)

    audio = _remove_dc(audio)
    audio = _highpass(audio, sample_rate)

    speech_start, speech_end, has_speech = _find_speech(audio, sample_rate)

    if has_speech:
        pad = int(PAD_SECONDS * sample_rate)
        lo = max(0, speech_start - pad)
        hi = min(len(audio), speech_end + pad)
        trimmed = audio[lo:hi]
        voiced = audio[speech_start:speech_end]
    else:
        # Hand the whole take to the ASR anyway. Our VAD being wrong is far
        # more likely than a real model finding nothing, and a wrong guess
        # here silently deletes what the user said.
        trimmed = audio
        voiced = audio

    levelled, gain, clipped = _level(trimmed, voiced)

    peak = float(np.abs(levelled).max()) if levelled.size else 0.0
    rms = float(np.sqrt(np.mean(levelled ** 2))) if levelled.size else 0.0

    return ProcessedAudio(
        samples=levelled,
        sample_rate=sample_rate,
        duration_s=len(levelled) / sample_rate,
        speech_detected=has_speech,
        peak=peak,
        rms=rms,
        gain_applied=gain,
        clipped=clipped,
    )


# ── stages ────────────────────────────────────────────────────────────────


def _remove_dc(audio: np.ndarray) -> np.ndarray:
    return audio - float(np.mean(audio))


def _highpass(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Single-pole high-pass at HIGHPASS_HZ.

    Runs through scipy.lfilter (C loop) -- the equivalent Python loop cost
    95 ms on a 15 s clip, which is real latency on the critical path.
    lfilter, not filtfilt: filtfilt's edge transients would corrupt exactly
    the frames we then measure energy on for VAD.
    """
    if len(audio) < 2:
        return audio

    rc = 1.0 / (2 * np.pi * HIGHPASS_HZ)
    dt = 1.0 / sample_rate
    a = rc / (rc + dt)

    try:
        from scipy.signal import lfilter
        # y[n] = a*y[n-1] + a*(x[n] - x[n-1])
        return lfilter([a, -a], [1.0, -a], audio).astype(np.float32)
    except Exception:
        # Filtering is an optimisation, not a correctness requirement.
        return audio


def _frame_energies(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // FRAME_LEN
    if n == 0:
        return np.array([np.sqrt(np.mean(audio ** 2))], dtype=np.float32)
    frames = audio[: n * FRAME_LEN].reshape(n, FRAME_LEN)
    return np.sqrt(np.mean(frames ** 2, axis=1))


def _find_speech(audio: np.ndarray, sample_rate: int) -> Tuple[int, int, bool]:
    """Return (start_sample, end_sample, found)."""
    energies = _frame_energies(audio)
    if len(energies) < 3:
        return 0, len(audio), bool(np.max(np.abs(audio)) > 1e-4)

    # Noise floor from the quietest 20% of frames -- robust to a recording
    # that is speech from the very first frame (where a "first N frames"
    # estimate would calibrate on the speech itself).
    quiet = np.sort(energies)[: max(1, len(energies) // 5)]
    noise = float(np.median(quiet))
    loudest = float(np.max(energies))

    # If nothing stands out above the floor, there is no speech here.
    if loudest < max(noise * 2.0, 2e-4):
        return 0, len(audio), False

    # Threshold sits between floor and peak, log-weighted so it tracks the
    # signal rather than sitting at a fixed dBFS that fails for whispers.
    threshold = max(noise * 2.5, loudest * 0.08, 1.5e-4)

    above = np.where(energies > threshold)[0]
    if len(above) == 0:
        return 0, len(audio), False

    start = int(above[0]) * FRAME_LEN
    end = min((int(above[-1]) + 1) * FRAME_LEN, len(audio))

    if (end - start) < MIN_SPEECH_SECONDS * sample_rate:
        return 0, len(audio), False

    return start, end, True


def _level(audio: np.ndarray, voiced: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """RMS-target levelling with a capped gain and soft limiting."""
    if audio.size == 0:
        return audio, 1.0, False

    ref = voiced if voiced.size else audio
    rms = float(np.sqrt(np.mean(ref ** 2)))
    if rms < 1e-7:
        return audio, 1.0, False

    gain = min(TARGET_RMS / rms, MAX_GAIN)
    out = audio * gain

    # Only intervene if we actually pushed something over the rail.
    peak = float(np.abs(out).max())
    clipped = peak > 1.0
    if peak > LIMIT_PEAK:
        out = out * (LIMIT_PEAK / peak)
        gain *= LIMIT_PEAK / peak

    return out.astype(np.float32), float(gain), clipped

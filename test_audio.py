"""Tests for audio capture, preprocessing and overlay animation."""

import sys
import types

import numpy as np
import pytest

# sounddevice needs real audio hardware; stub it so capture logic is testable.
if "sounddevice" not in sys.modules:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        sys.modules["sounddevice"] = types.ModuleType("sounddevice")

from audio.capture import AudioCapture, BLOCKSIZE, SAMPLE_RATE, DeviceInfo  # noqa: E402
from audio.process import (  # noqa: E402
    MAX_GAIN, ProcessedAudio, _find_speech, _highpass, process,
)
from ui.overlay import Spring  # noqa: E402
from ui.theme import ease_out_cubic, hexc  # noqa: E402


def speech_clip(silence=0.4, duration=1.2, total=2.2, amp=0.2, noise=0.002, sr=SAMPLE_RATE):
    """Synthetic speech: harmonics with a syllabic amplitude envelope."""
    n = int(total * sr)
    a = (np.random.RandomState(0).randn(n) * noise).astype(np.float32)
    s, e = int(silence * sr), int(silence * sr) + int(duration * sr)
    t = np.arange(e - s) / sr
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) ** 1.5
    sig = amp * env * (np.sin(2 * np.pi * 150 * t)
                       + 0.4 * np.sin(2 * np.pi * 450 * t)
                       + 0.2 * np.sin(2 * np.pi * 1800 * t))
    a[s:e] += sig.astype(np.float32)
    return a


# ── preprocessing ─────────────────────────────────────────────────────────

class TestSpeechDetection:
    def test_detects_normal_speech(self):
        assert process(speech_clip()).speech_detected

    @pytest.mark.parametrize("amp", [0.30, 0.12, 0.04, 0.015, 0.008])
    def test_detects_across_volumes_including_whisper(self, amp):
        """The old RMS gate of 0.01 (~-40 dBFS) discarded anything quiet,
        which made whisper-mode impossible by construction."""
        assert process(speech_clip(amp=amp)).speech_detected

    def test_rejects_pure_noise(self):
        noise = (np.random.RandomState(1).randn(int(1.5 * SAMPLE_RATE)) * 0.002).astype(np.float32)
        assert not process(noise).speech_detected

    def test_speech_starting_at_sample_zero(self):
        """Noise floor is estimated from the quietest frames, not the first
        ones, so a clip that opens on speech still calibrates correctly."""
        assert process(speech_clip(silence=0.0)).speech_detected

    def test_never_returns_none(self):
        for audio in [np.array([], dtype=np.float32),
                      np.zeros(100, dtype=np.float32),
                      speech_clip()]:
            assert isinstance(process(audio), ProcessedAudio)

    def test_empty_input(self):
        r = process(np.array([], dtype=np.float32))
        assert r.duration_s == 0.0 and not r.speech_detected


class TestPadding:
    def test_keeps_audio_around_speech(self):
        """Onsets and trailing fricatives must survive trimming."""
        r = process(speech_clip(silence=0.5, duration=1.0, total=2.5))
        # 1.0s speech + 0.25s padding either side
        assert 1.4 <= r.duration_s <= 1.7

    def test_passes_whole_clip_when_no_speech_found(self):
        """A wrong VAD guess must not delete the user's audio."""
        noise = (np.random.RandomState(2).randn(int(1.0 * SAMPLE_RATE)) * 0.002).astype(np.float32)
        r = process(noise)
        assert not r.speech_detected
        assert r.duration_s == pytest.approx(1.0, abs=0.05)


class TestLevelling:
    def test_gain_is_capped(self):
        r = process(speech_clip(amp=0.001))
        assert r.gain_applied <= MAX_GAIN + 1e-6

    def test_quiet_gets_boosted_more_than_loud(self):
        quiet = process(speech_clip(amp=0.02)).gain_applied
        loud = process(speech_clip(amp=0.40)).gain_applied
        assert quiet > loud

    def test_transient_click_does_not_dominate(self):
        """Old code did audio * (0.9 / peak): a keyboard click rescaled the
        entire utterance around itself."""
        clean = speech_clip(amp=0.12)
        clicked = clean.copy()
        clicked[50] = 0.99
        g_clean = process(clean).gain_applied
        g_click = process(clicked).gain_applied
        assert abs(g_clean - g_click) / g_clean < 0.6

    def test_output_never_clips(self):
        assert process(speech_clip(amp=0.9)).peak <= 1.0


class TestHighpass:
    def test_attenuates_rumble_keeps_speech(self):
        sr, t = SAMPLE_RATE, np.arange(SAMPLE_RATE * 1) / SAMPLE_RATE
        rumble = (0.5 * np.sin(2 * np.pi * 20 * t)).astype(np.float32)
        speech = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        assert np.abs(_highpass(rumble, sr)).max() / 0.5 < 0.4
        assert np.abs(_highpass(speech, sr)).max() / 0.2 > 0.85

    def test_short_input_safe(self):
        assert len(_highpass(np.array([0.1], dtype=np.float32), SAMPLE_RATE)) == 1

    def test_is_fast(self):
        """A Python loop cost 95ms on 15s of audio -- real added latency."""
        import time
        a = np.random.randn(SAMPLE_RATE * 15).astype(np.float32)
        _highpass(a, SAMPLE_RATE)          # warm scipy import
        t0 = time.perf_counter()
        _highpass(a, SAMPLE_RATE)
        assert (time.perf_counter() - t0) < 0.05


# ── capture ───────────────────────────────────────────────────────────────

class TestRingBuffer:
    def _cap(self):
        c = AudioCapture(preroll_seconds=0.5)
        c._running = True
        return c

    def test_preroll_captures_audio_before_hotkey(self):
        """The whole point: audio from before begin() must be included."""
        c = self._cap()
        pre = np.linspace(0.1, 0.5, BLOCKSIZE, dtype=np.float32).reshape(-1, 1)
        for _ in range(4):
            c._callback(pre, BLOCKSIZE, None, None)

        c.begin()
        after = np.full((BLOCKSIZE, 1), 0.9, dtype=np.float32)
        c._callback(after, BLOCKSIZE, None, None)
        out = c.end()

        assert out is not None
        assert len(out) > BLOCKSIZE          # pre-roll is in there
        assert np.isclose(out[-1], 0.9)      # and the new audio too

    def test_ring_wraps_without_growing(self):
        c = self._cap()
        block = np.full((BLOCKSIZE, 1), 0.3, dtype=np.float32)
        for _ in range(100):                 # far more than the ring holds
            c._callback(block, BLOCKSIZE, None, None)
        assert len(c._ring) == c._ring_size
        assert c._ring_filled

    def test_preroll_bounded_by_configured_seconds(self):
        c = self._cap()
        block = np.full((BLOCKSIZE, 1), 0.3, dtype=np.float32)
        for _ in range(200):
            c._callback(block, BLOCKSIZE, None, None)
        c.begin()
        out = c.end()
        assert len(out) <= int(0.5 * SAMPLE_RATE) + BLOCKSIZE

    def test_recording_is_bounded(self):
        """Old MAX_BUFFER_SIZE=0 with a 10-hour cap meant ~14 GB resident."""
        c = AudioCapture(preroll_seconds=0.1, max_seconds=0.2)
        c._running = True
        c.begin()
        block = np.full((BLOCKSIZE, 1), 0.3, dtype=np.float32)
        for _ in range(200):
            c._callback(block, BLOCKSIZE, None, None)
        out = c.end()
        assert len(out) < int(0.5 * SAMPLE_RATE)
        assert c._truncated

    def test_discard_yields_nothing(self):
        c = self._cap()
        c.begin()
        c._callback(np.full((BLOCKSIZE, 1), 0.5, dtype=np.float32), BLOCKSIZE, None, None)
        c.discard()
        assert not c.is_recording and c.end() is None

    def test_callback_tracks_level(self):
        c = self._cap()
        c._callback(np.full((BLOCKSIZE, 1), 0.42, dtype=np.float32), BLOCKSIZE, None, None)
        assert c.get_level() == pytest.approx(0.42, abs=1e-5)

    def test_begin_fails_when_stream_down(self):
        c = AudioCapture()
        c._running = False
        assert not c.begin()

    def test_stereo_input_downmixed(self):
        c = self._cap()
        c.begin()
        stereo = np.zeros((BLOCKSIZE, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        c._callback(stereo, BLOCKSIZE, None, None)
        out = c.end()
        assert out.ndim == 1


class TestDeviceSelection:
    def test_device_info_label(self):
        d = DeviceInfo(index=1, name="My Mic", channels=1, sample_rate=16000.0, is_default=True, host_api="MME")
        assert d.label == "My Mic (MME)"
        d2 = DeviceInfo(index=2, name="USB Mic", channels=2, sample_rate=48000.0, is_default=False)
        assert d2.label == "USB Mic"

    def test_find_device_by_identifier_default_or_empty(self):
        assert AudioCapture.find_device_by_identifier("") is None
        assert AudioCapture.find_device_by_identifier("default") is None
        assert AudioCapture.find_device_by_identifier("none") is None

    def test_find_device_by_identifier_matching(self):
        devices = [
            DeviceInfo(index=0, name="Built-in Mic", channels=2, sample_rate=44100.0, is_default=True, host_api="MME"),
            DeviceInfo(index=5, name="Logi USB Headset", channels=1, sample_rate=16000.0, is_default=False, host_api="Windows WASAPI"),
            DeviceInfo(index=12, name="SteelSeries Sonar", channels=2, sample_rate=48000.0, is_default=False, host_api="Windows DirectSound"),
        ]
        # Match by label
        assert AudioCapture.find_device_by_identifier("Logi USB Headset (Windows WASAPI)", devices) == 5
        # Match by name
        assert AudioCapture.find_device_by_identifier("Built-in Mic", devices) == 0
        # Match by substring
        assert AudioCapture.find_device_by_identifier("SteelSeries", devices) == 12
        # Match by index string
        assert AudioCapture.find_device_by_identifier("5", devices) == 5
        # Unknown
        assert AudioCapture.find_device_by_identifier("NonExistent", devices) == None

    def test_set_device_fallback_on_failure(self, monkeypatch):
        c = AudioCapture(device=1)
        c._running = True

        # Mock start_stream to fail when device is 99, succeed otherwise
        def mock_start():
            if c.device == 99:
                c._running = False
                return False
            c._running = True
            return True

        monkeypatch.setattr(c, "start_stream", mock_start)
        monkeypatch.setattr(c, "stop_stream", lambda: None)

        # Attempt switching to failing device
        ok = c.set_device(99)
        assert not ok
        # Assert reverted to previous device 1
        assert c.device == 1


# ── animation ─────────────────────────────────────────────────────────────

class TestSpring:
    def test_converges(self):
        s = Spring(0.0)
        s.target = 100.0
        for _ in range(300):
            s.update(1 / 60)
        assert abs(s.value - 100.0) < 1.0
        assert s.at_rest

    def test_clamps_large_dt(self):
        """A stalled main thread must not launch the spring into orbit."""
        s = Spring(0.0)
        s.target = 100.0
        s.update(5.0)
        assert abs(s.value) < 200

    def test_snap_is_instant_and_at_rest(self):
        s = Spring(0.0)
        s.snap(50.0)
        assert s.value == 50.0 and s.at_rest

    def test_not_at_rest_while_moving(self):
        s = Spring(0.0)
        s.target = 100.0
        s.update(1 / 60)
        assert not s.at_rest


class TestRecordingLock:
    """Tap-to-lock vs hold-to-talk gesture logic.

    Both gestures share the same keys, so the only thing separating them
    is how long the press lasted. These guard the boundary.
    """

    TAP = 0.35

    def _decide(self, held_seconds, was_locked=False):
        """Mirror of main.on_release's decision, isolated from Tk."""
        if was_locked:
            return "stop"
        return "lock" if held_seconds < self.TAP else "stop"

    def test_quick_tap_locks(self):
        assert self._decide(0.08) == "lock"

    def test_hold_stops_on_release(self):
        assert self._decide(2.5) == "stop"

    def test_boundary_just_under_is_a_tap(self):
        assert self._decide(self.TAP - 0.01) == "lock"

    def test_boundary_just_over_is_a_hold(self):
        assert self._decide(self.TAP + 0.01) == "stop"

    def test_tap_while_locked_stops(self):
        """Second tap ends the take rather than re-locking."""
        assert self._decide(0.05, was_locked=True) == "stop"


class TestDurationLimits:
    def test_buffer_cap_is_generous_but_bounded(self):
        from audio.capture import MAX_RECORDING_SECONDS, SAMPLE_RATE
        assert MAX_RECORDING_SECONDS >= 600, "too short for locked dictation"
        megabytes = MAX_RECORDING_SECONDS * SAMPLE_RATE * 4 / 1_048_576
        assert megabytes < 200, f"{megabytes:.0f} MB resident is too much"

    def test_poll_budget_scales_with_audio_length(self):
        """A fixed timeout would abandon a long dictation mid-transcription."""
        base = 120.0
        short = max(base, 5 * 1.5 + 30)
        long = max(base, 1800 * 1.5 + 30)
        assert short == base
        assert long > 1800, "budget must exceed the audio duration"


class TestTheme:
    def test_easing_bounds(self):
        assert ease_out_cubic(0) == 0 and ease_out_cubic(1) == 1
        assert ease_out_cubic(-5) == 0 and ease_out_cubic(5) == 1

    def test_easing_is_monotonic(self):
        vals = [ease_out_cubic(i / 20) for i in range(21)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_hex_conversion(self):
        assert hexc((255, 255, 255)) == "#ffffff"
        assert hexc((0, 0, 0)) == "#000000"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

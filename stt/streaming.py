"""
Streaming transcription via AssemblyAI Universal-Streaming.

Why this matters more than raw WER: with batch transcription the user
finishes speaking and then waits ~1-2s staring at nothing. With streaming,
audio is uploaded *while they talk*, so the wait after they stop is only
the tail -- typically 300-600 ms. Perceived latency drops by roughly 60%
even though total compute is unchanged.

We use it for live partial text in the overlay and, when it succeeds, for
the final transcript too. If the socket fails at any point we fall back to
the batch client for that utterance -- audio is buffered locally the whole
time, so nothing is lost. (This is the one fallback worth keeping: it is
between two paths to the *same* model, not a silent downgrade to a weaker
one.)

Protocol: wss://streaming.assemblyai.com/v3/ws with 16 kHz signed 16-bit
PCM frames. Server emits Turn messages with `transcript`, `end_of_turn`
and `turn_is_formatted`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time
from typing import Callable, List, Optional
from urllib.parse import urlencode

import numpy as np

from .base import TranscriptionResult, WordInfo

logger = logging.getLogger(__name__)

WS_URL = "wss://streaming.assemblyai.com/v3/ws"
SAMPLE_RATE = 16000
CHUNK_MS = 100                      # server wants 50-1000 ms frames
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


def websockets_available() -> bool:
    return importlib.util.find_spec("websockets") is not None


class StreamingSession:
    """One dictation's worth of streaming transcription.

    Usage:
        s = StreamingSession(api_key, keyterms, on_partial)
        await s.open()
        s.feed(samples)      # repeatedly, from the audio thread
        result = await s.close_and_finalise()
    """

    def __init__(
        self,
        api_key: str,
        keyterms: Optional[List[str]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        format_turns: bool = True,
    ):
        self.api_key = api_key
        self.keyterms = keyterms or []
        self.on_partial = on_partial or (lambda _t: None)
        self.format_turns = format_turns

        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None

        self._turns: List[str] = []          # finalised turns, in order
        self._current = ""                   # latest partial
        self._words: List[WordInfo] = []
        self._started = 0.0
        self._samples_sent = 0

        self.ok = False
        self.error: Optional[str] = None
        self._closed = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def open(self, timeout: float = 4.0) -> bool:
        """Connect. Returns False (never raises) so the caller can fall
        back to batch without special-casing exceptions."""
        if not websockets_available():
            self.error = "websockets package not installed"
            return False
        if not self.api_key:
            self.error = "no API key"
            return False

        params = {
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "format_turns": str(self.format_turns).lower(),
        }
        if self.keyterms:
            # Streaming caps keyterms at 100 entries / 50 chars each.
            params["keyterms_prompt"] = json.dumps(
                [t for t in self.keyterms if len(t) <= 50][:100]
            )

        url = f"{WS_URL}?{urlencode(params)}"
        try:
            import websockets
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    additional_headers={"Authorization": self.api_key},
                    ping_interval=20,
                    max_queue=64,
                ),
                timeout=timeout,
            )
        except TypeError:
            # websockets < 14 spells the kwarg differently.
            try:
                import websockets
                self._ws = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        extra_headers={"Authorization": self.api_key},
                        ping_interval=20,
                    ),
                    timeout=timeout,
                )
            except Exception as e:
                self.error = f"connect failed: {e}"
                return False
        except Exception as e:
            self.error = f"connect failed: {e}"
            return False

        self._started = time.perf_counter()
        self._recv_task = asyncio.create_task(self._receive_loop())
        self._send_task = asyncio.create_task(self._send_loop())
        self.ok = True
        return True

    def feed(self, samples: np.ndarray) -> None:
        """Queue float32 mono audio. Safe to call from any thread via
        call_soon_threadsafe, or directly from the loop thread."""
        if not self.ok or self._closed:
            return
        try:
            self._queue.put_nowait(samples)
        except Exception:
            pass

    async def _send_loop(self) -> None:
        buffer = np.zeros(0, dtype=np.float32)
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                buffer = np.concatenate((buffer, np.asarray(chunk, dtype=np.float32).ravel()))
                while len(buffer) >= CHUNK_SAMPLES:
                    frame, buffer = buffer[:CHUNK_SAMPLES], buffer[CHUNK_SAMPLES:]
                    await self._send_pcm(frame)
            if len(buffer):
                await self._send_pcm(buffer)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("stream send loop ended: %s", e)

    async def _send_pcm(self, frame: np.ndarray) -> None:
        pcm = (np.clip(frame, -1.0, 1.0) * 32767.0).astype(np.int16)
        self._samples_sent += len(frame)
        try:
            await self._ws.send(pcm.tobytes())
        except Exception as e:
            self.ok = False
            self.error = f"send failed: {e}"

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except Exception:
                    continue

                mtype = data.get("type")
                if mtype == "Turn":
                    self._handle_turn(data)
                elif mtype == "Termination":
                    break
                elif mtype == "Error":
                    self.error = str(data.get("error", "stream error"))
                    self.ok = False
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("stream receive loop ended: %s", e)

    def _handle_turn(self, data: dict) -> None:
        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            return

        if data.get("end_of_turn"):
            # Only keep formatted finals when we asked for formatting,
            # otherwise the same turn arrives twice (raw then formatted).
            if self.format_turns and not data.get("turn_is_formatted"):
                self._current = transcript
                self.on_partial(self._live_text())
                return
            self._turns.append(transcript)
            self._current = ""
            for w in data.get("words") or []:
                self._words.append(WordInfo(
                    text=w.get("text", ""),
                    confidence=float(w.get("confidence", 1.0)),
                    start=int(w.get("start", 0)),
                    end=int(w.get("end", 0)),
                ))
        else:
            self._current = transcript

        self.on_partial(self._live_text())

    def _live_text(self) -> str:
        parts = self._turns + ([self._current] if self._current else [])
        return " ".join(p for p in parts if p).strip()

    # ── finish ────────────────────────────────────────────────────────────

    async def close_and_finalise(self, timeout: float = 5.0) -> TranscriptionResult:
        """Flush, ask the server to terminate, and collect the transcript."""
        if self._closed:
            return self._result()
        self._closed = True

        try:
            await self._queue.put(None)
            if self._send_task:
                await asyncio.wait_for(self._send_task, timeout=2.0)
        except Exception:
            if self._send_task:
                self._send_task.cancel()

        try:
            if self._ws is not None:
                await self._ws.send(json.dumps({"type": "Terminate"}))
        except Exception:
            pass

        try:
            if self._recv_task:
                await asyncio.wait_for(self._recv_task, timeout=timeout)
        except Exception:
            if self._recv_task:
                self._recv_task.cancel()

        await self._shutdown()
        return self._result()

    async def abort(self) -> None:
        """Tear down without waiting -- used when the user cancels."""
        self._closed = True
        self.ok = False
        for task in (self._send_task, self._recv_task):
            if task:
                task.cancel()
        await self._shutdown()

    async def _shutdown(self) -> None:
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None

    def _result(self) -> TranscriptionResult:
        text = " ".join(self._turns).strip() or self._current.strip()
        if not text and self.error:
            return TranscriptionResult.failure(self.error, "universal-streaming")

        confidence = (
            sum(w.confidence for w in self._words) / len(self._words)
            if self._words else 1.0
        )
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            words=self._words,
            language="en",
            model="universal-streaming",
            latency_ms=int((time.perf_counter() - self._started) * 1000),
            audio_duration_s=self._samples_sent / SAMPLE_RATE,
            ok=True,
        )

"""
Streaming transcription via AssemblyAI's realtime WebSocket.

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

Protocol: wss://streaming.assemblyai.com/v3/ws, 16 kHz signed 16-bit PCM
frames of ~50 ms, plus JSON control frames. Contracts below were read off
the vendor docs on 2026-09-13 and are worth keeping in one place because
each one is a silent-data-loss bug if ignored:

  * Model selection
    https://www.assemblyai.com/docs/streaming/select-the-speech-model
    `speech_model` is a *connection parameter*; omitting it means the
    account default (documented as universal-3-5-pro today). We pin it
    explicitly so the partial/final text on the socket always comes from
    the same model the batch fallback will use.

  * Unrecognised connection parameters are silently ignored, not rejected
    https://www.assemblyai.com/docs/streaming/message-sequence
    "Always check that `configuration.model` matches the `speech_model` you
    requested." So we read the echo in `Begin` and treat a mismatch as a
    degraded session instead of believing our own request.

  * Turn supersede / dedupe
    "Within a turn, each `Turn` message supersedes the previous one. Render
    the latest `transcript`; do not append. A turn is complete on the message
    where both `end_of_turn` and `turn_is_formatted` are `true`."
    With `format_turns=true` the final arrives *twice* (raw, then
    formatted), so completing on `end_of_turn` alone duplicates every turn.

  * Ending the session
    "After you send `Terminate`, keep reading from the WebSocket until you
    receive the `Termination` message... Closing the socket as soon as you
    send `Terminate` silently discards your last transcript."

  * Forcing the last turn out
    `ForceEndpoint` ("for example when your own VAD or a push-to-talk button
    decides the user is done") makes the server return the final Turn
    immediately "without waiting for silence or punctuation" -- i.e. the
    documented way to cut end-of-dictation latency.

  * Throughput guardrails: sessions close at 3 h, or if more than 5 minutes
    of audio queues ahead of processing (error 3007). Both are far outside a
    dictation, but we record the close code so an unexpected one shows up in
    the log instead of looking like a truncated transcript.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time
from typing import Callable, Dict, List, Optional
from urllib.parse import urlencode

import numpy as np

from .base import TranscriptionResult, WordInfo

logger = logging.getLogger(__name__)

WS_URL = "wss://streaming.assemblyai.com/v3/ws"
SAMPLE_RATE = 16000
# Vendor: "For PCM, send audio in chunks of roughly 50 ms (800 samples at
# 16 kHz)". Smaller frames than we used to send also make the *cancel* path
# cheaper: at most one frame of audio can be lost when a session is torn down.
CHUNK_MS = 50
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

#: The three ids the realtime endpoint accepts. Anything else is either a
#: batch-only model or a typo that the server would ignore rather than reject.
STREAMING_MODELS = ("universal-3-5-pro", "universal-streaming-english",
                    "universal-streaming-multilingual")
DEFAULT_STREAM_MODEL = "universal-3-5-pro"

#: Grace windows (seconds) to keep reading after the final control frame.
FINALISE_GRACE = 1.5      # normal stop: wait for the flush + Termination
CANCEL_GRACE = 0.35       # user cancelled: don't make them wait, but flush


def streaming_model_for(batch_model: str) -> str:
    """Pick the realtime model that matches the batch model we're configured
    for, so the streaming result and the batch fallback are the same engine.

    The batch-only ids (`universal-3-5-pro` is shared; `slam-1`,
    `universal-2` are not) map down to the multilingual English model rather
    than being passed through, because an unknown `speech_model` is ignored
    silently and would change behaviour without any error to see.
    """
    m = (batch_model or "").strip()
    if m in STREAMING_MODELS:
        return m
    if m.startswith("universal-3") or m.startswith("universal-3-5"):
        return DEFAULT_STREAM_MODEL
    return "universal-streaming-english"


def websockets_available() -> bool:
    return importlib.util.find_spec("websockets") is not None


class _Turn:
    """One turn's transcript, overwritten by every later message for it."""

    __slots__ = ("text", "final", "words")

    def __init__(self) -> None:
        self.text = ""
        self.final = False
        self.words: Optional[List[WordInfo]] = None


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
        speech_model: Optional[str] = None,
        batch_model: str = "",
    ):
        self.api_key = api_key
        self.keyterms = keyterms or []
        self.on_partial = on_partial or (lambda _t: None)
        self.format_turns = format_turns
        self.speech_model = speech_model or streaming_model_for(
            batch_model or DEFAULT_STREAM_MODEL)

        self._ws = None
        self._recv_task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._send_task: Optional[asyncio.Task] = None

        self._turns: Dict[int, _Turn] = {}
        self._next_order = 0            # ids below this are complete
        self._current_order: Optional[int] = None
        self._current = ""              # rolling partial for turn_order-less models
        self._turn_words: List[List[WordInfo]] = []
        self._started = 0.0
        self._samples_sent = 0

        self.ok = False
        self.error: Optional[str] = None
        self._closed = False
        self._terminating = False
        self.terminated = False
        self.session_id: Optional[str] = None
        self.server_model: Optional[str] = None
        self.server_duration_s: Optional[float] = None
        self.close_code: Optional[int] = None
        #: True once the audio pump stops; feed() after this is a no-op.
        self.degraded = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def connection_params(self) -> dict:
        """Query string sent on connect. Split out so the contract is testable
        without a socket."""
        params = {
            "sample_rate": SAMPLE_RATE,
            "speech_model": self.speech_model,
            "encoding": "pcm_s16le",
            "format_turns": str(self.format_turns).lower(),
        }
        if self.keyterms:
            # Streaming caps keyterms at 100 entries / 50 chars each.
            params["keyterms_prompt"] = json.dumps(
                [t for t in self.keyterms if len(t) <= 50][:100]
            )
        return params

    async def open(self, timeout: float = 4.0) -> bool:
        """Connect. Returns False (never raises) so the caller can fall
        back to batch without special-casing exceptions."""
        if not websockets_available():
            self.error = "websockets package not installed"
            return False
        if not self.api_key:
            self.error = "no API key"
            return False

        url = f"{WS_URL}?{urlencode(self.connection_params())}"
        try:
            import websockets
            try:
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

        self._started = time.perf_counter()
        self._recv_task = asyncio.create_task(self._receive_loop())
        self._send_task = asyncio.create_task(self._send_loop())
        self.ok = True
        return True

    def feed(self, samples: np.ndarray) -> None:
        """Queue float32 mono audio. Safe to call from any thread via
        call_soon_threadsafe, or directly from the loop thread."""
        if not self.ok or self._closed or self.degraded:
            return
        try:
            self._queue.put_nowait(samples)
        except Exception:
            pass

    async def _send_loop(self) -> None:
        # A byte accumulator, not a growing numpy buffer: repeatedly
        # np.concatenate-ing the whole remainder is quadratic in the length
        # of the take (~29 GB of allocation over a 300 s dictation in
        # measurement). Frames are a whole number of samples, so raw bytes
        # of interleaved int16 PCM slice exactly on frame boundaries.
        frame_bytes = CHUNK_SAMPLES * 2
        pending = bytearray()
        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                arr = np.asarray(chunk, dtype=np.float32).ravel()
                if arr.size == 0:
                    continue
                pending += (np.clip(arr, -1.0, 1.0) * 32767.0).astype(
                    np.int16).tobytes()
                while len(pending) >= frame_bytes:
                    await self._send_bytes(bytes(pending[:frame_bytes]))
                    del pending[:frame_bytes]
            if pending:
                await self._send_bytes(bytes(pending))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug("stream send loop ended: %s", e)

    async def _send_bytes(self, pcm: bytes) -> None:
        self._samples_sent += len(pcm) // 2
        try:
            await self._ws.send(pcm)
        except Exception as e:
            self.ok = False
            self.error = f"send failed: {e}"
            self.degraded = True

    async def _send_control(self, payload: dict) -> bool:
        """Send one JSON control frame. Never raises: a dead socket while we
        are trying to close politely must not become an exception on top of
        the real problem."""
        if self._ws is None:
            return False
        try:
            await self._ws.send(json.dumps(payload))
            return True
        except Exception as e:
            logger.debug("control frame %s failed: %s", payload.get("type"), e)
            return False

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
                if mtype == "Begin":
                    self._handle_begin(data)
                elif mtype == "Turn":
                    self._handle_turn(data)
                elif mtype == "Termination":
                    self.terminated = True
                    dur = data.get("audio_duration_seconds")
                    if isinstance(dur, (int, float)):
                        self.server_duration_s = float(dur)
                    break
                elif mtype == "Error":
                    self.error = str(data.get("error") or data.get("message")
                                     or "stream error")
                    self.ok = False
                    self.degraded = True
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # websockets raises on abnormal close; keep the code, an
            # unexpected one is the difference between "user said nothing"
            # and "the provider dropped us mid-sentence".
            self.close_code = getattr(e, "code", None)
            logger.debug("stream receive loop ended: %s (%s)", e,
                         self.close_code if self.close_code else "-")

    def _handle_begin(self, data: dict) -> None:
        """Record the session id and verify the model the server says it is
        actually running. Support asks for `id`; a wrong model would otherwise
        be invisible."""
        self.session_id = data.get("id") or None
        cfg = data.get("configuration") or {}
        applied = cfg.get("model") or cfg.get("speech_model")
        self.server_model = applied
        if applied and applied != self.speech_model:
            self.degraded = True
            self.error = (
                f"server is running '{applied}', not the requested "
                f"'{self.speech_model}'"
            )
            logger.warning(
                "streaming model mismatch (requested %s, got %s); final "
                "transcript will come from the batch path",
                self.speech_model, applied)

    def _handle_turn(self, data: dict) -> None:
        transcript = (data.get("transcript") or "").strip()
        order = data.get("turn_order")
        if not isinstance(order, int):
            # Model without turn_order (older Universal Streaming): one
            # rolling partial plus appended finals, which is what the previous
            # code did and what that model's message stream means.
            if not (bool(data.get("end_of_turn")) and (
                    not self.format_turns or bool(data.get("turn_is_formatted")))):
                self._current = transcript
                self.on_partial(self._live_text())
                return
            order = self._next_order
            self._next_order += 1

        turn = self._turns.setdefault(order, _Turn())
        if transcript:
            turn.text = transcript          # supersedes, never appends
        completed = bool(data.get("end_of_turn")) and (
            not self.format_turns or bool(data.get("turn_is_formatted")))
        if completed:
            turn.final = True
            turn.words = [
                WordInfo(
                    text=w.get("text", ""),
                    confidence=float(w.get("confidence", 1.0)),
                    start=int(w.get("start", 0)),
                    end=int(w.get("end", 0)),
                )
                for w in (data.get("words") or [])
            ]
            # Everything up to this turn is settled; a later partial can only
            # belong to a later turn.
            self._next_order = max(self._next_order, order + 1)
            self._current_order = None
        else:
            self._current_order = order

        self.on_partial(self._live_text())

    def _ordered_turns(self) -> List[_Turn]:
        return [self._turns[k] for k in sorted(self._turns)]

    def _live_text(self) -> str:
        parts = [t.text for t in self._ordered_turns() if t.text]
        cur = self._current_order
        if cur is not None and cur in self._turns and not self._turns[cur].final:
            text = self._turns[cur].text
            if not parts or parts[-1] != text:
                parts.append(text)
        if self._current and (not parts or parts[-1] != self._current):
            parts.append(self._current)
        return " ".join(p for p in parts if p).strip()

    def _unfinal_text(self) -> str:
        """Best text available when the session died before finals arrived.

        Prefers an unformatted `end_of_turn` final over a partial: the server
        had already decided the turn was over, it just never got to polish
        punctuation. Losing those words is what "the tail of my sentence
        disappeared" means.
        """
        parts = [t.text for t in self._ordered_turns() if t.text]
        return " ".join(p for p in parts if p).strip()

    # ── finish ────────────────────────────────────────────────────────────

    async def force_endpoint(self) -> bool:
        """Ask the server to close the current turn right now.

        Called the moment the user releases the hotkey: without it the server
        waits for its own silence/VAD window before emitting the final Turn,
        and that window is exactly the "nothing happens when I let go" delay.
        """
        if self._closed or self.degraded:
            return False
        return await self._send_control({"type": "ForceEndpoint"})

    def _ws_alive(self) -> bool:
        return self._ws is not None

    async def close_and_finalise(self, timeout: float = 5.0) -> TranscriptionResult:
        """Flush, force the endpoint, terminate politely, and collect text."""
        if self._closed:
            return self._result()
        self._closed = True

        # 1. let the pump finish the audio it already has
        # 1. end the in-flight turn before we stop pumping, so the frame goes
        #    out with the socket still warm and the tail is flushed in order.
        await self._send_control({"type": "ForceEndpoint"})

        # 2. stop the pump and let the queued audio drain
        try:
            await self._queue.put(None)
            if self._send_task:
                await asyncio.wait_for(self._send_task, timeout=2.0)
        except Exception:
            if self._send_task:
                self._send_task.cancel()

        await self._drain(FINALISE_GRACE)

        # 3. terminate and keep reading: the vendor is explicit that closing
        #    right after Terminate "silently discards your last transcript".
        self._terminating = True
        await self._send_control({"type": "Terminate"})
        await self._drain(max(timeout - FINALISE_GRACE, 0.5))

        await self._shutdown()
        return self._result()

    async def _drain(self, seconds: float) -> None:
        """Wait for the receive task to reach its last message, briefly."""
        if not self._recv_task or self._recv_task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._recv_task),
                                   timeout=seconds)
        except Exception:
            pass

    async def abort(self) -> None:
        """Tear down after a cancel. Still says goodbye: an abrupt close
        leaves the session open and billed on connection duration, and the
        server's final frames are free to read if we stay half a second."""
        self._closed = True
        self.ok = False
        self.degraded = True
        if self._ws is not None:
            await self._send_control({"type": "Terminate"})
            await self._drain(CANCEL_GRACE)
        for task in (self._send_task, self._recv_task):
            if task and not task.done():
                task.cancel()
        await self._shutdown()

    async def _shutdown(self) -> None:
        try:
            if self._ws is not None:
                await self._ws.close()
        except Exception:
            pass
        self._ws = None

    # ── result ────────────────────────────────────────────────────────────

    def _result(self) -> TranscriptionResult:
        finals = [t for t in self._ordered_turns() if t.final and t.text]
        text = " ".join(t.text for t in finals).strip()
        words: List[WordInfo] = []
        for t in finals:
            words.extend(t.words or [])
        if not text:
            # Nothing was ever *formatted*. Fall back to whatever partial or
            # unformatted-final text the server did send, so an early socket
            # close costs punctuation instead of the user's words; only
            # report a failure (batch takes over) when there is nothing at all.
            live = self._unfinal_text()
            if live and not self.error:
                return self._wrap(live, words)
            if self.error:
                return TranscriptionResult.failure(self.error, self.speech_model)
            if not live:
                return TranscriptionResult(
                    text="", model=self.speech_model, ok=True,
                    latency_ms=int((time.perf_counter() - self._started) * 1000),
                    audio_duration_s=self._audio_seconds())
            return self._wrap(live, words)
        return self._wrap(text, words)

    def _wrap(self, text: str, words: List[WordInfo]) -> TranscriptionResult:
        confidence = (sum(w.confidence for w in words) / len(words)
                      if words else 1.0)
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            words=words,
            language="en",
            model=self.speech_model,
            latency_ms=int((time.perf_counter() - self._started) * 1000),
            audio_duration_s=self._audio_seconds(),
            ok=True,
        )

    def _audio_seconds(self) -> float:
        """Prefer the server's own tally -- it knows what it actually heard."""
        if self.server_duration_s is not None:
            return self.server_duration_s
        return self._samples_sent / SAMPLE_RATE

    def get_info(self) -> dict:
        return {
            "speech_model": self.speech_model,
            "server_model": self.server_model,
            "session_id": self.session_id,
            "terminated": self.terminated,
            "degraded": self.degraded,
            "close_code": self.close_code,
            "error": self.error,
        }

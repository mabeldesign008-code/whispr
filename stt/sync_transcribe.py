"""
AssemblyAI **Sync** STT: one HTTP request, transcript in the response.

Why this exists at all (audit C12). The async flow costs three round trips
(POST /v2/upload, POST /v2/transcript, GET /v2/transcript/{id}) plus a poll
interval, so a two-second dictation waits ~1.5-2.5 s for text *after* the user
stopped talking. Sync posts the audio and gets the transcript in the same
exchange: the API reference's own example response carries
``request_time_ms: 243.7`` for a 101 s clip. That is the whole difference
between "feels instant" and "feels broken" for a push-to-talk tool.

Everything in this module follows the vendor contract as documented on
2026-09-13; the URLs are cited at each decision.

  * Endpoint and auth
    https://www.assemblyai.com/docs/api-reference/sync-api/transcribe
      POST https://sync.assemblyai.com/transcribe
      (https://sync.us.assemblyai.com / https://sync.eu.assemblyai.com for
      pinned data zones -- https://www.assemblyai.com/docs/sync-stt/endpoints-and-data-zones)
      header `Authorization: <key>` (Bearer optional, raw key documented as
      the norm), `X-AAI-Model: universal-3-5-pro` is **required**, body is
      multipart with an `audio` file part and an optional `config` JSON part.
  * Live upload
    https://www.assemblyai.com/docs/api-reference/sync-api/transcribe-live
      POST https://sync.assemblyai.com/v1/transcribe/live
      `config` is **required and must be the first part, ahead of `audio`**;
      "Send an empty object ({}) when you have no options to set"; audio is
      "uploaded in chunks as they are produced", Content-Type `audio/pcm`
      for raw S16LE. This is what lets the upload overlap with the user
      talking, which is strictly better than sending bytes after they stop.
  * Audio constraints
    https://www.assemblyai.com/docs/sync-stt/audio-requirements
      min 80 ms, max 120 s, max 40 MB, 16-bit only, mono or stereo,
      sample rates {8000,16000,22050,24000,32000,44100,48000}, WAV or raw
      S16LE PCM; "For raw PCM, pass sample_rate and channels in the config
      part"; over 120 s "use Pre-recorded STT" (that is the async client).
  * Keyterms / prompt caps
    https://www.assemblyai.com/docs/sync-stt/prompting-and-keyterms
      `keyterms_prompt` array, "Maximum: **100 terms**, **8000 characters**
      total across all terms"; prompt <= 6000 chars; and the explicit warning
      that "Including a large number of terms or common terms ... could lead
      to overcorrections and hallucinations".
  * Errors
    https://www.assemblyai.com/docs/sync-stt/error-handling
      body is {"error_code", "message"} for audio/capacity/inference failures
      and {"detail"} for auth/rate-limit; 429 and 503 are transient and
      honour `Retry-After`, 400/413/415 mean the request itself is wrong (do
      not retry), 500/504 are "safe to retry once". Include `session_id` in
      support requests.
  * Pre-warming
    https://www.assemblyai.com/docs/sync-stt/connection-pre-warming
      `GET /warm` is an unauthenticated no-op whose only job is to do DNS +
      TCP + TLS up front; "httpx drops idle connections after 5 seconds by
      default", so warming at *recording start* would be worthless and the
      correct moment is when we know audio is coming but don't have it yet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, List, Optional

import httpx
import numpy as np

from .base import TranscriptionResult, WordInfo

logger = logging.getLogger(__name__)

SYNC_TRANSCRIBE_PATH = "/transcribe"
SYNC_LIVE_PATH = "/v1/transcribe/live"
SYNC_WARM_PATH = "/warm"
#: `X-AAI-Model` is required; the reference lists exactly one canonical value
#: (with u3-sync-pro / u3-pro accepted as legacy aliases).
SYNC_MODEL = "universal-3-5-pro"
SYNC_MODELS = (SYNC_MODEL, "u3-sync-pro", "u3-pro")

MIN_AUDIO_MS = 80
MAX_AUDIO_SECONDS = 120.0
MAX_BODY_BYTES = 40 * 1024 * 1024
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

#: Sync keyterm caps, straight from the prompting page. The async API accepts
#: more, but overcorrection is a documented risk of a big term list, so we
#: honour the tighter Sync numbers on the Sync path.
MAX_KEYTERMS = 100
MAX_KEYTERMS_CHARS = 8000
MAX_PROMPT_CHARS = 6000

#: Words that are common enough that biasing toward them only adds risk of
#: "correcting" real speech into them. Vendor: "Do not include single, common
#: English words (e.g. 'information') as keyterms."
COMMON_SINGLE_WORDS = frozenset("""
a an the and or but if so nor for to of in on at by with from up down out
off over under again then once here there this that these those it its its
is was were be been being am are do does did doing have has had having
will would shall should can could may might must need let us you your we
our they them their he she his her i me my what which who whom whose when
where why how all any both each few more most other some such no nor not
only own same than too very just also like okay ok yeah please thanks hello hi
""".split())


def sync_supported(samples: int, sample_rate: int,
                   duration_s: Optional[float] = None) -> bool:
    """Can this take go through the Sync endpoint at all?"""
    rate = int(sample_rate)
    if rate not in SUPPORTED_SAMPLE_RATES:
        return False
    dur = duration_s if duration_s is not None else (
        samples / float(rate) if rate else 0.0)
    # 16-bit mono PCM: 2 bytes per sample.
    if samples * 2 > MAX_BODY_BYTES:
        return False
    return 0.08 <= dur <= MAX_AUDIO_SECONDS


def sync_keyterms(terms: List[str]) -> List[str]:
    """Filter + cap a term list to the documented Sync limits.

    Drops the handful of stop-word-ish single words a learner is most likely
    to have collected ("like", "ok", "yeah") -- the vendor's warning that
    common terms "could lead to overcorrections and hallucinations" is the
    whole reason -- phrases over 6 words (the async cap), duplicates, then
    trims to 100 terms / 8000 characters total.

    Deliberately *not* a general stop-word filter: a user dictionary is by
    definition unusual vocabulary, and silently dropping a term the user typed
    is worse than a slightly over-eager bias.
    """
    out: List[str] = []
    seen = set()
    used = 0
    for raw in terms or []:
        t = (raw or "").strip()
        if not t:
            continue
        words = t.split()
        if len(words) > 6:
            continue
        if len(words) == 1 and t.lower().strip(".,;:!?'\"") in COMMON_SINGLE_WORDS:
            continue
        key = t.lower()
        if key in seen:
            continue
        if len(out) >= MAX_KEYTERMS:
            break
        if used + len(t) + 1 > MAX_KEYTERMS_CHARS:
            break
        seen.add(key)
        used += len(t) + 1
        out.append(t)
    return out


def clip_prompt(text: str) -> str:
    """The prompt field is capped at 6000 characters; keep the head, which is
    where the descriptive part of a context blurb lives."""
    text = (text or "").strip()
    if len(text) <= MAX_PROMPT_CHARS:
        return text
    return text[:MAX_PROMPT_CHARS].rsplit(" ", 1)[0]


def float_to_pcm_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Raw S16LE mono PCM -- what the Sync API wants for live audio.

    Raw rather than WAV because a WAV header would have to describe a length
    we do not know yet while the take is still being recorded, and because
    44 bytes of header is pure noise on the wire.
    """
    arr = np.asarray(samples, dtype=np.float32).ravel()
    if arr.size == 0:
        return b""
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


@dataclass
class SyncConfig:
    """The `config` part of the multipart body."""

    sample_rate: int = 16000
    channels: int = 1
    timestamps: bool = True
    prompt: str = ""
    keyterms_prompt: List[str] = field(default_factory=list)
    conversation_context: List[dict] = field(default_factory=list)
    language_code: str = ""

    def as_json(self, raw_pcm: bool = True) -> str:
        body: dict = {"timestamps": bool(self.timestamps)}
        if raw_pcm:
            # Required for headerless audio: "For raw PCM, pass sample_rate
            # and channels in the config part". Omitting them is a 400
            # bad_audio with "missing sample_rate/channels for PCM".
            body["sample_rate"] = int(self.sample_rate)
            body["channels"] = int(self.channels)
        if self.prompt:
            body["prompt"] = clip_prompt(self.prompt)
            # Documented: language_code is ignored when a custom prompt is
            # set, so don't send a field that would silently do nothing.
        elif self.language_code:
            body["language_code"] = self.language_code
        if self.keyterms_prompt:
            body["keyterms_prompt"] = list(self.keyterms_prompt)
        if self.conversation_context:
            body["conversation_context"] = list(self.conversation_context)
        return json.dumps(body)


@dataclass
class SyncOutcome:
    """What the caller needs in order to decide: use it, retry, or fall back."""

    result: Optional[TranscriptionResult] = None
    retry_after: Optional[float] = None
    #: The async upload/submit path is the right answer (too long, too big).
    fall_back: bool = False
    #: Transient provider trouble; a retry may work but the user is waiting.
    transient: bool = False
    #: Do not try the other endpoint: the failure is about the credential or
    #: the account, and every endpoint will agree about it.
    fatal: bool = False
    status: int = 0
    error_code: str = ""
    session_id: Optional[str] = None
    request_time_ms: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok


def _retry_after(resp: httpx.Response) -> Optional[float]:
    raw = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 5.0))
    except ValueError:
        return 1.0


class SyncTranscriber:
    """One endpoint, one request, no polling.

    The `base_url` argument exists for tests: the mock server in
    ``eval/`` implements the same multipart contract on localhost.
    """

    model = SYNC_MODEL

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://sync.assemblyai.com",
        client: Optional[httpx.AsyncClient] = None,
        request_timeout: float = 30.0,
        model: str = SYNC_MODEL,
        allow_retry: bool = True,
    ):
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self._own_client = client is None
        self._client = client
        self.request_timeout = request_timeout
        self.model = model if model in SYNC_MODELS else SYNC_MODEL
        self.allow_retry = allow_retry

        self.calls = 0
        self.failures = 0
        self.fallbacks = 0            # handed the take to the async path
        self.last_error: Optional[str] = None
        self.last_session_id: Optional[str] = None
        self.last_request_time_ms: Optional[float] = None
        self.retried: int = 0

    # -- client ------------------------------------------------------------

    def owns_client(self) -> bool:
        return self._own_client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout, connect=10.0,
                                      # A stalled pool must fail fast and
                                      # visibly rather than impersonate a
                                      # provider timeout (default is 5 s).
                                      pool=2.0),
                headers={
                    "Authorization": self.api_key,      # raw key: Bearer optional
                    "X-AAI-Model": self.model,          # required on every request
                },
                limits=httpx.Limits(max_keepalive_connections=4,
                                    # The vendor calls out httpx's 5 s idle
                                    # eviction as the reason a warm connection
                                    # goes missing; outliving a typical pause
                                    # between dictations costs nothing.
                                    keepalive_expiry=60.0),
                http2=_http2_available(),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed and self._own_client:
            await self._client.aclose()
        self._client = None

    async def warm(self) -> bool:
        """DNS + TCP + TLS, off the critical path.

        Call it *immediately before* the request -- not at app start, because
        the pool evicts. Idempotent and unauthenticated.
        """
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}{SYNC_WARM_PATH}",
                                    timeout=3.0)
            return resp.status_code < 500
        except Exception as e:
            logger.debug("sync warm failed: %s", e)
            return False

    # -- one-shot ----------------------------------------------------------

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        config: Optional[SyncConfig] = None,
    ) -> SyncOutcome:
        arr = np.asarray(samples, dtype=np.float32).ravel()
        rate = int(sample_rate) or 16000
        duration = arr.size / float(rate)
        cfg = config or SyncConfig(sample_rate=rate)

        if not self.api_key:
            return SyncOutcome(result=TranscriptionResult.failure(
                "No AssemblyAI API key.", self.model), error_code="no_key")
        if duration * 1000 < MIN_AUDIO_MS:
            return SyncOutcome(fall_back=True, error_code="audio_too_short",
                                result=TranscriptionResult.failure(
                                    f"Audio shorter than {MIN_AUDIO_MS} ms.",
                                    self.model))
        if not sync_supported(arr.size, rate, duration):
            # Too long or too big for this endpoint: not an error, just the
            # other endpoint's job. The async client handles <= 120 s fine too,
            # so callers can treat this as "use the fallback".
            return SyncOutcome(fall_back=True, error_code="not_for_sync")

        pcm = float_to_pcm_bytes(arr, rate)
        # httpx's `files=` shape is (name, (filename, value, content_type));
        # filename=None makes the part carry no filename, which is how a
        # non-file part such as `config` is meant to look.
        files = [
            ("config", (None, cfg.as_json(raw_pcm=True).encode(), "application/json")),
            ("audio", ("take.pcm", pcm, "audio/pcm")),
        ]
        return await self._post(SYNC_TRANSCRIBE_PATH, files, duration=duration)

    async def _post(self, path: str, files, duration: float) -> SyncOutcome:
        self.calls += 1
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                client = await self._get_client()
                resp = await client.post(
                    f"{self.base_url}{path}",
                    files=files,
                    timeout=self.request_timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self.failures += 1
                self.last_error = f"{type(e).__name__}: {e}"
                if attempt == 1 and self.allow_retry:
                    self.retried += 1
                    await asyncio.sleep(0.15)
                    continue
                return SyncOutcome(transient=True,
                                   result=TranscriptionResult.failure(
                                       self.last_error, self.model))

            outcome = self._handle_response(resp, started, duration)
            retryable = (
                outcome.transient
                and self.allow_retry
                and attempt == 1
                and (outcome.retry_after or 0.0) <= 1.0
            )
            if retryable:
                # 429/503 with no/short Retry-After: one quick retry, because
                # a dictation is worthless once the user has moved on, and the
                # vendor explicitly calls these transient.
                self.retried += 1
                await asyncio.sleep(outcome.retry_after or 0.4)
                continue
            return outcome

    def _handle_response(self, resp: httpx.Response, started: float,
                         duration: float) -> SyncOutcome:
        status = resp.status_code
        if status == 200:
            try:
                data = resp.json()
            except Exception as e:
                self.failures += 1
                self.last_error = f"unparsable success body: {e}"
                return SyncOutcome(status=status,
                                   result=TranscriptionResult.failure(
                                       self.last_error, self.model))
            sid = data.get("session_id")
            self.last_session_id = sid
            rtm = data.get("request_time_ms")
            if isinstance(rtm, (int, float)):
                self.last_request_time_ms = float(rtm)
            result = self._parse(data, started, duration)
            return SyncOutcome(result=result, status=status, session_id=sid,
                               request_time_ms=self.last_request_time_ms)

        code, message = _error_fields(resp)
        self.last_error = f"HTTP {status}{(': ' + message) if message else ''}"

        if status == 401:
            # Worth being blunt: this is a broken install, not a bad take.
            self.failures += 1
            return SyncOutcome(
                status=status, error_code="bad_key", fatal=True,
                result=TranscriptionResult.failure(
                    "AssemblyAI rejected the API key (HTTP 401). "
                    "Re-save it in Settings.", self.model))
        if status in (429, 503):
            self.failures += 1
            return SyncOutcome(status=status, error_code=code, transient=True,
                               retry_after=_retry_after(resp),
                               result=TranscriptionResult.failure(
                                   self.last_error, self.model))
        if status in (500, 504):
            # "safe to retry once"
            self.failures += 1
            return SyncOutcome(status=status, error_code=code, transient=True,
                               result=TranscriptionResult.failure(
                                   self.last_error, self.model))
        if status in (400, 413, 415):
            # Our request is wrong; the async endpoint is the documented
            # answer for >120 s, and for anything else a fallback at least
            # gets the user their text instead of an error dialog.
            self.fallbacks += 1
            return SyncOutcome(status=status, error_code=code, fall_back=True)
        self.failures += 1
        return SyncOutcome(status=status, error_code=code, fall_back=True,
                           transient=True)

    def _parse(self, data: dict, started: float, duration: float) -> TranscriptionResult:
        words = [
            WordInfo(
                text=w.get("text", ""),
                confidence=float(w.get("confidence", 1.0)),
                start=int(w.get("start", 0)),
                end=int(w.get("end", 0)),
            )
            for w in (data.get("words") or [])
        ]
        confidence = data.get("confidence")
        if confidence is None:
            confidence = sum(w.confidence for w in words) / len(words) if words else 1.0
        server_ms = data.get("request_time_ms")
        latency = int(server_ms) if isinstance(server_ms, (int, float)) else \
            int((time.perf_counter() - started) * 1000)
        audio_ms = data.get("audio_duration_ms")
        return TranscriptionResult(
            text=(data.get("text") or "").strip(),
            confidence=float(confidence),
            words=words,
            # Sync's response carries no language_code field; only the
            # language we pinned (if any) is known.
            language=data.get("language_code") or None,
            model=self.model,
            session_id=data.get("session_id"),
            request_time_ms=int(server_ms) if isinstance(server_ms, (int, float)) else None,
            latency_ms=latency,
            audio_duration_s=(audio_ms / 1000.0) if isinstance(audio_ms, (int, float))
            else duration,
            ok=True,
        )

    def get_stats(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "failures": self.failures,
            "fallbacks": self.fallbacks,
            "retried": self.retried,
            "last_error": self.last_error,
            "last_session_id": self.last_session_id,
            "last_request_time_ms": self.last_request_time_ms,
        }


# -- live upload -----------------------------------------------------------


class LiveSyncSession:
    """Stream PCM to `/v1/transcribe/live` while the user is still talking.

    The vendor's framing: "authorization, the upload, and every speech
    segment but the last are done by the time the speaker stops. What is left
    to wait for is the final segment." For push-to-talk that is the best case
    available without a WebSocket, and it degrades to nothing if we never
    open it -- the caller keeps the local buffer either way.

    `write()` is callable from any thread (it hands bytes to an asyncio
    queue); `close()` ends the audio; `result()` awaits the transcript.
    """

    def __init__(
        self,
        transcriber: SyncTranscriber,
        config: SyncConfig,
        on_state: Optional[Callable[[str], None]] = None,
        chunk_frames: int = 800,        # 50 ms at 16 kHz
    ):
        self.t = transcriber
        self.config = config
        self.on_state = on_state or (lambda _s: None)
        self.chunk_frames = max(160, int(chunk_frames))

        self._boundary = "----whisprflow" + format(time.time_ns(), "x")
        self._queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._closed = False
        self._outcome = SyncOutcome(fall_back=True)
        self._samples = 0
        self._opened = False
        self._failed = False
        self.error: Optional[str] = None
        self.started_at = 0.0

    @property
    def active(self) -> bool:
        return self._opened and not self._failed

    @property
    def seconds(self) -> float:
        rate = self.config.sample_rate or 16000
        return self._samples / float(rate)

    async def open(self) -> bool:
        """Start the request. Returns False (never raises) on any problem, in
        which case the caller's local buffer is still complete."""
        if not self.t.api_key:
            self.error = "no API key"
            return False
        self._opened = True
        self.started_at = time.perf_counter()
        self._task = asyncio.create_task(self._run())
        return True

    def write(self, samples: np.ndarray) -> None:
        """Queue a chunk of float32 mono audio. Never blocks, never raises."""
        if not self.active or self._closed:
            return
        if self.seconds >= MAX_AUDIO_SECONDS:
            # Past the documented cap the server will only answer 413, so stop
            # feeding and let the caller finish from its local buffer.
            self._failed = True
            self.error = "live audio exceeded the 120 s Sync limit"
            self.on_state("capped")
            return
        pcm = float_to_pcm_bytes(samples, self.config.sample_rate)
        if not pcm:
            return
        self._samples += len(samples)
        try:
            self._queue.put_nowait(pcm)
        except Exception as e:                      # pragma: no cover - defensive
            self._failed = True
            self.error = f"queue failed: {e}"

    async def warm(self) -> bool:
        """Pre-warm the connection for the caller (recording just started and
        the queue is still empty). Same pool, so /transcribe reuses it."""
        return await self.t.warm()

    async def close(self) -> SyncOutcome:
        """End the audio and wait for the transcript."""
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=35.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                self.error = self.error or "live request timed out"
            except Exception as e:
                self.error = self.error or f"{type(e).__name__}: {e}"
        return self._outcome

    async def abort(self) -> None:
        """Give up on the live request (user cancelled)."""
        self._closed = True
        self._failed = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._drain_queue()

    def _drain_queue(self) -> None:
        while True:
            try:
                if self._queue.get_nowait() is None:
                    return
            except asyncio.QueueEmpty:
                return
            except Exception:
                return

    async def _iter_body(self) -> AsyncIterator[bytes]:
        """Hand-rolled multipart: the `config` part has to be flushed *first*
        so the server can parse the config while audio is still arriving."""
        boundary = self._boundary
        cfg = self.config.as_json(raw_pcm=True).encode()
        yield (f"--{boundary}\r\n"
               'Content-Disposition: form-data; name="config"\r\n'
               "Content-Type: application/json\r\n\r\n").encode()
        yield cfg
        yield b"\r\n"
        yield (f"--{boundary}\r\n"
               'Content-Disposition: form-data; name="audio"; filename="live.pcm"\r\n'
               "Content-Type: audio/pcm\r\n\r\n").encode()
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk
        yield f"\r\n--{boundary}--\r\n".encode()

    async def _run(self) -> None:
        try:
            client = await self.t._get_client()
            # The boundary must be identical in the header and the body, so it
            # is generated once in __init__ rather than per call.
            headers = {"content-type": f"multipart/form-data; boundary={self._boundary}"}
            resp = await client.post(
                f"{self.t.base_url}{SYNC_LIVE_PATH}",
                content=self._iter_body(),
                headers=headers,
                timeout=httpx.Timeout(self.t.request_timeout, connect=10.0,
                                      # Uploading in chunks takes as long as the
                                      # user talks; only the *finish* needs a cap.
                                      read=max(self.t.request_timeout, 30.0)),
            )
            self.on_state("ended")
            self._outcome = self.t._handle_response(resp, self.started_at,
                                                     self.seconds)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._failed = True
            self.error = f"{type(e).__name__}: {e}"
            self.t.failures += 1
            self.t.last_error = self.error
            self._outcome = SyncOutcome(fall_back=True, transient=True,
                                        error_code="live_failed")
            self.on_state("failed")

    def result(self) -> SyncOutcome:
        return self._outcome


def _error_fields(resp: httpx.Response) -> tuple:
    """Sync errors are {"error_code","message"} or {"detail"} -- both shapes
    are documented, so read either and never trust one."""
    try:
        data = resp.json()
    except Exception:
        return "", (resp.text or "").strip()[:200]
    if not isinstance(data, dict):
        return "", str(data)[:200]
    return (str(data.get("error_code") or ""),
            str(data.get("message") or data.get("detail") or "")[:200])


def _http2_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("h2") is not None

"""
AssemblyAI transcription.

The only transcription engine. No local fallback: a silent downgrade to a
weaker model is worse than an honest error, because the user cannot tell
their text got worse and ships it anyway.

Model: universal-3-5-pro, AssemblyAI's most accurate async model
(~2-5% WER on clean English). Replaces SenseVoice-Small at ~14.7%.

Flow: POST /v2/upload -> POST /v2/transcript -> poll GET /v2/transcript/{id}.
No official SDK: it is synchronous and would block the asyncio loop.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from typing import List, Optional

import httpx
import logging
import numpy as np

from .base import TranscriptionResult, WordInfo, float_to_wav_bytes
from .sync_transcribe import (
    MAX_AUDIO_SECONDS,
    LiveSyncSession,
    SyncConfig,
    SyncTranscriber,
    sync_keyterms,
    sync_supported,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.assemblyai.com"

#: Sync STT lives on a different host, and a warmed/kept-alive connection to
#: one does not help the other ("a connection warmed against one endpoint
#: doesn't help a request sent to another"):
#: https://www.assemblyai.com/docs/sync-stt/connection-pre-warming
SYNC_BASE_URL = os.getenv("ASSEMBLYAI_SYNC_URL", "https://sync.assemblyai.com").strip()

# Models accepting keyterms_prompt. universal-2 uses the older word_boost
# and returns 400 if keyterms are sent.
KEYTERM_MODELS = {"universal-3-5-pro", "universal-3-pro", "slam-1"}


def _http2_available() -> bool:
    """httpx raises ImportError at construction if http2=True without `h2`.
    HTTP/1.1 works fine, so detect rather than crash."""
    return importlib.util.find_spec("h2") is not None


class AssemblyAIClient:
    name = "assemblyai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "universal-3-5-pro",
        language_code: str = "en",
        request_timeout: float = 30.0,
        poll_interval: float = 0.05,
        max_wait: float = 120.0,
        sync_enabled: bool = True,
        sync_url: str = "",
        live_upload: bool = False,
        keyterms_limit: Optional[int] = None,
    ):
        self.api_key = (api_key or os.getenv("ASSEMBLYAI_API_KEY", "")).strip()
        self.model = model
        self.language_code = language_code
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait
        self.keyterms_limit = keyterms_limit
        self.sync_enabled = bool(sync_enabled) and os.getenv(
            "WHISPRFLOW_SYNC_STT", "1").strip().lower() not in ("0", "false", "no", "off")
        self._sync_url = sync_url
        self._live_upload = bool(live_upload) or os.getenv(
            "WHISPRFLOW_SYNC_LIVE_UPLOAD", "0").strip().lower() in ("1", "true", "yes", "on")

        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        # Shared client: /warm and /transcribe must use the same connection
        # pool, or the pre-warm buys nothing (the vendor calls this out
        # explicitly under connection pre-warming).
        self.sync = SyncTranscriber(
            self.api_key,
            base_url=self._sync_url or SYNC_BASE_URL,
            # A live upload's request stays open as long as the user talks, so
            # the read timeout has to outlast a two-minute take; the one-shot
            # path passes its own per-request timeout anyway.
            request_timeout=max(self.request_timeout, 60.0),
        ) if self.sync_enabled else None
        #: set by open_live(); the pipeline closes it.
        self._live: Optional[LiveSyncSession] = None
        self.fallbacks_to_async = 0
        self.sync_hits = 0
        self.last_error: Optional[str] = None
        self.last_session_id: Optional[str] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, key: str) -> None:
        self.api_key = (key or "").strip()
        self._client = None  # force rebuild with new auth header
        if self.sync is not None:
            self.sync.api_key = self.api_key
            self.sync._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(
                    base_url=BASE_URL,
                    timeout=httpx.Timeout(self.request_timeout, connect=10.0),
                    headers={"authorization": self.api_key},
                    http2=_http2_available(),
                    # Keep TLS warm between dictations; a cold handshake
                    # costs ~200ms on every utterance otherwise.
                    limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300),
                )
            return self._client

    async def warmup(self) -> None:
        """Open the HTTPS connection before the first dictation."""
        if not self.is_configured:
            return
        try:
            client = await self._get_client()
            await client.get("/v2/transcript", params={"limit": 1}, timeout=5.0)
        except Exception:
            pass  # best effort

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if self.sync is not None:
            await self.sync.close()

    # -- account health ----------------------------------------------------

    async def validate_key(self) -> tuple:
        """Ask the API whether the key works, before we ever send audio.

        `GET /v2/account` with the raw key in `authorization` (no Bearer) is
        the documented authentication smoke test -- AssemblyAI's own wording
        is "If your key is valid, you get a 200 ... If it's missing or wrong,
        you get a 401. That's your authentication smoke test before you send
        any audio."
        (https://www.assemblyai.com/blog/speech-to-text-api-fundamentals,
         https://docs.redhuntlabs.com/docs/exposure-risks/credentials/assemblyai_api_key)

        Returns (ok, message). Never raises: a network blip must not be
        reported to the user as "your key is invalid".
        """
        if not self.is_configured:
            return False, "No API key set."
        try:
            client = await self._get_client()
            resp = await client.get("/v2/account", timeout=8.0)
        except Exception as e:
            return None, f"Could not reach AssemblyAI to check the key: {e}"
        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            bits = []
            for k in ("credits_amount", "credits_type", "name", "email"):
                v = data.get(k) if isinstance(data, dict) else None
                if v not in (None, ""):
                    bits.append(f"{k}={v}")
            return True, "Key accepted" + ((" (" + ", ".join(bits) + ")") if bits else "")
        if resp.status_code == 401:
            return False, "AssemblyAI rejected this key (HTTP 401). Check for stray spaces or a revoked key."
        if resp.status_code == 403:
            return False, "Key rejected with HTTP 403 (permissions/payment)."
        return False, f"Unexpected reply checking the key (HTTP {resp.status_code})."

    def sync_available(self) -> bool:
        """Whether the one-shot Sync endpoint will be tried for a short take.

        Named apart from the `sync_enabled` constructor flag, which is kept
        as a plain attribute -- the two would otherwise shadow each other.
        """
        return self.sync is not None and self.is_configured

    def has_live(self) -> bool:
        return self._live is not None

    async def warm_sync(self) -> bool:
        """Take DNS+TCP+TLS off the critical path of the next take."""
        if self.sync is None:
            return False
        return await self.sync.warm()

    # -- live upload (Sync streaming body) --------------------------------

    def live_upload_enabled(self) -> bool:
        return self._live_upload and self.sync is not None and self.is_configured

    async def open_live(self, sample_rate: int, keyterms=None,
                        prompt: str = "", language_code: str = "") -> Optional[LiveSyncSession]:
        """Start uploading the take as it is recorded.

        Optional and off by default: it duplicates the same audio the
        WebSocket already receives, so enabling it means choosing it over
        streaming (see docs/sync-live-upload.md). Everything stays
        fall-through -- if this never opens, nothing else notices.
        """
        if not self.live_upload_enabled() or self._live is not None:
            return None
        cfg = SyncConfig(
            sample_rate=sample_rate, channels=1, timestamps=True,
            prompt=prompt or "",
            keyterms_prompt=sync_keyterms(list(keyterms or [])),
            language_code=language_code or self.language_code,
        )
        session = LiveSyncSession(self.sync, cfg)
        if not await session.open():
            return None
        self._live = session
        return session

    def feed_live(self, samples) -> None:
        """Called from the pump loop with whatever was captured since last time.
        Never raises, never blocks: a dead live session must not cost the user
        anything, because the local buffer is the source of truth."""
        if self._live is None:
            return
        try:
            self._live.write(samples)
        except Exception:
            self._live = None

    async def finish_live(self) -> Optional[TranscriptionResult]:
        """Close the live request and take its transcript if we have one."""
        session, self._live = self._live, None
        if session is None:
            return None
        try:
            outcome = await session.close()
        except Exception as e:
            logger.debug("live upload failed: %s", e)
            return None
        if outcome.result is not None and outcome.result.ok and outcome.result.text.strip():
            return outcome.result
        return None

    async def abort_live(self) -> None:
        session, self._live = self._live, None
        if session is not None:
            try:
                await session.abort()
            except Exception:
                pass

    # ── transcription ─────────────────────────────────────────────────────

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        keyterms: Optional[List[str]] = None,
        prompt: str = "",
        live: Optional[TranscriptionResult] = None,
    ) -> TranscriptionResult:
        """Transcribe a take.

        Order of preference, cheapest first, all the way down to the path
        this app shipped with:

          1. a live-upload transcript that is already finished (audio arrived
             while the user was still talking);
          2. Sync: one POST to sync.assemblyai.com, transcript in the
             response, no polling (~130-250 ms in the vendor's own example);
          3. async: upload -> submit -> poll, which is the only path for
             takes longer than the documented 120 s Sync cap.
        """
        if live is not None and live.ok and live.text.strip():
            self.sync_hits += 1
            return live

        started = time.perf_counter()
        duration = len(samples) / float(sample_rate) if sample_rate else 0.0

        if not self.is_configured:
            self.last_error = "no API key"
            return TranscriptionResult.failure(
                "No AssemblyAI API key. Add one in Settings.", self.model
            )
        if samples is None or len(samples) == 0:
            return TranscriptionResult.failure("No audio to transcribe.", self.model)

        if self.sync is not None and sync_supported(len(samples), sample_rate, duration):
            outcome = await self.sync.transcribe(
                samples, sample_rate,
                SyncConfig(
                    sample_rate=sample_rate,
                    prompt=prompt or "",
                    keyterms_prompt=sync_keyterms(list(keyterms or [])),
                    language_code=self.language_code,
                ),
            )
            self.last_session_id = outcome.session_id or self.last_session_id
            if outcome.ok:
                self.sync_hits += 1
                if outcome.result is not None:
                    outcome.result.model = f"{self.model} (sync)"
                return outcome.result
            if outcome.fatal:
                # A rejected key will not improve by asking the other
                # endpoint; report the precise reason and stop.
                self.last_error = outcome.result.error if outcome.result else "auth failed"
                return outcome.result
            if not outcome.fall_back:
                # A provider-side failure on the only cheap path. Try async
                # once rather than telling the user their dictation failed;
                # a slow correct answer beats a fast error dialog.
                self.fallbacks_to_async += 1
                self.last_error = outcome.result.error if outcome.result else "sync failed"

        return await self._transcribe_async(started, samples, sample_rate,
                                            keyterms, duration)

    async def _transcribe_async(
        self,
        started: float,
        samples: np.ndarray,
        sample_rate: int,
        keyterms: Optional[List[str]],
        duration: float,
    ) -> TranscriptionResult:
        try:
            client = await self._get_client()
            audio_url = await self._upload(client, float_to_wav_bytes(samples, sample_rate))
            transcript_id = await self._submit(client, audio_url, keyterms)
            payload = await self._poll(client, transcript_id, duration)
            return self._parse(payload, started, duration)

        except httpx.TimeoutException:
            self.last_error = "AssemblyAI timed out."
            return TranscriptionResult.failure(self.last_error, self.model)
        except httpx.HTTPStatusError as e:
            return TranscriptionResult.failure(
                f"AssemblyAI HTTP {e.response.status_code}: {_snippet(e.response)}", self.model
            )
        except httpx.RequestError as e:
            return TranscriptionResult.failure(f"Network error: {e}", self.model)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return TranscriptionResult.failure(self.last_error, self.model)

    async def _upload(self, client: httpx.AsyncClient, wav_bytes: bytes) -> str:
        resp = await client.post(
            "/v2/upload",
            content=wav_bytes,
            headers={"content-type": "application/octet-stream"},
        )
        resp.raise_for_status()
        return resp.json()["upload_url"]

    async def _submit(
        self, client: httpx.AsyncClient, audio_url: str, keyterms: Optional[List[str]]
    ) -> str:
        body = {
            "audio_url": audio_url,
            # `speech_model` (singular) was deprecated in 2026 -- the API now
            # takes a list and picks the first model supporting the request.
            "speech_models": [self.model],
            "language_code": self.language_code,
            "punctuate": True,
            # Server-side casing + ITN (numbers, dates, currency). Removes
            # work from the LLM stage, where it would cost latency and risk.
            "format_text": True,
            # Strip "um"/"uh" at the source rather than asking an LLM to.
            "disfluencies": False,
        }
        if keyterms and self.model in KEYTERM_MODELS:
            body["keyterms_prompt"] = clean_keyterms(
                keyterms, limit=self.keyterms_limit or 1000)

        resp = await client.post("/v2/transcript", json=body)
        resp.raise_for_status()
        return resp.json()["id"]

    async def _poll(
        self, client: httpx.AsyncClient, transcript_id: str, audio_seconds: float = 0.0
    ) -> dict:
        endpoint = f"/v2/transcript/{transcript_id}"
        # Scale the deadline with audio length. AssemblyAI runs far faster
        # than real time, but a fixed timeout would abandon a long
        # dictation the server is still working on.
        budget = max(self.max_wait, audio_seconds * 1.5 + 30.0)
        deadline = time.monotonic() + budget
        # A dictation finishes in a few hundred ms of server time, so a fixed
        # 200 ms first sleep added 200 ms of pure latency to every take
        # (audit C12). Start at tens of ms and back off; the ceiling keeps a
        # long take from turning into a poll storm.
        delay = self.poll_interval
        transient_errors = 0

        while True:
            try:
                resp = await client.get(endpoint)
                if resp.status_code in (429, 500, 502, 503, 504):
                    # One retry with the server's own pacing, then give up and
                    # let the caller fall back / report. Silently looping here
                    # is how a take becomes a 2-minute hang.
                    transient_errors += 1
                    if transient_errors > 3:
                        resp.raise_for_status()
                    await asyncio.sleep(min(
                        float(resp.headers.get("Retry-After") or 0.4) * transient_errors,
                        2.0))
                    continue
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError:
                raise
            except (httpx.TimeoutException, httpx.TransportError):
                transient_errors += 1
                if time.monotonic() > deadline or transient_errors > 4:
                    raise
                await asyncio.sleep(min(0.3 * transient_errors, 1.5))
                continue

            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise RuntimeError(data.get("error", "transcription failed"))
            if time.monotonic() > deadline:
                raise TimeoutError(f"Not ready after {budget:.0f}s")

            await asyncio.sleep(delay)
            delay = min(delay * 1.35, 1.0)

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

        self.last_session_id = data.get("id") or self.last_session_id
        return TranscriptionResult(
            text=(data.get("text") or "").strip(),
            confidence=float(confidence),
            words=words,
            language=data.get("language_code") or self.language_code,
            session_id=data.get("id"),
            model=self.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
            audio_duration_s=duration,
            ok=True,  # empty text here is genuine silence, not an error
        )

    def get_info(self) -> dict:
        return {
            "engine": "AssemblyAI",
            "model": self.model,
            "language": self.language_code,
            "configured": self.is_configured,
            "keyterms_supported": self.model in KEYTERM_MODELS,
            "sync_enabled": self.sync is not None,
            "sync_limit_s": MAX_AUDIO_SECONDS if self.sync is not None else 0,
            "live_upload": self.live_upload_enabled(),
            "sync_hits": self.sync_hits,
            "async_fallbacks": self.fallbacks_to_async,
            "last_error": self.last_error,
            "last_session_id": self.last_session_id,
            "sync_stats": (self.sync.get_stats() if self.sync is not None else None),
        }


def clean_keyterms(terms: List[str], limit: int = 1000, max_words: int = 6) -> List[str]:
    """AssemblyAI caps phrases at 6 words and the list at ~1000 terms."""
    out, seen = [], set()
    for t in terms:
        t = (t or "").strip()
        if not t or len(t.split()) > max_words:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _snippet(resp: httpx.Response, limit: int = 200) -> str:
    try:
        data = resp.json()
        return str(data.get("error") or data)[:limit]
    except Exception:
        return resp.text[:limit]

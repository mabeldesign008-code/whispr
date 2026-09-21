"""AssemblyAI Dictation API client — the app's only transcription path.

One POST per take: audio goes up, verbatim text AND an LLM-cleaned
version come back in the same response. This replaces the old pipeline of
stream/upload -> Groq refiner -> guard; the cleanup that used to be three
moving parts is now one request AssemblyAI runs for us (filler removal,
self-correction resolution, punctuation).

    POST https://dictation.assemblyai.com/v1/transcribe/live
    multipart: config (JSON, FIRST part) + audio (WAV)
    -> { text, llm_response, llm_error, confidence, words[], session_id,
         request_time_ms, audio_duration_ms }

Docs: https://www.assemblyai.com/docs/dictation
      https://www.assemblyai.com/docs/api-reference/dictation-api/transcribe-live

Rules that shaped this code (from the docs):

- `config` must be the FIRST multipart part. Unknown config keys are a
  400, so only documented fields are ever sent.
- The rewrite runs on clips up to 120 s; audio beyond that is a 4xx, so
  we refuse locally instead of paying for a round trip that fails.
- Server-side rewrite is weakest on very short fragments ("yes", "stop
  that"): for clips under ~4 s the docs recommend injecting `text`
  rather than `llm_response`, so `kind` comes back "verbatim" and the
  caller applies the local cleanup instead.
- 429 and 5xx are retryable with backoff; 400/401/403/413/422 are not.
- The response is only guaranteed to contain `text`; `llm_response` can
  be null with `llm_error` set ("timeout"/"error"). The user must never
  see a failure where a slightly-less-clean transcript existed, so that
  case degrades to verbatim + local cleanup, visibly logged.
- Client timeout 90 s: the rewrite itself has a 5 s server deadline, so
  anything beyond this is a connection problem, not a slow model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .base import WordInfo, float_to_wav_bytes

logger = logging.getLogger(__name__)

DICTATE_URL = "https://dictation.assemblyai.com/v1/transcribe/live"
ACCOUNT_URL = "https://api.assemblyai.com/v2/account"

#: Below this the server rewrite is blunt (docs: inject `text` for very
#: short clips), so the caller uses the local cleanup on `text` instead.
MIN_ENHANCE_SECONDS = 4.0

#: Hard API limit. The app's own recording cap sits below this, so
#: exceeding it means something upstream is wrong; fail loudly, locally.
MAX_SECONDS = 120.0

_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 3

#: Field caps from the API reference. Exceeding them is a 400, and a
#: config 400 wastes a whole take, so everything is clamped on the way in.
MAX_KEYTERMS = 100
MAX_KEYTERMS_CHARS = 8000
MAX_INSTRUCTION_CHARS = 2048


@dataclass
class DictationResult:
    """What one Dictation request produced.

    kind:
      "enhanced" -- llm_response is usable; inject `clean`.
      "verbatim" -- too short for the rewrite (or cleanup disabled);
                    inject text via the local cleanup.
      "fallback" -- the server rewrite failed (llm_error set); inject
                    text via the local cleanup and tell the user.
    """

    text: str = ""
    clean: Optional[str] = None
    kind: str = "verbatim"
    llm_error: Optional[str] = None

    confidence: float = 1.0
    words: List[WordInfo] = field(default_factory=list)
    session_id: str = ""
    request_time_ms: float = 0.0
    audio_duration_s: float = 0.0
    latency_ms: int = 0

    ok: bool = True
    error: Optional[str] = None

    @property
    def display_text(self) -> str:
        """The string the caller should inject after its own cleanup decision."""
        return self.text

    def low_confidence_words(self, threshold: float = 0.55) -> List[WordInfo]:
        return [w for w in self.words if w.confidence < threshold]

    @classmethod
    def failure(cls, error: str) -> "DictationResult":
        return cls(ok=False, error=error)


class DictationClient:
    """Async wrapper around the Dictation endpoint with key validation."""

    def __init__(self, api_key: str = "", timeout: float = 90.0):
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self._client = None

        self.requests = 0
        self.fallbacks = 0
        self.last_session_id = ""
        self.last_error = ""
        self.last_request_ms = 0.0

    # ── lifecycle ─────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, key: str) -> None:
        self.api_key = (key or "").strip()
        self._client = None  # headers carry the key; rebuild

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            import httpx

            try:
                client = httpx.AsyncClient(
                    http2=True,
                    timeout=httpx.Timeout(self.timeout, connect=10.0),
                    limits=httpx.Limits(max_keepalive_connections=2,
                                        keepalive_expiry=300),
                )
            except ImportError:  # h2 extra not installed
                client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout, connect=10.0),
                    limits=httpx.Limits(max_keepalive_connections=2,
                                        keepalive_expiry=300),
                )
            self._client = client
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ── key validation + TLS pre-warm ─────────────────────────────────

    async def validate_key(self):
        """(ok, human message). None means 'could not check' (offline)."""
        if not self.api_key:
            return False, "no key"
        try:
            client = await self._get_client()
            resp = await client.get(
                ACCOUNT_URL, headers={"Authorization": self.api_key})
            if resp.status_code == 200:
                return True, "account reachable"
            if resp.status_code in (401, 403):
                return False, "the key was rejected (check it was copied in full)"
            return None, f"unexpected HTTP {resp.status_code}"
        except Exception as e:
            return None, f"could not reach AssemblyAI ({type(e).__name__})"

    async def warm(self) -> None:
        """Complete DNS/TCP/TLS(+H2) to the dictation host ahead of the take.

        The endpoint itself costs nothing to touch -- any status code means
        the expensive part (handshake) is done and pooled for the real POST.
        """
        try:
            client = await self._get_client()
            await client.get("https://dictation.assemblyai.com/")
        except Exception:
            pass

    # ── main entry point ──────────────────────────────────────────────

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        instruction: str = "",
        prompt: str = "",
        keyterms: Optional[List[str]] = None,
        language_codes: Optional[List[str]] = None,
    ) -> DictationResult:
        """Send one take, return verbatim + cleaned text.

        `instruction` (the llm_instruction) REPLACES the server's default
        cleanup prompt -- build it with refine.build_instruction, which
        re-states the cleanup basics and then adds tone/app specifics.
        """
        if not self.is_configured:
            return DictationResult.failure("No AssemblyAI API key — add one in Settings.")

        n = len(samples)
        duration = n / float(sample_rate or 16000)
        if duration > MAX_SECONDS:
            return DictationResult.failure(
                f"Clip is {duration / 60:.1f} min; the Dictation API accepts "
                "at most 2 minutes. Retry with a shorter take.")
        if n < sample_rate // 10:  # < 100 ms cannot be speech
            return DictationResult.failure("Too short.")

        config: Dict[str, Any] = {
            "sample_rate": int(sample_rate),
            "channels": 1,
        }
        if prompt:
            # stt_prompt tunes what the transcriber hears. It REPLACES the
            # managed default prompt, so keep it purely descriptive of the
            # audio and never put field content in it.
            config["stt_prompt"] = prompt[: MAX_KEYTERMS_CHARS]
        if keyterms:
            terms = [t for t in keyterms if t and t.strip()][:MAX_KEYTERMS]
            if terms:
                config["keyterms_prompt"] = terms
        if language_codes:
            config["language_codes"] = language_codes
        if instruction and duration >= MIN_ENHANCE_SECONDS:
            config["llm_instruction"] = instruction[:MAX_INSTRUCTION_CHARS]

        wav = float_to_wav_bytes(samples, sample_rate)
        # config MUST be the first part (API rejects it otherwise).
        files = [
            ("config", (None, json.dumps(config), "application/json")),
            ("audio", ("audio.wav", wav, "audio/wav")),
        ]

        client = await self._get_client()
        headers = {"Authorization": self.api_key}
        started = time.perf_counter()
        resp = None
        delay = 0.4
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = await client.post(DICTATE_URL, headers=headers, files=files)
            except Exception as e:
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                self.last_error = f"network: {type(e).__name__}"
                return DictationResult.failure(
                    f"Could not reach the Dictation API ({type(e).__name__}).")
            if resp.status_code in _RETRYABLE and attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            break

        latency_ms = (time.perf_counter() - started) * 1000
        self.requests += 1

        if resp.status_code != 200:
            detail = ""
            try:
                detail = str(resp.json())[:300]
            except Exception:
                detail = resp.text[:300]
            if resp.status_code in (401, 403):
                msg = "AssemblyAI rejected your API key. Update it in Settings."
            elif resp.status_code == 400:
                msg = f"Request rejected ({detail})"
            elif resp.status_code == 413:
                msg = "Audio too large."
            elif resp.status_code == 422:
                msg = f"Audio unreadable ({detail})"
            else:
                msg = f"HTTP {resp.status_code}: {detail}"
            self.last_error = msg
            return DictationResult.failure(msg)

        try:
            data = resp.json()
        except Exception:
            self.last_error = "invalid JSON in response"
            return DictationResult.failure("The Dictation API returned an unreadable response.")

        return self._parse(data, duration, latency_ms,
                           enhanced_wanted=bool(instruction)
                           and duration >= MIN_ENHANCE_SECONDS)

    # ── response handling ─────────────────────────────────────────────

    def _parse(self, data: Dict[str, Any], duration: float,
               latency_ms: float, enhanced_wanted: bool) -> DictationResult:
        text = (data.get("text") or "").strip()
        llm_response = data.get("llm_response")
        llm_error = data.get("llm_error")
        words = [
            WordInfo(text=str(w.get("text", "")),
                     confidence=float(w.get("confidence", 1.0)))
            for w in (data.get("words") or [])
            if isinstance(w, dict)
        ]

        self.last_session_id = str(data.get("session_id") or "")
        self.last_request_ms = float(data.get("request_time_ms") or 0.0)

        enhanced_wanted = enhanced_wanted and bool(
            duration >= MIN_ENHANCE_SECONDS)
        clean = None
        kind = "verbatim"
        if enhanced_wanted:
            if llm_response:
                clean = str(llm_response).strip() or None
                kind = "enhanced" if clean else "fallback"
            elif llm_error:
                kind = "fallback"
        if kind == "fallback":
            self.fallbacks += 1
            self.last_error = f"rewrite failed ({llm_error}) — used local cleanup"
        else:
            self.last_error = ""

        return DictationResult(
            text=text,
            clean=clean,
            kind=kind,
            llm_error=llm_error,
            confidence=float(data.get("confidence") or 1.0),
            words=words,
            session_id=self.last_session_id,
            request_time_ms=self.last_request_ms,
            audio_duration_s=duration,
            latency_ms=int(latency_ms),
        )

    # ── status for the Settings page ──────────────────────────────────

    def get_info(self) -> dict:
        return {
            "configured": self.is_configured,
            "model": "Dictation API (universal-3-5-pro + cleanup)",
            "requests": self.requests,
            "fallbacks": self.fallbacks,
            "last_session_id": self.last_session_id,
            "last_error": self.last_error,
            "last_request_ms": self.last_request_ms,
        }

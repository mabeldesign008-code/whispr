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
import numpy as np

from .base import TranscriptionResult, WordInfo, float_to_wav_bytes

BASE_URL = "https://api.assemblyai.com"

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
        poll_interval: float = 0.2,
        max_wait: float = 120.0,
    ):
        self.api_key = (api_key or os.getenv("ASSEMBLYAI_API_KEY", "")).strip()
        self.model = model
        self.language_code = language_code
        self.request_timeout = request_timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait

        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, key: str) -> None:
        self.api_key = (key or "").strip()
        self._client = None  # force rebuild with new auth header

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

    # ── transcription ─────────────────────────────────────────────────────

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        keyterms: Optional[List[str]] = None,
    ) -> TranscriptionResult:
        started = time.perf_counter()
        duration = len(samples) / float(sample_rate) if sample_rate else 0.0

        if not self.is_configured:
            return TranscriptionResult.failure(
                "No AssemblyAI API key. Add one in Settings.", self.model
            )
        if samples is None or len(samples) == 0:
            return TranscriptionResult.failure("No audio to transcribe.", self.model)

        try:
            client = await self._get_client()
            audio_url = await self._upload(client, float_to_wav_bytes(samples, sample_rate))
            transcript_id = await self._submit(client, audio_url, keyterms)
            payload = await self._poll(client, transcript_id, duration)
            return self._parse(payload, started, duration)

        except httpx.TimeoutException:
            return TranscriptionResult.failure("AssemblyAI timed out.", self.model)
        except httpx.HTTPStatusError as e:
            return TranscriptionResult.failure(
                f"AssemblyAI HTTP {e.response.status_code}: {_snippet(e.response)}", self.model
            )
        except httpx.RequestError as e:
            return TranscriptionResult.failure(f"Network error: {e}", self.model)
        except Exception as e:
            return TranscriptionResult.failure(
                f"{type(e).__name__}: {e}", self.model
            )

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
            body["keyterms_prompt"] = clean_keyterms(keyterms)

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
        delay = self.poll_interval

        while True:
            resp = await client.get(endpoint)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status")
            if status == "completed":
                return data
            if status == "error":
                raise RuntimeError(data.get("error", "transcription failed"))
            if time.monotonic() > deadline:
                raise TimeoutError(f"Not ready after {budget:.0f}s")

            await asyncio.sleep(delay)
            # Dictation clips finish fast: poll tightly, then ease off.
            delay = min(delay * 1.3, 1.0)

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

        return TranscriptionResult(
            text=(data.get("text") or "").strip(),
            confidence=float(confidence),
            words=words,
            language=data.get("language_code") or self.language_code,
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

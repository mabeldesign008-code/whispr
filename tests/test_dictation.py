"""DictationClient against a mocked Dictation API.

Covers the request contract (multipart order, field caps, instruction
gating by clip length) and the response contract (enhanced / verbatim /
fallback kinds, retries, errors).

The API's own rule -- "config is the FIRST multipart part; unknown fields
are a 400" -- is exactly the kind of thing that breaks silently after a
refactor, hence the ordering assertion in _last_config().
"""

import json
import re

import httpx
import numpy as np
import pytest

from stt.dictation import (
    DICTATE_URL,
    MAX_SECONDS,
    MIN_ENHANCE_SECONDS,
    DictationClient,
)

SR = 16000


def speech(seconds: float, rate: int = SR) -> np.ndarray:
    """Any non-constant signal would do; a sine keeps float_to_wav happy."""
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def ok_response(**over):
    data = {
        "text": "send this this to john",
        "llm_response": "Send this to John.",
        "llm_error": None,
        "confidence": 0.97,
        "words": [{"text": "send", "confidence": 0.99},
                  {"text": "this", "confidence": 0.4}],
        "session_id": "sess-123",
        "request_time_ms": 250.0,
        "audio_duration_ms": 2924,
    }
    data.update(over)
    return data


def _config_of(request: httpx.Request) -> dict:
    body = request.read()
    # config must be the first part; a regression there is a 400 at the API.
    i_cfg = body.find(b'name="config"')
    i_aud = body.find(b'name="audio"')
    assert 0 <= i_cfg < i_aud, "config part missing or not first"
    m = re.search(rb'name="config"\r\nContent-Type: application/json\r\n\r\n(\{.*?\})\r\n--', body, re.S)
    assert m, "config JSON not found in multipart body"
    return json.loads(m.group(1))


def make_client(handler) -> DictationClient:
    client = DictationClient(api_key="test-key")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=5.0,
    )
    return client


# ── the happy path ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_enhanced_take_returns_verbatim_and_clean():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL(DICTATE_URL)
        assert request.headers["Authorization"] == "test-key"
        assert "multipart/form-data" in request.headers["content-type"]
        seen["config"] = _config_of(request)
        return httpx.Response(200, json=ok_response())

    client = make_client(handler)
    res = await client.transcribe(
        speech(8), SR,
        instruction="Clean up this dictated text: remove filler words...",
        prompt="A single speaker dictating",
        keyterms=["WhisprFlow", "Kojo"],
    )

    assert res.ok
    assert res.kind == "enhanced"
    assert res.clean == "Send this to John."
    assert res.text == "send this this to john"
    assert res.session_id == "sess-123"
    assert res.request_time_ms == 250.0
    assert 0.9 < res.audio_duration_s < 8.1
    assert res.latency_ms >= 0

    cfg = seen["config"]
    assert cfg["sample_rate"] == SR
    assert cfg["channels"] == 1
    assert cfg["stt_prompt"] == "A single speaker dictating"
    assert cfg["keyterms_prompt"] == ["WhisprFlow", "Kojo"]
    assert cfg["llm_instruction"].startswith("Clean up")
    # nothing beyond the documented fields leaves the building
    assert set(cfg) <= {"sample_rate", "channels", "stt_prompt",
                        "keyterms_prompt", "llm_instruction", "language_codes"}

    low = res.low_confidence_words()
    assert [w.text for w in low] == ["this"]


@pytest.mark.asyncio
async def test_short_clip_gets_no_instruction_and_stays_verbatim():
    """Docs: the server rewrite is blunt on very short clips; the caller
    must inject `text` (kind 'verbatim') rather than llm_response."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["config"] = _config_of(request)
        return httpx.Response(200, json=ok_response())

    client = make_client(handler)
    res = await client.transcribe(
        speech(MIN_ENHANCE_SECONDS - 1), SR,
        instruction="Clean up this dictated text...")

    assert res.ok and res.kind == "verbatim"
    assert res.clean is None
    assert "llm_instruction" not in seen["config"]


@pytest.mark.asyncio
async def test_no_cleanup_means_no_instruction_is_sent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["config"] = _config_of(request)
        return httpx.Response(200, json=ok_response(llm_response=None))

    client = make_client(handler)
    res = await client.transcribe(speech(8), SR, instruction="")
    assert res.ok and res.kind == "verbatim"
    assert "llm_instruction" not in seen["config"]


# ── degradation contract ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_error_degrades_to_fallback_not_failure():
    """The one contract we refuse to lose: a slow rewrite must never look
    like a failed dictation when a verbatim transcript exists."""
    def handler(request):
        return httpx.Response(200, json=ok_response(
            llm_response=None, llm_error="timeout"))

    client = make_client(handler)
    res = await client.transcribe(
        speech(8), SR, instruction="clean")
    assert res.ok
    assert res.kind == "fallback"
    assert res.clean is None
    assert res.llm_error == "timeout"
    assert client.get_info()["fallbacks"] == 1
    assert "rewrite failed" in client.get_info()["last_error"]


# ── errors and retries ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_400_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": "unknown field llm_instrucion"})

    client = make_client(handler)
    res = await client.transcribe(speech(8), SR)
    assert not res.ok
    assert "unknown field" in res.error
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_401_says_update_the_key():
    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    client = make_client(handler)
    res = await client.transcribe(speech(8), SR)
    assert not res.ok
    assert "API key" in res.error


@pytest.mark.asyncio
async def test_429_is_retried_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=ok_response())

    client = make_client(handler)
    res = await client.transcribe(speech(8), SR, instruction="clean")
    assert res.ok and res.kind == "enhanced"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_500_eventually_fails_cleanly():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    client = make_client(handler)
    res = await client.transcribe(speech(8), SR)
    assert not res.ok
    assert "500" in res.error


# ── local gates ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_over_two_minutes_refused_without_http():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=ok_response())

    client = make_client(handler)
    big = speech(MAX_SECONDS + 5)
    res = await client.transcribe(big, SR)
    assert not res.ok
    assert "2 minutes" in res.error
    assert calls == [], "a locally-detectable 4xx must not cost a request"


@pytest.mark.asyncio
async def test_unconfigured_client_fails_fast():
    client = DictationClient(api_key="")
    res = await client.transcribe(speech(8), SR)
    assert not res.ok
    assert "API key" in res.error


@pytest.mark.asyncio
async def test_field_caps_are_clamped():
    seen = {}

    def handler(request):
        seen["config"] = _config_of(request)
        return httpx.Response(200, json=ok_response())

    client = make_client(handler)
    await client.transcribe(
        speech(8), SR,
        instruction="x" * 5000,
        keyterms=[f"term{i}" for i in range(200)],
    )
    assert len(seen["config"]["llm_instruction"]) <= 2048
    assert len(seen["config"]["keyterms_prompt"]) <= 100


# ── key validation ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_validate_key():
    def handler(request):
        assert request.url.path == "/v2/account"
        assert request.headers["Authorization"] == "test-key"
        return httpx.Response(200, json={})

    client = make_client(handler)
    ok, msg = await client.validate_key()
    assert ok is True


@pytest.mark.asyncio
async def test_validate_key_rejected():
    def handler(request):
        return httpx.Response(401, json={})

    client = make_client(handler)
    ok, msg = await client.validate_key()
    assert ok is False


@pytest.mark.asyncio
async def test_warm_never_raises():
    def handler(request):
        return httpx.Response(404)

    client = make_client(handler)
    await client.warm()  # any status fine; the TLS handshake was the point
    await client.close()

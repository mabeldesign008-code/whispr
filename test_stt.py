"""Tests for the STT layer and the refinement guard."""

import asyncio
import io
import wave

import numpy as np
import pytest

from stt.base import TranscriptionResult, WordInfo, float_to_wav_bytes
from stt.assemblyai_client import AssemblyAIClient, clean_keyterms
from stt.dictionary import UserDictionary
from refine.guard import basic_cleanup, check, levenshtein
from refine.refiner import Refiner, _strip_wrapping


# ── WAV encoding ──────────────────────────────────────────────────────────

class TestWavEncoding:
    def test_roundtrip(self):
        sr = 16000
        tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        with wave.open(io.BytesIO(float_to_wav_bytes(tone, sr)), "rb") as wf:
            assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, sr)
            assert wf.getnframes() == sr

    def test_clips_instead_of_wrapping(self):
        """np.int16(x*32767) wraps on overflow -> loud noise bursts."""
        data = float_to_wav_bytes(np.array([3.0, -3.0, 0.0], dtype=np.float32), 16000)
        with wave.open(io.BytesIO(data), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(3), dtype=np.int16)
        assert pcm[0] == 32767 and pcm[1] == -32767 and pcm[2] == 0

    def test_empty(self):
        assert len(float_to_wav_bytes(np.array([], dtype=np.float32), 16000)) > 0


# ── result type ───────────────────────────────────────────────────────────

class TestResult:
    def test_failure_is_not_ok(self):
        r = TranscriptionResult.failure("boom", "m")
        assert not r.ok and r.error == "boom" and r.is_empty

    def test_silence_distinct_from_failure(self):
        """The old code conflated these and discarded user speech."""
        assert TranscriptionResult(text="", ok=True).ok
        assert not TranscriptionResult.failure("network down").ok

    def test_low_confidence_words(self):
        r = TranscriptionResult(words=[
            WordInfo("deploy", 0.99), WordInfo("cuban", 0.31), WordInfo("artists", 0.29),
        ])
        assert [w.text for w in r.low_confidence_words()] == ["cuban", "artists"]


# ── AssemblyAI ────────────────────────────────────────────────────────────

class TestKeyterms:
    def test_drops_long_phrases(self):
        assert clean_keyterms(["ok", "one two three four five six seven"]) == ["ok"]

    def test_dedupes_case_insensitively(self):
        assert clean_keyterms(["Groq", "groq", "GROQ"]) == ["Groq"]

    def test_respects_limit(self):
        assert len(clean_keyterms([f"t{i}" for i in range(50)], limit=10)) == 10


class TestAssemblyAIClient:
    def test_unconfigured_fails_loudly(self):
        r = asyncio.run(AssemblyAIClient(api_key="").transcribe(np.zeros(1600, np.float32), 16000))
        assert not r.ok and "key" in r.error.lower()

    def test_empty_audio_fails(self):
        r = asyncio.run(AssemblyAIClient(api_key="k").transcribe(np.array([], np.float32), 16000))
        assert not r.ok

    def test_parse_words_and_confidence(self):
        r = AssemblyAIClient(api_key="k")._parse({
            "text": "Hello world", "confidence": 0.94, "language_code": "en",
            "words": [{"text": "Hello", "confidence": 0.99, "start": 0, "end": 400},
                      {"text": "world", "confidence": 0.89, "start": 410, "end": 800}],
        }, started=0.0, duration=1.0)
        assert r.ok and r.text == "Hello world" and r.words[1].confidence == 0.89

    def test_parse_empty_is_ok(self):
        r = AssemblyAIClient(api_key="k")._parse({"text": "", "words": []}, 0.0, 1.0)
        assert r.ok and r.is_empty

    def test_keyterms_gated_by_model(self):
        assert AssemblyAIClient(model="universal-3-5-pro").get_info()["keyterms_supported"]
        assert not AssemblyAIClient(model="universal-2").get_info()["keyterms_supported"]

    def test_set_api_key_resets_client(self):
        c = AssemblyAIClient(api_key="old")
        c._client = object()
        c.set_api_key("new")
        assert c.api_key == "new" and c._client is None


# ── refinement guard ──────────────────────────────────────────────────────

class TestGuard:
    DICT = ["Kubernetes"]

    def test_accepts_punctuation_fix(self):
        assert check("i cant make it", "I can't make it.")

    def test_rejects_negation_flip(self):
        """The single worst failure mode: silently inverts meaning."""
        v = check("i cant make the meeting", "I can make the meeting.")
        assert not v and "negation" in v.reason

    def test_rejects_negation_removal(self):
        assert not check("we should not deploy this", "We should deploy this.")

    def test_allows_negation_paraphrase(self):
        assert check("we cannot ship today", "We are unable to ship today.",
                     allow_restructure=True)

    def test_rejects_changed_number(self):
        v = check("send 50 dollars", "Send $15.")
        assert not v and "number" in v.reason

    def test_allows_digit_to_word(self):
        assert check("meet at 3", "Meet at three.")

    def test_rejects_dropped_dictionary_term(self):
        v = check("deploy the kubernetes cluster", "Deploy the Cuban artists cluster.",
                  dictionary_terms=self.DICT)
        assert not v and "dictionary" in v.reason

    def test_rejects_invented_content(self):
        assert not check("hello", "Hello there my friend how are you doing today ok")

    def test_rejects_wholesale_rewrite(self):
        assert not check("the quick brown fox jumps over",
                         "Something completely different entirely now here")

    def test_short_text_not_penalised_by_ratio(self):
        """A 3-word phrase hits 50% edit ratio from one legitimate change."""
        assert check("send fifty dollars", "Send $50.")

    def test_articulate_allows_restructure(self):
        assert check("um so basically the thing is that it is broken now",
                     "The thing is broken.", allow_restructure=True)

    def test_articulate_still_blocks_negation_flip(self):
        assert not check("we cannot ship", "We can ship.", allow_restructure=True)

    def test_rejects_empty(self):
        assert not check("hello world", "")

    def test_levenshtein(self):
        assert levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0
        assert levenshtein(["a", "b"], ["a", "x"]) == 1


class TestBasicCleanup:
    def test_capitalises_and_punctuates(self):
        assert basic_cleanup("hello world") == "Hello world."

    def test_preserves_existing_punctuation(self):
        assert basic_cleanup("Is it done?") == "Is it done?"

    def test_never_changes_words(self):
        assert basic_cleanup("i cant do it") == "I cant do it."

    def test_empty(self):
        assert basic_cleanup("") == "" and basic_cleanup("   ") == ""


class TestStripWrapping:
    def test_removes_preamble(self):
        assert _strip_wrapping("Here is the corrected text: Hello.") == "Hello."

    def test_removes_quotes(self):
        assert _strip_wrapping('"Hello there."') == "Hello there."

    def test_removes_code_fence(self):
        assert _strip_wrapping("```\nHello\n```") == "Hello"

    def test_leaves_clean_text(self):
        assert _strip_wrapping("Hello there.") == "Hello there."


class TestRefiner:
    def test_unconfigured_uses_basic_cleanup(self):
        r = Refiner(api_key="")
        assert asyncio.run(r.refine("hello world")) == "Hello world."

    def test_empty_passthrough(self):
        assert asyncio.run(Refiner(api_key="").refine("")) == ""

    def test_builds_prompt_with_evidence(self):
        r = Refiner(api_key="k")
        out = r._build_input("deploy it", ["cuban"], ["Kubernetes"], "Code.exe")
        assert "APP: Code.exe" in out
        assert "KNOWN TERMS: Kubernetes" in out
        assert "UNCERTAIN: cuban" in out
        assert "TEXT:\ndeploy it" in out

    def test_set_api_key_resets_client(self):
        r = Refiner(api_key="old")
        r._client = object()
        r.set_api_key("new")
        assert r.api_key == "new" and r._client is None


class TestModelFallback:
    """Audit C1/C2: a retired model id used to fail every call in silence."""

    def test_dead_model_id_falls_back_and_is_remembered(self):
        from refine.api_health import ApiHealth
        from refine.refiner import Refiner

        health = ApiHealth()

        def dead(body):
            import httpx
            return httpx.Response(400, json={
                "error": {"message": f"The model `{body['model']}` is not a valid model"}})

        def good(body):
            import httpx
            return httpx.Response(200, json={"choices": [
                {"message": {"content": "Deploy it."}, "finish_reason": "stop"}]})

        import refine.refiner as R
        orig_health = R.health
        try:
            R.health = health
            r = Refiner(api_key="k")
            r.models = ["brand-new-id", "second-id"]
            calls = []

            import httpx as _h
            import json as _j

            def handler(request):
                body = _j.loads(request.content)
                calls.append(body["model"])
                return dead(body) if body["model"] == "brand-new-id" else good(body)

            r._get_client = _client(_h.AsyncClient(
                transport=_h.MockTransport(handler), timeout=1.0))
            R.API_URL = "http://fake.test/v1/chat/completions"

            out = asyncio.run(r.refine("deploy it"))
            assert out == "Deploy it."
            assert calls == ["brand-new-id", "second-id"]
            assert r.last_model_used == "second-id"
            assert health.is_dead("refiner", "brand-new-id")
            # The dead id is tried last on the *next* call, not dropped.
            assert health.ordered("refiner", r.models) == ["second-id", "brand-new-id"]
        finally:
            R.health = orig_health

    def test_truncated_completion_is_refused(self):
        """finish_reason=length must not inject a half sentence (A3)."""
        import httpx as _h

        import refine.refiner as R
        orig = R.health
        calls = []

        def handler(request):
            calls.append(1)
            return _h.Response(200, json={"choices": [
                {"message": {"content": "The quarterly report for Q3 shows a clear rise in"},
                 "finish_reason": "length"}]})

        from refine.refiner import Refiner
        r = Refiner(api_key="k")
        r._get_client = _client(_h.AsyncClient(
            transport=_h.MockTransport(handler), timeout=1.0))
        R.API_URL = "http://fake.test/v1/chat/completions"
        try:
            out = asyncio.run(r.refine(
                "the quarterly report for q3 shows a clear rise in revenue and margin"))
            assert "revenue" in out, "raw words must survive when the LLM truncates"
            assert not out.endswith(" in"), out
            assert len(calls) == 1, "a truncated answer is not retried on another model"
            assert "truncated" in (r.last_api_error or "")
        finally:
            R.health = orig

    def test_http_error_is_recorded_not_swallowed(self):
        import httpx as _h

        import refine.refiner as R
        from refine.refiner import Refiner
        orig = R.health
        try:
            R.health = __import__("refine.api_health", fromlist=["ApiHealth"]).ApiHealth()

            def handler(request):
                return _h.Response(401, json={"error": {"message": "invalid API key"}})

            r = Refiner(api_key="k")
            r._get_client = _client(_h.AsyncClient(
                transport=_h.MockTransport(handler), timeout=1.0))
            R.API_URL = "http://fake.test/v1/chat/completions"
            out = asyncio.run(r.refine("ship the release"))
            assert out == "Ship the release."          # text is still delivered
            assert r.api_failures == 1
            assert "invalid API key" in (r.last_api_error or "")
            assert "401" in (R.health.summary() or "")
        finally:
            R.health = orig


def _client(client):
    """Stand in for Refiner._get_client, handing it a MockTransport client."""
    async def _get():
        return client
    return _get


# ── dictionary ────────────────────────────────────────────────────────────

class TestDictionary:
    def test_add_and_persist(self, tmp_path):
        p = tmp_path / "d.txt"
        d = UserDictionary(p)
        assert d.add("Kubernetes") and d.add("WhisprFlow")
        assert UserDictionary(p).terms == ["Kubernetes", "WhisprFlow"]

    def test_rejects_duplicates(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        assert d.add("Groq") and not d.add("groq") and len(d) == 1

    def test_rejects_overlong(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        assert not d.add("one two three four five six seven")
        assert not d.add("x" * 60)

    def test_comments_ignored(self, tmp_path):
        p = tmp_path / "d.txt"
        p.write_text("# c\nAlpha\n\nBeta\n", encoding="utf-8")
        assert UserDictionary(p).terms == ["Alpha", "Beta"]

    def test_remove(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        d.add("Temp")
        assert d.remove("temp") and len(d) == 0

    def test_prompt_line(self, tmp_path):
        d = UserDictionary(tmp_path / "d.txt")
        d.add_many(["Groq", "Parakeet"])
        assert d.as_prompt_line() == "Groq, Parakeet"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

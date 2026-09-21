"""The smart-formatting pass and its word-for-word guarantee.

The guarantee is the whole feature: OUTPUT = INPUT words, plus layout.
Everything else -- speed, model choice, fallbacks -- exists to protect it.
"""

import httpx
import pytest

from refine.formatter import (
    DEFAULT_MODEL,
    MIN_WORDS,
    Formatter,
    words_preserved,
)


CLEAN = ("I wanted to follow up on the proposal we discussed on Tuesday and "
         "also confirm the three things you asked for. First, the pricing "
         "sheet goes out tomorrow morning. Second, the contract draft is "
         "with legal now. Third, I will call the supplier on Friday to "
         "confirm the delivery date for the Accra shipment.")

FORMATTED = ("I wanted to follow up on the proposal we discussed on Tuesday "
             "and also confirm the three things you asked for.\n\n"
             "1. First, the pricing sheet goes out tomorrow morning.\n"
             "2. Second, the contract draft is with legal now.\n"
             "3. Third, I will call the supplier on Friday to confirm the "
             "delivery date for the Accra shipment.")


# ── the verifier ─────────────────────────────────────────────────────────

class TestWordsPreserved:
    def test_paragraph_breaks_pass(self):
        a = "Hello world. This is fine."
        b = "Hello world.\n\nThis is fine."
        assert words_preserved(a, b)

    def test_list_markers_pass(self):
        assert words_preserved(CLEAN, FORMATTED)

    def test_numbering_markers_pass(self):
        a = "Do alpha then beta then gamma."
        b = "Do:\n\n1. alpha\n2. then beta\n3. then gamma."
        # '1.'/'2.' markers are layout; the word stream must match
        assert _streams(a) == _streams(b)

    def test_a_changed_word_fails(self):
        a = "The meeting is on Tuesday."
        b = "The meeting is on Wednesday."
        assert not words_preserved(a, b)

    def test_a_dropped_word_fails(self):
        assert not words_preserved("I really do think so", "I do think so")

    def test_an_added_word_fails(self):
        assert not words_preserved("Meeting moved to Friday",
                                   "Hi, meeting moved to Friday. Best regards")

    def test_casing_change_is_not_a_word_change(self):
        # 'friday' -> 'Friday' is fine (it was already the cleanup's call).
        assert words_preserved("see you friday", "See you Friday")

    def test_bullet_symbols_all_accepted(self):
        for m in ("- ", "* ", "• ", "1. ", "12) "):
            assert words_preserved("buy milk and eggs", f"buy:\n{m}milk and eggs")


def _streams(t):
    from refine.formatter import _word_stream
    return _word_stream(t)


# ── the formatter contract ────────────────────────────────────────────────

def make_formatter(handler) -> Formatter:
    f = Formatter(api_key="k")
    f._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=5.0)
    return f


def groq_response(content: str, model: str = DEFAULT_MODEL):
    return {"choices": [{"message": {"content": content},
                         "finish_reason": "stop"}]}


@pytest.mark.asyncio
async def test_applies_a_valid_layout():
    def handler(request):
        import json
        body = json.loads(request.read())
        # the request spec that must never regress (audit §2)
        assert body["reasoning_effort"] == "low"
        assert body["include_reasoning"] is False
        assert body["max_completion_tokens"] >= 2048
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert "NEVER change" in body["messages"][0]["content"]
        return httpx.Response(200, json=groq_response(FORMATTED))

    f = make_formatter(handler)
    res = await f.apply(CLEAN)
    assert res.ok and res.changed
    assert res.text == FORMATTED
    assert f.get_stats()["changed"] == 1


@pytest.mark.asyncio
async def test_word_change_is_rejected_and_input_kept():
    sabotaged = FORMATTED.replace("Tuesday", "Thursday")

    def handler(request):
        return httpx.Response(200, json=groq_response(sabotaged))

    f = make_formatter(handler)
    res = await f.apply(CLEAN)
    assert not res.ok
    assert res.text == CLEAN  # fail-safe: identical to input
    assert "words" in res.error


@pytest.mark.asyncio
async def test_http_error_keeps_input():
    def handler(request):
        return httpx.Response(500, json={"error": {"message": "boom"}})

    f = make_formatter(handler)
    res = await f.apply(CLEAN)
    assert not res.ok and res.text == CLEAN
    assert f.get_stats()["failures"] == 1


@pytest.mark.asyncio
async def test_short_text_skips_the_network():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=groq_response("x"))

    f = make_formatter(handler)
    res = await f.apply("See you at three.")
    assert res.ok and not res.changed
    assert calls == [], "short takes must not pay for a formatting call"


@pytest.mark.asyncio
async def test_unconfigured_keeps_input():
    f = Formatter(api_key="")
    res = await f.apply(CLEAN)
    assert not res.ok and res.text == CLEAN


@pytest.mark.asyncio
async def test_retired_model_falls_through_to_next():
    models_seen = []

    def handler(request):
        import json
        model = json.loads(request.read())["model"]
        models_seen.append(model)
        if len(models_seen) == 1:
            return httpx.Response(400, json={"error": {
                "message": "The model `x` has been decommissioned"}})
        return httpx.Response(200, json=groq_response(CLEAN))

    f = make_formatter(handler)
    res = await f.apply(CLEAN)
    assert res.ok
    assert len(models_seen) == 2 and models_seen[0] != models_seen[1]


@pytest.mark.asyncio
async def test_worth_formatting_threshold():
    f = Formatter(api_key="k")
    assert not f.worth_formatting("one two three")
    assert f.worth_formatting(" ".join(["word"] * MIN_WORDS))

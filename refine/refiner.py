"""Builds the `llm_instruction` sent with each Dictation API request.

This module is what remains of the old local refiner (Groq LLM + guard).
The cleanup itself now happens on AssemblyAI's side of the same POST that
transcribes — see stt/dictation.py — so the only job left here is to say
*how* we want the text cleaned, per take:

    base cleanup rules  +  one of three tones  +  app-specific line

Why the base rules are re-stated explicitly: `llm_instruction` REPLACES
the server's managed default cleanup prompt, it does not extend it. Any
behaviour we still want (filler removal, self-correction resolution,
punctuation) has to appear in our instruction or it will stop happening.

Guards worth keeping from the old design, carried over:

- Keep the speaker's words. The old Groq refiner was instructed to light-
  paraphrase and then needed an entirely separate guard module to veto
  meaning changes; it still false-rejected 42% of takes in the audit.
  Telling the rewrite to preserve sentence structure up front removes the
  failure mode instead of policing it after the fact.
- Never invent greetings/sign-offs — dictation LLMs love to add
  "Best regards" that the user never said; the audit caught this in eval.
- No pleasantries the user didn't speak, in every tone.

Also kept: `basic_cleanup`, the deterministic local fallback for when the
server's rewrite is unavailable (short clips) or failed (llm_error).
"""

from __future__ import annotations

import re

# ── instruction building ──────────────────────────────────────────────────

BASE_INSTRUCTION = (
    "Clean up this dictated text: remove filler words ('um', 'uh', er, ah, "
    "'like', 'you know', 'I mean' when used as fillers), resolve "
    "self-corrections (when the speaker restarts a phrase, keep only the "
    "final wording), and apply correct punctuation, capitalisation and "
    "written spacing. Convert spoken numbers, emails and URLs to their "
    "written form. Keep the speaker's exact wording and sentence "
    "structure — paraphrase only where grammar requires it. Never "
    "summarise, translate, answer questions, or add content the speaker "
    "did not say — including greetings, sign-offs and pleasantries. If "
    "the audio is just noise or silence, reply in kind."
)

TONES = {
    "general": "",
    "casual": (
        " Tone: relaxed chat — contractions and informal wording are fine; "
        "keep it brief and do not add greetings or sign-offs."
    ),
    "formal": (
        " Tone: professional — complete sentences and standard written "
        "punctuation; keep every point the speaker made; no added "
        "greetings, sign-offs or pleasantries."
    ),
}

#: UI label per tone id (order matters for the dropdown).
TONE_NAMES = {
    "general": "General",
    "casual": "Casual",
    "formal": "Formal",
}

#: Settings written by older builds (and one that only ever existed with a
#: typo in the repo) map onto the new set.
LEGACY_TONE_ALIASES = {
    "auto": "general",
    "chat_acro-fromal": "casual",
    "chat_acro-formal": "casual",
    "polished": "formal",
    "professional": "formal",
    "restructure": "formal",
}


def default_tone(value: str) -> str:
    """Normalise any stored/env tone to a valid id."""
    t = (value or "").strip().lower()
    t = LEGACY_TONE_ALIASES.get(t, t)
    return t if t in TONES else "general"


def build_instruction(tone: str = "general", app_instruction: str = "") -> str:
    """The complete llm_instruction for one take.

    AssemblyAI caps the field at ~2048 chars and returns a 400 over the
    limit, so clamp rather than hope. App-specific profile lines are short
    by contract (one sentence), so overflow is not expected in practice.
    """
    t = default_tone(tone)
    out = BASE_INSTRUCTION + TONES[t]
    extra = (app_instruction or "").strip()
    if extra:
        out += " " + extra
    return out[:2048]


# ── deterministic local cleanup ───────────────────────────────────────────


def basic_cleanup(text: str) -> str:
    """Deterministic fallback when the server rewrite is unavailable.

    Capitalise, ensure terminal punctuation. Never changes words.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return t
    first = t.split()[0]
    # iPhone / macOS / eBay / PayPal keep their internal capitals: the
    # brand's own casing is the correct casing, and raising the first letter
    # of such a word is a visible mistake in every paste.
    if not any(c.isupper() for c in first[1:]):
        t = t[0].upper() + t[1:]

    if t[-1] in ",;:":
        # Measured: basic_cleanup("called John,") produced "Called John,." A
        # dangling comma is not punctuation to keep -- replace it.
        t = t[:-1] + "."
    elif t[-1] in "-\u2013\u2014":
        t = t[:-1].rstrip()
    elif t[-1] not in ".!?\"'\u201d\u2019)]}":
        t += "."
    return t

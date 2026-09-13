"""
Safety guard for LLM text refinement.

Audit finding 1.12: the LLM corrector receives only the 1-best string --
no audio, no confidences, no dictionary -- and is told to "fix misheard
words". That converts *recognition* errors (visible, obviously wrong) into
*fluent* errors (invisible, plausibly wrong). Users trust fluent text and
ship it.

The old code tried to prevent this with prompt rules ("NEVER flip
negations"). A prompt rule is a hope, not an invariant. This module makes
it mechanical: if a rewrite looks like a hallucination, we reject it and
keep the raw transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

# Flipping any of these inverts meaning. The single worst failure mode.
# Includes non-contraction equivalents ("unable", "impossible") so that a
# legitimate paraphrase -- "cannot ship" -> "unable to ship" -- keeps the
# count balanced, while "cannot ship" -> "can ship" does not.
NEGATIONS: Set[str] = {
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "cannot", "cant", "wont", "dont", "doesnt", "didnt",
    "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
    "shouldnt", "wouldnt", "couldnt", "aint", "without",
    "unable", "impossible", "unavailable", "declined", "refused",
}

# Below this many words the edit ratio is meaningless -- a 2-word phrase
# hits 50% from a single legitimate substitution. Short text is checked by
# the meaning-preserving rules (negation, numbers, dictionary) only.
MIN_WORDS_FOR_RATIO = 6

_WORD_RE = re.compile(r"[a-z0-9']+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


@dataclass
class GuardVerdict:
    accepted: bool
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.accepted


def _words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _negation_count(text: str) -> int:
    """Count negations, normalising apostrophes so can't == cant."""
    return sum(1 for w in _words(text.replace("'", "")) if w in NEGATIONS)


def _numbers(text: str) -> List[str]:
    """Digit strings, normalised so 3,000 == 3000."""
    return [n.replace(",", "").rstrip(".") for n in _NUM_RE.findall(text or "")]


def levenshtein(a: List[str], b: List[str]) -> int:
    """Word-level edit distance."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] if x == y else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[len(b)]


def check(
    original: str,
    refined: str,
    max_edit_ratio: float = 0.45,
    dictionary_terms: Optional[List[str]] = None,
    allow_restructure: bool = False,
) -> GuardVerdict:
    """Decide whether a refinement is safe to inject.

    Args:
        original: raw transcript from AssemblyAI
        refined: the LLM's rewrite
        max_edit_ratio: reject if more than this fraction of words changed
        dictionary_terms: user dictionary; dropping one is a red flag
        allow_restructure: True in Articulate mode, which is *meant* to
            rewrite heavily -- so the edit-distance check is skipped, but
            the meaning-inverting checks still apply.
    """
    if not refined or not refined.strip():
        return GuardVerdict(False, "empty refinement")

    orig_words, ref_words = _words(original), _words(refined)
    if not orig_words:
        return GuardVerdict(True)

    # 1. Negation flip -- inverts meaning. Never acceptable, any mode.
    if _negation_count(original) != _negation_count(refined):
        return GuardVerdict(False, "negation added or removed")

    # 2. Numbers changed. "$50" -> "$15" is catastrophic and invisible.
    #    Additions are fine (ITN spelling out "twenty" as "20").
    orig_nums, ref_nums = set(_numbers(original)), set(_numbers(refined))
    dropped = orig_nums - ref_nums
    if dropped and not _explained_by_wordform(dropped, original, refined):
        return GuardVerdict(False, f"number changed or dropped: {sorted(dropped)}")

    # 3. Dictionary term dropped -- the LLM "corrected" a product name into
    #    an ordinary English word, exactly what the dictionary exists to stop.
    if dictionary_terms:
        low_ref = refined.lower()
        for term in dictionary_terms:
            t = term.lower()
            if t in original.lower() and t not in low_ref:
                return GuardVerdict(False, f"dictionary term dropped: {term!r}")

    # 4. Content invented or lost wholesale. Checked at every length --
    #    "hello" -> a 14-word sentence is a hallucination regardless of mode.
    if len(ref_words) > len(orig_words) * 1.8 + 3:
        return GuardVerdict(False, "refinement much longer than original")
    if not allow_restructure and len(ref_words) < len(orig_words) * 0.4:
        return GuardVerdict(False, "refinement much shorter than original")

    # 5. Wholesale rewrite. Standard mode should correct, not reinvent.
    #    Skipped on very short text where the ratio is statistically noise.
    if not allow_restructure and len(orig_words) >= MIN_WORDS_FOR_RATIO:
        distance = levenshtein(orig_words, ref_words)
        ratio = distance / len(orig_words)
        if ratio > max_edit_ratio:
            return GuardVerdict(False, f"rewrote {ratio:.0%} of words (limit {max_edit_ratio:.0%})")

    return GuardVerdict(True)


def _explained_by_wordform(dropped: Set[str], original: str, refined: str) -> bool:
    """Allow a digit to vanish if it reappears as a word ("3" -> "three")."""
    spelled = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve",
    }
    low = refined.lower()
    return all(n in spelled and spelled[n] in low for n in dropped)


def basic_cleanup(text: str) -> str:
    """Deterministic fallback when refinement is rejected or unavailable.
    Capitalise, ensure terminal punctuation. Never changes words."""
    t = (text or "").strip()
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?":
        t += "."
    return t

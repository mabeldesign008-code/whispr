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
from difflib import SequenceMatcher
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
# Standalone quantities only. The bare \d+ pattern matched the 42 in
# "word42" and the 1 in "v1.2.3", so an identifier being re-flowed looked
# like a changed number (audit A3, rule 4) -- and, worse, made the rule too
# noisy to extend in the direction that mattered.
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w.])")

# Pairs where swapping one member for the other inverts the meaning without
# touching a negation, a digit or a dictionary term -- the exact class the
# audit measured as invisible (delaying->proceeding, increase->decrease,
# Alice->Bob). Only the halves that a dictation could plausibly contain; a
# long list here costs false rejections, and a rejection is cheap (the raw
# transcript is kept) but not free.
OPPOSITES: List[tuple] = [
    ("increase", "decrease"), ("raise", "lower"), ("higher", "lower"),
    ("up", "down"), ("more", "less"), ("added", "removed"),
    ("accept", "reject"), ("approve", "decline"), ("allow", "block"),
    ("enable", "disable"), ("start", "stop"), ("begin", "end"),
    ("before", "after"), ("early", "late"), ("first", "last"),
    ("previous", "next"), ("incoming", "outgoing"), ("delaying", "advancing"),
    ("recommend", "deter"), ("praise", "criticize"), ("support", "oppose"),
    ("success", "failure"), ("win", "lose"), ("gains", "losses"),
    ("improved", "worsened"), ("better", "worse"), ("stronger", "weaker"),
    ("faster", "slower"), ("always", "never"), ("optional", "required"),
    ("cheap", "expensive"), ("safe", "unsafe"), ("likely", "unlikely"),
    ("yes", "no"), ("true", "false"), ("enabled", "disabled"),
    ("deployed", "rolled back"), ("shipping", "cancelled"),
]

# Modals whose appearance or disappearance changes a plan into a claim.
_MODAL_RE = re.compile(
    r"\b(will|would|shall|should|can|could|may|might|must)\b")


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

    # 2. Numbers changed -- in either direction. "$50" -> "$15" is
    #    catastrophic and invisible, and so is "the cost rose" -> "the cost
    #    rose to 3000": an *invented* quantity is exactly as dangerous as a
    #    dropped one, and the old rule only looked one way (audit A3, rule 4).
    orig_nums, ref_nums = set(_numbers(original)), set(_numbers(refined))
    dropped = orig_nums - ref_nums
    if dropped and not _explained_by_wordform(dropped, original, refined):
        return GuardVerdict(False, f"number changed or dropped: {sorted(dropped)}")
    invented = ref_nums - orig_nums
    if invented and not _explained_by_wordform_back(invented, original, refined):
        return GuardVerdict(False, f"number invented: {sorted(invented)}")

    # 2b. Antonym swap. Counts on both sides must move together, so a
    #     legitimate paraphrase that keeps the pair intact passes, while
    #     "delaying" -> "proceeding" style inversions do not.
    swapped = _opposite_swap(original, refined)
    if swapped:
        return GuardVerdict(False, f"meaning inverted: {swapped}")

    # 2c. The general case the table is a sample of: a content word replaced
    #     by a *dissimilar* content word at the same aligned position. ASR
    #     fixes are near-misses (their/there, accept/except, Kuberentes/
    #     Kubernetes), so distance is the signal; Alice->Bob and
    #     delaying->proceeding are far apart in spelling and close together
    #     in sentence, which is precisely the shape of an invented edit.
    risky = _risky_substitution(orig_words, ref_words)
    if risky:
        return GuardVerdict(False, f"word substituted without cause: {risky}")

    # 2d. Modality. Dropping or adding a modal turns "we will ship" into
    #     "we shipped" and "it must be reviewed" into "it is reviewed" --
    #     a plan becomes a claim, which is the kind of edit a reader never
    #     re-checks. Legitimate cleanups do not need to touch modals, so any
    #     change in the count is rejected and the raw text is kept.
    # Apostrophes stripped first: ASR writes "cant" and the refiner writes
    # "can't", and \bcan\b matches one but not the other. Counting the same
    # way rule 1 does keeps a pure punctuation fix from looking like a
    # modality change.
    norm_o = (original or '').replace("'", '')
    norm_r = (refined or '').replace("'", '')
    om, rm = len(_MODAL_RE.findall(norm_o)), len(_MODAL_RE.findall(norm_r))
    if om != rm:
        return GuardVerdict(False, f"modal added or removed ({om} -> {rm})")

    # 3. Dictionary term dropped -- the LLM "corrected" a product name into
    #    an ordinary English word, exactly what the dictionary exists to stop.
    if dictionary_terms:
        low_ref, low_org = refined.lower(), original.lower()
        for term in dictionary_terms:
            t = term.lower()
            # Substring matching counted "ai" as present inside "email", so a
            # short dictionary entry was never really checked (audit A3,
            # rule 5). Word boundaries, like context/snippets.py already does.
            if _has_term(low_org, t) and not _has_term(low_ref, t):
                return GuardVerdict(False, f"dictionary term dropped: {term!r}")

    # 3b. Proper nouns. "Alice" -> "Bob" is the most damaging single-token
    #     error a refiner can make and none of the original four rules saw it:
    #     no negation moved, no digit moved, and "Bob" is probably not in the
    #     user's dictionary. A capitalised token that disappears is a cheap
    #     proxy for an entity -- no tagger, no model, no new dependency.
    entities = _entities(original) - _entities(refined)
    if entities:
        return GuardVerdict(False, f"entity dropped: {sorted(entities)[:4]}")

    # 3c. A name at the very start of a sentence is indistinguishable from an
    #     ordinary sentence-initial capital -- until the two texts start with
    #     *different* capitalised words.
    fo = (original or '').lstrip().split()
    fr = (refined or '').lstrip().split()
    if fo and fr and fo[0][:1].isupper() and fr[0][:1].isupper():
        wa = fo[0].strip(',.;:!?')
        wb = fr[0].strip(',.;:!?')
        if (wa.lower() != wb.lower()
                and wa.lower() not in _SENTENCE_STARTS
                and wb.lower() not in _SENTENCE_STARTS
                and SequenceMatcher(None, wa.lower(), wb.lower()).ratio() < 0.6):
            return GuardVerdict(False, f'leading entity swapped: {wa} -> {wb}')

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


# Words a refiner is *expected* to change: fillers and stutter repeats. They
# are deletions rather than substitutions, but a subset also gets swapped for
# a real word ("uh" -> "the"), so they are exempt from the substitution rule.
_FILLER = {"um", "uh", "er", "ah", "oh", "hmm", "so", "like", "you", "the",
           "a", "an", "to", "too", "two", "i", "im", "ill", "id", "its",
           "lets", "thats", "cant", "wont", "isnt", "dont", "didnt", "its"}


def _risky_substitution(orig: List[str], ref: List[str]) -> str:
    """Return "a -> b" for the first unexplained content-word swap, else ''."""
    if not orig or not ref:
        return ''
    sm = SequenceMatcher(a=orig, b=ref, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'replace':
            continue
        # Pair the blocks up positionally; unequal block sizes are ordinary
        # re-wording ("a lot of" -> "many") and are left to the ratio rules.
        if (i2 - i1) != (j2 - j1):
            continue
        for a, b in zip(orig[i1:i2], ref[j1:j2]):
            if len(a) < 5 or len(b) < 5 or a == b:
                continue
            if a in _FILLER or b in _FILLER:
                continue
            if a in NEGATIONS or b in NEGATIONS:
                continue            # already counted by rule 1
            if any(ch.isdigit() for ch in a) or any(ch.isdigit() for ch in b):
                continue            # number rules cover these
            if SequenceMatcher(None, a, b).ratio() >= 0.55:
                continue            # a plausible spelling correction
            return f"{a} \u2192 {b}"
    return ''


def _has_term(haystack: str, term: str) -> bool:
    """Word-boundary containment, tolerant of trailing punctuation."""
    if not term:
        return False
    try:
        return re.search(r"\b" + re.escape(term) + r"\b", haystack) is not None
    except re.error:
        return term in haystack


def _counts(text: str) -> "dict[str, int]":
    from collections import Counter
    return Counter(_words(text))


def _opposite_swap(original: str, refined: str) -> str:
    """Describe a pair whose two members traded places, else ''."""
    if not original or not refined:
        return ''
    oc, rc = _counts(original), _counts(refined)
    for a, b in OPPOSITES:
        da, db = rc.get(a, 0) - oc.get(a, 0), rc.get(b, 0) - oc.get(b, 0)
        if da and db and da == -db:
            return f"{a}/{b}"
    return ''


def _entities(text: str) -> Set[str]:
    """Capitalised tokens that are not sentence-initial or a known proper
    noun pattern. Lower-cased for comparison; "I" and acronyms are excluded
    because they carry no identity that could be swapped."""
    out = set()
    for i, m in enumerate(re.finditer(r"\b([A-Z][a-z]{1,20})\b", text or '')):
        if i == 0:
            continue                      # sentence start carries no signal
        word = m.group(1)
        if word in _SENTENCE_STARTS or word.lower() in NEGATIONS:
            continue
        out.add(word.lower())
    return out


# Words that are capitalised at the start of a clause for grammar, not
# because they name something, and therefore safe for a refiner to move.
_SENTENCE_STARTS = {"the", "this", "that", "these", "those", "it", "he", "she",
                    "they", "we", "you", "i", "and", "but", "so", "if", "when",
                    "while", "after", "before", "however", "also", "then",
                    "monday", "tuesday", "wednesday", "thursday", "friday",
                    "saturday", "sunday", "january", "february", "march",
                    "april", "june", "july", "august", "september", "october",
                    "november", "december"}


def _explained_by_wordform_back(invented: Set[str], original: str, refined: str) -> bool:
    """Allow a digit to appear if it was spelled out as a word before."""
    words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12",
    }
    low = original.lower()
    return all(n in words.values() and
               any(w == n for w, n in words.items() if re.search(r"\b" + w + r"\b", low))
               for n in invented)


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

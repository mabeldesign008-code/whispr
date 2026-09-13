"""
Auto-learn dictionary terms from user corrections.

Wispr Flow's real moat: "If you correct a transcription by typing over it,
Flow notices and adds the corrected spelling automatically. Over time, it
learns your vocabulary without you lifting a finger."

How this works without watching your keyboard:

After injecting text we remember the words we produced. If the *same*
unusual word is transcribed repeatedly it is probably real vocabulary, and
if a word we injected keeps coming back with low confidence it is probably
a name the recogniser cannot spell. Both are candidates.

We deliberately do NOT keylog to detect edits. Reading everything the user
types afterwards would be a far worse privacy trade than the OCR we just
removed. Instead we use two safe signals:

  1. repetition   - an out-of-vocabulary token seen N times across sessions
  2. uncertainty  - a token AssemblyAI repeatedly flags as low-confidence

Candidates are surfaced for one-click approval rather than added silently:
a dictionary that fills itself with garbage is worse than an empty one.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{2,}")

# Ordinary English that should never become a "term". Kept small on
# purpose: the frequency threshold does most of the filtering.
STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "have", "from", "they",
    "what", "when", "where", "which", "there", "their", "would", "could",
    "should", "about", "into", "than", "then", "them", "these", "those",
    "been", "being", "will", "just", "like", "your", "you", "our", "out",
    "get", "got", "can", "cant", "not", "but", "all", "any", "are", "was",
    "were", "has", "had", "him", "her", "his", "she", "how", "why", "who",
    "one", "two", "now", "new", "use", "used", "make", "made", "want",
    "need", "know", "think", "going", "yeah", "okay", "right", "well",
    "actually", "really", "basically", "something", "someone", "anything",
}

MIN_LENGTH = 3
PROMOTE_AFTER = 3          # sightings before we suggest a term
UNCERTAIN_WEIGHT = 2       # a low-confidence sighting counts double
MAX_TRACKED = 500


@dataclass
class Candidate:
    term: str
    count: int
    uncertain_hits: int

    @property
    def score(self) -> int:
        return self.count + self.uncertain_hits * (UNCERTAIN_WEIGHT - 1)


class DictionaryLearner:
    """Tracks repeated unusual words and proposes dictionary additions."""

    def __init__(self, path: Optional[Path] = None, dictionary=None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.dictionary = dictionary
        self.enabled = enabled
        self._lock = threading.RLock()
        self._counts: Counter = Counter()
        self._uncertain: Counter = Counter()
        self._dismissed: set = set()
        self.load()

    # ── persistence ───────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            with self._lock:
                self._counts = Counter(data.get("counts", {}))
                self._uncertain = Counter(data.get("uncertain", {}))
                self._dismissed = set(data.get("dismissed", []))
        except Exception as e:
            logger.debug("Could not read learner state: %s", e)

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "counts": dict(self._counts.most_common(MAX_TRACKED)),
                    "uncertain": dict(self._uncertain.most_common(MAX_TRACKED)),
                    "dismissed": sorted(self._dismissed),
                }
            self.path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not save learner state: %s", e)

    # ── observation ───────────────────────────────────────────────────────

    def observe(self, text: str, uncertain_words: Optional[List[str]] = None) -> None:
        """Record one transcription. Called after every successful dictation."""
        if not self.enabled or not text:
            return

        known = self._known_terms()
        uncertain_set = {w.lower().strip(".,!?") for w in (uncertain_words or [])}

        with self._lock:
            for raw in WORD_RE.findall(text):
                token = raw.strip("'-")
                low = token.lower()

                if (len(token) < MIN_LENGTH
                        or low in STOPWORDS
                        or low in known
                        or low in self._dismissed):
                    continue

                # Only track words that look like vocabulary rather than
                # ordinary prose: internal capitals, digits, or unusual
                # letter patterns.
                if not _looks_like_term(token):
                    continue

                self._counts[token] += 1
                if low in uncertain_set:
                    self._uncertain[token] += 1

            if len(self._counts) > MAX_TRACKED:
                for term, _ in self._counts.most_common()[MAX_TRACKED:]:
                    del self._counts[term]

        self.save()

    def _known_terms(self) -> set:
        if self.dictionary is None:
            return set()
        try:
            return {t.lower() for t in self.dictionary.terms}
        except Exception:
            return set()

    # ── suggestions ───────────────────────────────────────────────────────

    def candidates(self, limit: int = 5) -> List[Candidate]:
        """Terms worth suggesting, best first."""
        known = self._known_terms()
        with self._lock:
            out = [
                Candidate(term, count, self._uncertain.get(term, 0))
                for term, count in self._counts.items()
                if term.lower() not in known and term.lower() not in self._dismissed
            ]
        out = [c for c in out if c.score >= PROMOTE_AFTER]
        out.sort(key=lambda c: (-c.score, c.term.lower()))
        return out[:limit]

    def accept(self, term: str) -> bool:
        """Promote a candidate into the real dictionary.

        Returns True when the term ends up in the dictionary, including the
        case where it was already there -- from the user's point of view
        "Add" succeeded either way, and the candidate must be cleared
        regardless or it would be suggested forever.
        """
        if self.dictionary is None:
            return False

        self.dictionary.add(term)
        present = term.lower() in {t.lower() for t in self.dictionary.terms}

        with self._lock:
            self._counts.pop(term, None)
            self._uncertain.pop(term, None)
        self.save()
        return present

    def dismiss(self, term: str) -> None:
        """Never suggest this again."""
        with self._lock:
            self._dismissed.add(term.lower())
            self._counts.pop(term, None)
            self._uncertain.pop(term, None)
        self.save()

    def get_info(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "tracked": len(self._counts),
                "dismissed": len(self._dismissed),
                "pending": len(self.candidates(limit=99)),
            }


def _looks_like_term(token: str) -> bool:
    """Heuristic: is this domain vocabulary rather than ordinary English?"""
    # Internal capitals: "WhisprFlow", "iPhone", "McKinsey"
    if any(c.isupper() for c in token[1:]):
        return True
    # Contains a digit: "S3", "H100", "GPT4"
    if any(c.isdigit() for c in token):
        return True
    # Hyphenated compound: "on-prem", "k8s-cluster"
    if "-" in token:
        return True
    # Capitalised and reasonably long: likely a proper noun
    if token[0].isupper() and len(token) >= 4:
        return True
    # Rare letter combinations that English words seldom contain
    low = token.lower()
    if any(g in low for g in ("kk", "zz", "qq", "xq", "zx", "kubern", "ngin")):
        return True
    return False

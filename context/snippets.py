"""
Voice snippets -- say a trigger phrase, get a block of canned text.

Wispr Flow ships these ("my email address" -> your address, "standard
disclaimer" -> the paragraph). They are the cheapest possible win: no
model, no latency, no network, and they remove the most tedious typing
there is.

Matching runs on the *transcript*, before refinement, so it costs nothing
and cannot be mangled by the LLM. Two modes:

  exact   the whole utterance is the trigger -> replace it entirely
  inline  the trigger appears mid-sentence  -> substitute in place

Stored at %APPDATA%/WhisprFlow/snippets.json so it is hand-editable.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_TRIGGER_WORDS = 8
MAX_SNIPPETS = 200

# Spoken triggers arrive punctuated and cased inconsistently ("My email
# address." vs "my email address"), so both sides are normalised before
# comparison.
_NORMALISE = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    return _WS.sub(" ", _NORMALISE.sub(" ", (text or "").lower())).strip()


class SnippetSet:
    """Trigger phrase -> expansion text."""

    def __init__(self, path: Optional[Path] = None, enabled: bool = True):
        self.path = Path(path) if path else None
        self.enabled = enabled
        self._lock = threading.RLock()
        self._snippets: Dict[str, str] = {}      # normalised trigger -> text
        self._display: Dict[str, str] = {}       # normalised -> as written
        self.load()

    # ── persistence ───────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not read snippets.json: %s", e)
            return

        with self._lock:
            self._snippets.clear()
            self._display.clear()
            for entry in data.get("snippets", []):
                try:
                    self._add_unlocked(entry["trigger"], entry["text"])
                except Exception:
                    continue

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                payload = {
                    "_comment": (
                        "Say the trigger phrase while dictating and it is "
                        "replaced by the text. Triggers are matched without "
                        "punctuation or case."
                    ),
                    "snippets": [
                        {"trigger": self._display[k], "text": v}
                        for k, v in self._snippets.items()
                    ],
                }
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("Could not save snippets: %s", e)

    def write_template(self) -> None:
        if not self.path or self.path.exists():
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({
                "_comment": (
                    "Say the trigger phrase while dictating and it is "
                    "replaced by the text. Triggers are matched without "
                    "punctuation or case."
                ),
                "snippets": [
                    {"trigger": "my email address", "text": "you@example.com"},
                    {"trigger": "standard signoff",
                     "text": "Best regards,\nYour Name"},
                ],
            }, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── mutation ──────────────────────────────────────────────────────────

    def add(self, trigger: str, text: str) -> bool:
        with self._lock:
            if self._add_unlocked(trigger, text):
                self.save()
                return True
            return False

    def _add_unlocked(self, trigger: str, text: str) -> bool:
        key = normalise(trigger)
        if not key or not (text or "").strip():
            return False
        if len(key.split()) > MAX_TRIGGER_WORDS:
            return False
        if len(self._snippets) >= MAX_SNIPPETS and key not in self._snippets:
            return False
        self._snippets[key] = text
        self._display[key] = trigger.strip()
        return True

    def remove(self, trigger: str) -> bool:
        with self._lock:
            key = normalise(trigger)
            if key in self._snippets:
                del self._snippets[key]
                self._display.pop(key, None)
                self.save()
                return True
            return False

    # ── expansion ─────────────────────────────────────────────────────────

    def expand(self, text: str) -> Tuple[str, List[str]]:
        """Return (expanded_text, triggers_used).

        Longest trigger first, so "my work email address" wins over
        "my email address" when both are defined.
        """
        if not self.enabled or not text:
            return text, []

        with self._lock:
            if not self._snippets:
                return text, []
            triggers = sorted(self._snippets, key=lambda k: -len(k.split()))
            snapshot = dict(self._snippets)

        normalised = normalise(text)
        used: List[str] = []

        # Whole utterance is exactly one trigger -> replace outright, so
        # "my email address" alone yields just the address.
        for key in triggers:
            if normalised == key:
                return snapshot[key], [key]

        result = text
        for key in triggers:
            pattern = re.compile(
                r"\b" + r"[\s,.\-]+".join(re.escape(w) for w in key.split()) + r"\b",
                re.IGNORECASE,
            )
            if pattern.search(result):
                result = pattern.sub(lambda _m: snapshot[key], result)
                used.append(key)

        return result, used

    # ── access ────────────────────────────────────────────────────────────

    @property
    def triggers(self) -> List[str]:
        with self._lock:
            return [self._display[k] for k in self._snippets]

    def get(self, trigger: str) -> Optional[str]:
        with self._lock:
            return self._snippets.get(normalise(trigger))

    def __len__(self) -> int:
        with self._lock:
            return len(self._snippets)

    def get_info(self) -> dict:
        return {"enabled": self.enabled, "count": len(self)}

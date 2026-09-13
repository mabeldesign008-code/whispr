"""
User dictionary -- names, jargon, acronyms, product names.

This is the highest-ROI accuracy feature in dictation software and the app
had none. Terms flow to:
  * AssemblyAI -> keyterms_prompt (semantic biasing, supported on
                  universal-3-5-pro / universal-3-pro / slam-1)
  * LLM        -> the refinement prompt AND the guard, so a rewrite that
                  drops one of your product names is rejected outright

Stored as plain text at %APPDATA%/WhisprFlow/user_dictionary.txt so the
user can edit it by hand. One term per line, # for comments.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, List, Optional


def default_config_dir() -> Path:
    """Per-user config dir. Never CWD -- the old code used a relative .env,
    so launching from a shortcut with a different working directory silently
    lost all settings."""
    if os.name == "nt":
        base = os.getenv("APPDATA") or os.path.expanduser("~")
    else:
        base = os.getenv("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    d = Path(base) / "WhisprFlow"
    d.mkdir(parents=True, exist_ok=True)
    return d


class UserDictionary:
    MAX_TERMS = 1000
    MAX_WORDS_PER_TERM = 6      # AssemblyAI's per-phrase limit
    MAX_TERM_CHARS = 50

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_config_dir() / "user_dictionary.txt"
        self._lock = threading.RLock()
        self._terms: List[str] = []
        self.load()

    # ── persistence ───────────────────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            self._terms = []
            if not self.path.exists():
                self._write_template()
                return
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    self._add_unlocked(line)
            except Exception:
                self._terms = []

    def save(self) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    "# WhisprFlow user dictionary\n"
                    "# One term or short phrase per line (max 6 words).\n"
                    "# Names, acronyms, product names, jargon.\n"
                    "# Lines starting with # are ignored.\n\n"
                )
                self.path.write_text(header + "\n".join(self._terms) + "\n", encoding="utf-8")
            except Exception:
                pass

    def _write_template(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                "# WhisprFlow user dictionary\n"
                "# One term or short phrase per line (max 6 words).\n"
                "# Add names, acronyms, product names and jargon here so the\n"
                "# recogniser stops guessing at them.\n"
                "#\n"
                "# Examples:\n"
                "# Kubernetes\n"
                "# WhisprFlow\n"
                "# Mabel Design\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    # ── mutation ──────────────────────────────────────────────────────────

    def add(self, term: str) -> bool:
        with self._lock:
            if self._add_unlocked(term):
                self.save()
                return True
            return False

    def add_many(self, terms: Iterable[str]) -> int:
        with self._lock:
            n = sum(1 for t in terms if self._add_unlocked(t))
            if n:
                self.save()
            return n

    def _add_unlocked(self, term: str) -> bool:
        term = (term or "").strip()
        if not term or term.startswith("#"):
            return False
        if len(term) > self.MAX_TERM_CHARS:
            return False
        if len(term.split()) > self.MAX_WORDS_PER_TERM:
            return False
        if len(self._terms) >= self.MAX_TERMS:
            return False
        if any(t.lower() == term.lower() for t in self._terms):
            return False
        self._terms.append(term)
        return True

    def remove(self, term: str) -> bool:
        with self._lock:
            before = len(self._terms)
            self._terms = [t for t in self._terms if t.lower() != term.strip().lower()]
            if len(self._terms) != before:
                self.save()
                return True
            return False

    # ── access ────────────────────────────────────────────────────────────

    @property
    def terms(self) -> List[str]:
        with self._lock:
            return list(self._terms)

    def __len__(self) -> int:
        with self._lock:
            return len(self._terms)

    def as_keyterms(self) -> List[str]:
        """For AssemblyAI keyterms_prompt."""
        return self.terms

    def as_prompt_line(self, limit: int = 60) -> str:
        """Compact form for the LLM refinement prompt."""
        t = self.terms[:limit]
        return ", ".join(t) if t else ""

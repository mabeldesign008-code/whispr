"""Smart formatting pass: paragraphs, bullets, numbering -- and nothing else.

The Dictation API returns correctly punctuated, filler-free text as one
or two flat blocks (it cleans, but it does not LAY OUT). This module adds
the layout: it hands the already-clean text to a small Groq model with a
prompt that permits exactly three edits -- insert paragraph breaks, add
'-' bullets to parallel items, add '1.' numbering to enumerations -- and
forbids changing any word.

Why this can't repeat the old refiner's failure (AUDIT_REPORT §2-§3):

1. VERIFIABLE. A formatting pass that may not touch words is checkable
   deterministically: the word streams of input and output must be
   identical once list markers and whitespace are ignored. There is no
   heuristic judgement, so there is no 42% false-rejection rate; the
   check is exact.

2. FAIL-SAFE, NOT FAIL-BROKEN. Any API error, timeout, truncation, or a
   verifier mismatch returns the Dictation text unchanged. The worst
   case is always "what you'd have got anyway", and it is logged.

3. THE GROK REQUEST SPEC THAT WORKED BEFORE IS REUSED: gpt-oss with a
   single user message (its docs discourage system prompts),
   reasoning_effort="low", include_reasoning=False so hidden tokens are
   never billed against the completion budget, generous token budget.

Skipped entirely for Code/Terminal profiles (profile.allow_format=False),
very short texts, and when cleanup is off.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

API_URL = "https://api.groq.com/openai/v1/chat/completions"

#: 20b is plenty for a layout task and runs at ~1000 tok/s on Groq, which
#: keeps the extra hop to ~0.5 s. 120b is the fallback id.
DEFAULT_MODEL = os.getenv("GROQ_FORMAT_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]

DEAD_MODEL_MARKERS = ("decommissioned", "model_decommissioned",
                      "no longer supported", "does not exist",
                      "model_not_found", "unknown model")

#: Below this there is nothing to lay out -- a short take is a sentence
#: or two and structure would only be noise (and another 500 ms wait).
MIN_WORDS = 15

#: Match (and allow) structural markers the formatter is permitted to add.
_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+", re.M)
_WORD = re.compile(r"[A-Za-z0-9']+")


def _word_stream(text: str) -> list:
    """The words of `text`, lowercased, ignoring whitespace and list
    markers. Two texts that lay out differently but say the same thing
    have the same stream -- which is exactly our acceptance test."""
    return _WORD.findall(_LIST_MARKER.sub("", text.lower()))


def words_preserved(before: str, after: str) -> bool:
    return _word_stream(before) == _word_stream(after)


FORMAT_PROMPT = """You are a text layout engine. You receive dictated text that is \
already correct: the words, word order and punctuation are final.

Make it READABLE by changing layout ONLY. You may:
- Split it into paragraphs at natural topic shifts, separated by blank lines.
- Turn a run of clearly parallel items into a "- " bulleted list, one item per line.
- Turn words that announce an explicit sequence ("first ... second ... third", "one ... two ... three", \
"number one ... number two") into a "1." numbered list, one item per line. \
The announcing words stay inside their items -- "1. First, ..." is correct; \
deleting "First," is not.

Hard rules:
- NEVER change, delete, add, reorder or paraphrase any word. The only things \
you may insert are newlines, "- " and "N.". Not even a greeting.
- Keep every sentence exactly where it is in the sequence.
- If the text is a command, code, a URL, an email address, or a single short \
message, output it completely unchanged.
- Output ONLY the formatted text. No preamble, no quotes, no code fences."""


@dataclass
class FormatResult:
    text: str
    ok: bool              # False = fell back to input
    changed: bool = False
    error: Optional[str] = None
    latency_ms: int = 0


class Formatter:
    """Applies the layout pass; fall back to the input on ANY problem."""

    def __init__(self, api_key: str = "", model: Optional[str] = None,
                 timeout: float = 10.0):
        self.api_key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        chain = [model or DEFAULT_MODEL, *FALLBACK_MODELS]
        self.models = list(dict.fromkeys(chain))
        self.timeout = timeout
        self._client = None

        self.calls = 0
        self.changed = 0
        self.failures = 0
        self.last_error = ""
        self.last_model_used = ""

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, key: str) -> None:
        self.api_key = (key or "").strip()
        self._client = None

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                limits=httpx.Limits(max_keepalive_connections=2,
                                    keepalive_expiry=300),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def worth_formatting(self, text: str) -> bool:
        return len(text.split()) >= MIN_WORDS

    async def apply(self, clean_text: str) -> FormatResult:
        """Lay out `clean_text`. Returns the input unchanged on ANY issue."""
        started = time.perf_counter()
        if not self.is_configured:
            return FormatResult(clean_text, ok=False, error="no Groq key")
        if not self.worth_formatting(clean_text):
            return FormatResult(clean_text, ok=True, changed=False)

        self.calls += 1
        payload = {
            "messages": [{"role": "user",
                          "content": FORMAT_PROMPT + "\n\nTEXT:\n" + clean_text}],
            "temperature": 0.3,
            "max_completion_tokens": min(max(len(clean_text), 2048), 8192),
            "reasoning_effort": "low",
            "include_reasoning": False,
        }

        try:
            client = await self._get_client()
            last_error = "unknown"
            for model in self.models:
                payload["model"] = model
                resp = await client.post(API_URL, json=payload)
                if resp.status_code != 200:
                    try:
                        msg = str((resp.json().get("error") or {}).get("message", ""))
                    except Exception:
                        msg = resp.text[:200]
                    if any(k in msg.lower() for k in DEAD_MODEL_MARKERS):
                        last_error = f"model {model} retired"
                        continue
                    return self._fail(clean_text, f"HTTP {resp.status_code}: {msg[:120]}",
                                      started)
                choice = (resp.json().get("choices") or [{}])[0]
                if choice.get("finish_reason") == "length":
                    last_error = "truncated"
                    continue
                raw = (choice.get("message") or {}).get("content") or ""
                if isinstance(raw, list):
                    raw = "".join(p.get("text", "") if isinstance(p, dict) else str(p)
                                  for p in raw)
                out = raw.strip()
                if not out:
                    last_error = "empty output"
                    continue
                # THE guarantee: layout changed, words did not.
                if not words_preserved(clean_text, out):
                    return self._fail(clean_text,
                                      "layout changed the words — kept Dictation text",
                                      started)
                self.last_model_used = model
                self.last_error = ""
                changed = out != clean_text
                if changed:
                    self.changed += 1
                return FormatResult(
                    out, ok=True, changed=changed,
                    latency_ms=int((time.perf_counter() - started) * 1000))
            return self._fail(clean_text, last_error, started)
        except Exception as e:
            return self._fail(clean_text, f"{type(e).__name__}: {e}", started)

    def _fail(self, text: str, error: str, started: float) -> FormatResult:
        self.failures += 1
        self.last_error = error
        return FormatResult(text, ok=False, error=error,
                            latency_ms=int((time.perf_counter() - started) * 1000))

    def get_stats(self) -> dict:
        return {
            "configured": self.is_configured,
            "calls": self.calls,
            "changed": self.changed,
            "failures": self.failures,
            "last_error": self.last_error,
            "model": self.models[0],
        }

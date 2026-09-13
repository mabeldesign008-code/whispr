"""
Shared API health record for the two Groq callers (refiner + command processor).

Why this exists: both callers swallow failures into a *degraded but silent*
path — a non-200 from Groq becomes `basic_cleanup()` text, a model that no
longer exists becomes "the refiner never runs" — and nothing anywhere surfaced
it. A user whose refinement had been failing for weeks still saw a UI that
looked healthy. That is the worst possible failure mode for a dictation tool:
the tool works, the *quality* feature quietly stops working.

This module is deliberately dumb: counters, the last error per caller, and a
"this model id is dead" memory so a fallback list is not retried on every
dictation. No I/O, no locks beyond a simple RLock (the refiner runs on a worker
thread, the command processor on the Tk callback thread).

Model-id drift is not hypothetical: Groq's deprecation page retired both ids
this app shipped with (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) on
2026-08-16, naming `openai/gpt-oss-20b` and `openai/gpt-oss-120b` /
`qwen/qwen3.6-27b` as the replacements
(https://console.groq.com/docs/deprecations). A retired id returns
HTTP 400/404 for the whole request, so a *list* of ids to try in order is the
cheap way to survive the next retirement.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

# Set by both callers; `refine/__init__.py` exports it.
__all__ = ["ApiHealth", "health", "DEAD_MODEL_MARKERS", "error_detail"]

# Substrings that mean "this exact request will never succeed, and switching
# model is the fix" rather than "the service is temporarily unhappy".
DEAD_MODEL_MARKERS = (
    "model is not available",
    "not a valid model",
    "unsupported model",
    "unknown model",
    "model_not_found",
    "does not exist",
    "no longer available",
    "has been deprecated",
    "decommissioned",
)


def error_detail(resp) -> str:
    """Provider error text, best effort.

    Groq puts the message in `{"error": {"message": ...}}`; other bodies are
    truncated as-is. Never raises -- this runs on the failure path, where the
    last thing we want is a second exception hiding the first.
    """
    try:
        data = resp.json()
    except Exception:
        body = " ".join((getattr(resp, "text", "") or "").split())[:200]
        return body or f"HTTP {getattr(resp, 'status_code', '?')}"
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or data)[:200]
        if err:
            return str(err)[:200]
        return str(data)[:200]
    return str(data)[:200]


class ApiHealth:
    """Per-caller success/failure counters plus dead-model memory."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.calls: int = 0
        self.failures: int = 0
        self.http_errors: int = 0
        self.timeouts: int = 0
        self.fallbacks: int = 0            # refiner used basic_cleanup instead of the LLM
        self.last_error: Optional[str] = None
        self.last_error_at: float = 0.0
        self.last_source: str = "refiner"
        self._dead_models: Dict[str, bool] = {}

    # -- recording ---------------------------------------------------------
    def note_call(self, source: str) -> None:
        with self._lock:
            self.calls += 1
            self.last_source = source

    def note_success(self, source: str) -> None:
        with self._lock:
            self.last_source = source
            self.last_error = None

    def note_failure(self, source: str, message: str, *,
                     http: bool = False, timeout: bool = False,
                     fallback: bool = False) -> None:
        text = " ".join((message or "").split())[:240]
        with self._lock:
            self.failures += 1
            if http:
                self.http_errors += 1
            if timeout:
                self.timeouts += 1
            if fallback:
                self.fallbacks += 1
            self.last_error = f"{source}: {text}"
            self.last_error_at = time.time()
            self.last_source = source

    def mark_dead(self, source: str, model: str) -> None:
        with self._lock:
            if model:
                self._dead_models[f"{source}:{model}"] = True

    # -- querying ----------------------------------------------------------
    def is_dead(self, source: str, model: str) -> bool:
        if not model:
            return False
        with self._lock:
            return bool(self._dead_models.get(f"{source}:{model}"))

    def ordered(self, source: str, models: List[str]) -> List[str]:
        """Candidate models for `source`, known-dead ones pushed to the back.

        Dead models are kept rather than dropped: if every id in the list is
        marked dead the caller still needs *something* to try, and a fresh
        process (hence a fresh memory) is the documented way for a user to
        retry an id that was marked dead by a transient error.
        """
        with self._lock:
            live = [m for m in models if not self._dead_models.get(f"{source}:{m}")]
            dead = [m for m in models if self._dead_models.get(f"{source}:{m}")]
        return live + dead

    @property
    def consecutive_failure_hint(self) -> bool:
        """True when the last thing we did was fail. Drives the pill's label."""
        with self._lock:
            return self.failures > 0 and self.last_error is not None

    def summary(self) -> str:
        """One short line for the Settings window / fatal dialog context."""
        with self._lock:
            if self.calls == 0:
                return "no API calls made yet this session"
            if self.failures == 0:
                return f"LLM calls OK ({self.calls})"
            when = time.strftime("%H:%M", time.localtime(self.last_error_at)) if self.last_error_at else "-"
            return (
                f"LLM: {self.failures}/{self.calls} calls failed, last at {when}. "
                f"{self.last_error or ''}"
                + (f" | {len(self._dead_models)} model id(s) marked unavailable"
                   if self._dead_models else "")
            )


health = ApiHealth()

"""
Read and replace the current text selection in any application.

Windows has no universal "give me the selected text" API. UI Automation
exposes it for well-behaved controls, but most real apps (browsers,
Electron, terminals, Office) either do not implement TextPattern or
return the whole document. The reliable path everywhere is the clipboard:
send Ctrl+C, read it, restore what was there.

That is invasive, so it is done carefully:
  * the previous clipboard is saved and restored afterwards
  * we detect "nothing was selected" by checking whether the clipboard
    actually changed, rather than assuming the copy worked
  * modifiers are released first, or Ctrl+C arrives as Ctrl+Win+C
"""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Optional

import pyperclip
from pynput.keyboard import Controller, Key

from injector import wait_for_modifier_release

logger = logging.getLogger(__name__)

# Time for the target app to service the clipboard request. Electron apps
# are the slow case; native Win32 controls respond almost instantly.
COPY_SETTLE = 0.14
PASTE_SETTLE = 0.10
RESTORE_DELAY = 0.45

# Windows' clock granularity is ~15ms, so time_ns() alone can return the
# same value for two rapid calls and the sentinel would not be unique.
# A monotonic counter guarantees uniqueness regardless of clock
# resolution. Caught by CI on the Windows runner.
_probe_counter = itertools.count()


@dataclass
class Selection:
    text: str = ""
    ok: bool = True
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class SelectionManager:
    """Clipboard-based selection capture and replacement."""

    def __init__(self):
        self._kb = Controller()

    # ── read ──────────────────────────────────────────────────────────────

    def capture(self) -> Selection:
        """Copy the current selection without disturbing the clipboard."""
        wait_for_modifier_release()

        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None

        # A sentinel lets us distinguish "nothing selected" (clipboard
        # unchanged) from "the user had already copied this exact text".
        sentinel = f"__whisprflow_probe_{time.time_ns()}_{next(_probe_counter)}__"
        try:
            pyperclip.copy(sentinel)
        except Exception as e:
            return Selection(ok=False, error=f"clipboard unavailable: {e}")

        try:
            self._kb.press(Key.ctrl)
            self._kb.press("c")
            self._kb.release("c")
            self._kb.release(Key.ctrl)
        except Exception as e:
            self._restore(previous)
            return Selection(ok=False, error=f"could not send copy: {e}")

        time.sleep(COPY_SETTLE)

        try:
            captured = pyperclip.paste()
        except Exception:
            captured = sentinel

        self._restore(previous)

        if captured == sentinel:
            return Selection(ok=False, error="No text selected.")
        if not (captured or "").strip():
            return Selection(ok=False, error="Selection is empty.")

        return Selection(text=captured, ok=True)

    # ── write ─────────────────────────────────────────────────────────────

    def replace(self, text: str) -> bool:
        """Paste over the current selection, then restore the clipboard."""
        if not text:
            return False

        wait_for_modifier_release()

        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None

        try:
            pyperclip.copy(text)
        except Exception as e:
            logger.error("clipboard write failed: %s", e)
            return False

        time.sleep(0.04)
        try:
            self._kb.press(Key.ctrl)
            self._kb.press("v")
            self._kb.release("v")
            self._kb.release(Key.ctrl)
        except Exception as e:
            logger.error("paste failed: %s", e)
            self._restore(previous)
            return False

        time.sleep(PASTE_SETTLE)
        self._restore(previous, delay=RESTORE_DELAY)
        return True

    # ── helpers ───────────────────────────────────────────────────────────

    def _restore(self, previous, delay: float = 0.0) -> None:
        if previous is None:
            return

        def do_restore():
            if delay:
                time.sleep(delay)
            try:
                pyperclip.copy(previous)
            except Exception:
                pass

        if delay:
            import threading
            threading.Thread(target=do_restore, daemon=True).start()
        else:
            do_restore()

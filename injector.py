"""
Text injection at the cursor.

Two audit findings fixed here.

1.4  The paste fired while the hotkey modifiers were still physically held.
     Push-to-talk stops on key *release*, but only one of Ctrl/Win has been
     released at that moment. If Win is still down the OS sees Ctrl+Win+V
     (clipboard history / virtual desktop), not paste. We now wait for all
     modifiers to come up before sending anything.

1.5  The app ran two global keyboard hooks: the `keyboard` package here and
     `pynput` in main.py. Two low-level hooks can see and interfere with
     each other's synthetic events, which is the classic cause of "the
     paste sometimes doesn't happen" and "recording sometimes re-triggers".
     This module now uses pynput only, matching the listener side.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time

import pyperclip
from pynput.keyboard import Controller, Key

logger = logging.getLogger(__name__)

# Virtual key codes for the modifiers we must not fight with.
_VK = {
    "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    "lwin": 0x5B, "rwin": 0x5C,
}

# Short unicode strings are typed directly: no clipboard round-trip, no
# risk of clobbering what the user copied, and it works in fields that
# block paste. Longer text goes via clipboard because per-character typing
# gets slow and some apps drop keystrokes.
DIRECT_TYPE_LIMIT = 120


def _modifiers_down() -> bool:
    """True while any modifier key is physically held."""
    try:
        gks = ctypes.windll.user32.GetAsyncKeyState
        return any(gks(vk) & 0x8000 for vk in _VK.values())
    except Exception:
        return False


def wait_for_modifier_release(timeout: float = 0.4) -> bool:
    """Block until the user lets go. Returns False if they never did."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _modifiers_down():
            return True
        time.sleep(0.012)
    return not _modifiers_down()


class TextInjector:
    def __init__(self):
        self._kb = Controller()
        self._lock = threading.Lock()

    def inject(self, text: str) -> bool:
        """Insert text at the cursor. Blocking; call via asyncio.to_thread."""
        if not text:
            return False

        with self._lock:
            # Never send keys while the hotkey is still held.
            wait_for_modifier_release()

            if len(text) <= DIRECT_TYPE_LIMIT:
                if self._type_directly(text):
                    return True
                # fall through to clipboard if typing failed

            return self._paste(text)

    def _type_directly(self, text: str) -> bool:
        try:
            self._kb.type(text)
            return True
        except Exception as e:
            logger.debug("Direct typing failed (%s); using clipboard", e)
            return False

    def _paste(self, text: str) -> bool:
        try:
            previous = pyperclip.paste()
        except Exception:
            previous = None

        try:
            pyperclip.copy(text)
        except Exception as e:
            logger.error("Clipboard write failed: %s", e)
            return False

        time.sleep(0.04)
        try:
            self._kb.press(Key.ctrl)
            self._kb.press("v")
            self._kb.release("v")
            self._kb.release(Key.ctrl)
        except Exception as e:
            logger.error("Paste failed: %s", e)
            self._restore_later(previous)
            return False

        self._restore_later(previous)
        return True

    def _restore_later(self, previous, delay: float = 0.5) -> None:
        """Put the user's clipboard back once the paste has landed."""
        if previous is None:
            return

        def restore():
            time.sleep(delay)
            try:
                pyperclip.copy(previous)
            except Exception:
                pass

        threading.Thread(target=restore, daemon=True).start()

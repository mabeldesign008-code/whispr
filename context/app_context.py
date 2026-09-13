"""
Foreground application context.

Replaces the deleted screen_context.py, which screenshotted the entire
desktop and ran PaddleOCR on it: 0.3-2s in the critical path, low-signal
text (taskbar, clock, unrelated windows), and your whole screen shipped to
a third party on every dictation.

This reads the same information from the OS instead:

    GetForegroundWindow -> process name + window title      ~0.1 ms
    UIAutomation        -> focused control's text/type      ~2-10 ms

That is 80% of the value for 0.5% of the cost, and it is scoped to the one
control you are typing into rather than everything on screen.

Everything degrades gracefully: no pywin32, no uiautomation, a COM error,
an app with no accessibility surface -- all just yield less context, never
an exception. Context is an accuracy optimisation, never a requirement.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── optional dependencies ─────────────────────────────────────────────────

try:
    import win32gui
    import win32process
    _WIN32 = True
except Exception:
    _WIN32 = False

try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False

try:
    import uiautomation as _uia
    _UIA = True
except Exception:
    _UIA = False


# Titles often carry "document — App Name"; strip the app suffix so the
# document name reads cleanly in the prompt.
_TITLE_SPLIT = re.compile(r"\s+[\u2014\u2013|-]\s+")

MAX_SURROUNDING_CHARS = 240
UIA_TIMEOUT = 0.35     # never let a hung accessibility call stall dictation


@dataclass
class AppContext:
    """What the user is currently working in."""

    process: str = ""            # "Code.exe"
    app_name: str = ""           # "Visual Studio Code"
    window_title: str = ""
    control_type: str = ""       # "Edit", "Document", "ComboBox" ...
    surrounding_text: str = ""   # text already in the focused field
    captured_ms: int = 0
    partial: bool = False        # True if UIA was unavailable/slow

    @property
    def is_empty(self) -> bool:
        return not (self.process or self.window_title)

    def as_prompt(self) -> str:
        """Compact description for the refinement prompt. Deliberately
        short: the LLM needs a hint, not a document."""
        if self.is_empty:
            return ""
        bits = []
        if self.app_name:
            bits.append(self.app_name)
        if self.window_title:
            doc = _TITLE_SPLIT.split(self.window_title)[0].strip()
            if doc and doc.lower() != self.app_name.lower():
                bits.append(f'"{_clip(doc, 60)}"')
        if self.control_type:
            bits.append(f"[{self.control_type}]")
        return " \u00b7 ".join(bits)

    def as_dict(self) -> dict:
        return {
            "process": self.process,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "control_type": self.control_type,
            "has_surrounding": bool(self.surrounding_text),
            "captured_ms": self.captured_ms,
            "partial": self.partial,
        }


# Friendly names for the apps people actually dictate into.
_APP_NAMES = {
    "code.exe": "Visual Studio Code",
    "cursor.exe": "Cursor",
    "devenv.exe": "Visual Studio",
    "windowsterminal.exe": "Windows Terminal",
    "powershell.exe": "PowerShell",
    "cmd.exe": "Command Prompt",
    "slack.exe": "Slack",
    "discord.exe": "Discord",
    "teams.exe": "Microsoft Teams",
    "ms-teams.exe": "Microsoft Teams",
    "outlook.exe": "Outlook",
    "thunderbird.exe": "Thunderbird",
    "winword.exe": "Microsoft Word",
    "excel.exe": "Microsoft Excel",
    "powerpnt.exe": "PowerPoint",
    "notepad.exe": "Notepad",
    "notepad++.exe": "Notepad++",
    "sublime_text.exe": "Sublime Text",
    "obsidian.exe": "Obsidian",
    "notion.exe": "Notion",
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "firefox.exe": "Firefox",
    "brave.exe": "Brave",
    "whatsapp.exe": "WhatsApp",
    "telegram.exe": "Telegram",
    "explorer.exe": "File Explorer",
}


class AppContextReader:
    """Reads foreground app context. Cheap enough to call on every hotkey."""

    def __init__(self, enabled: bool = True, read_surrounding: bool = False):
        self.enabled = enabled
        # Reading the focused field's existing text is the most useful
        # signal but also the most privacy-sensitive, so it is off by
        # default and surfaced as its own toggle.
        self.read_surrounding = read_surrounding
        self._uia_lock = threading.Lock()
        self._uia_failures = 0

    @property
    def available(self) -> bool:
        return _WIN32

    def capture(self) -> AppContext:
        """Never raises. Returns an empty context if nothing is readable."""
        if not self.enabled or not _WIN32:
            return AppContext()

        started = time.perf_counter()
        ctx = AppContext()

        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return ctx
            ctx.window_title = win32gui.GetWindowText(hwnd) or ""
            ctx.process = self._process_name(hwnd)
            ctx.app_name = _APP_NAMES.get(
                ctx.process.lower(), _prettify(ctx.process)
            )
        except Exception as e:
            logger.debug("Foreground window read failed: %s", e)
            return ctx

        # UIA is the slow part and the part most likely to misbehave in a
        # sandboxed or remote-desktop session. Budgeted and failure-capped.
        if _UIA and self._uia_failures < 3:
            self._read_uia(ctx)
        else:
            ctx.partial = True

        ctx.captured_ms = int((time.perf_counter() - started) * 1000)
        return ctx

    def _process_name(self, hwnd) -> str:
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if _PSUTIL:
                return psutil.Process(pid).name()
        except Exception:
            pass
        return ""

    def _read_uia(self, ctx: AppContext) -> None:
        """Focused control type and (optionally) its text."""
        done = threading.Event()

        def work():
            try:
                with self._uia_lock:
                    el = _uia.GetFocusedControl()
                    if el is None:
                        return
                    try:
                        ctx.control_type = (el.ControlTypeName or "").replace("Control", "")
                    except Exception:
                        pass

                    if self.read_surrounding:
                        text = _control_text(el)
                        if text:
                            # Keep the tail: the caret is usually at the end,
                            # so the nearest context is the most relevant.
                            ctx.surrounding_text = _clip_tail(text, MAX_SURROUNDING_CHARS)
                self._uia_failures = 0
            except Exception as e:
                self._uia_failures += 1
                logger.debug("UIA read failed (%d): %s", self._uia_failures, e)
            finally:
                done.set()

        t = threading.Thread(target=work, daemon=True, name="uia-read")
        t.start()
        if not done.wait(UIA_TIMEOUT):
            # A hung accessibility call must never delay the user's speech.
            ctx.partial = True
            self._uia_failures += 1

    def get_info(self) -> dict:
        return {
            "enabled": self.enabled,
            "win32": _WIN32,
            "uiautomation": _UIA,
            "psutil": _PSUTIL,
            "read_surrounding": self.read_surrounding,
            "uia_failures": self._uia_failures,
        }


# ── helpers ───────────────────────────────────────────────────────────────


def _control_text(el) -> str:
    """Try the value pattern, then the text pattern, then the name."""
    for getter in (
        lambda: el.GetValuePattern().Value,
        lambda: el.GetTextPattern().DocumentRange.GetText(MAX_SURROUNDING_CHARS * 2),
        lambda: el.Name,
    ):
        try:
            value = getter()
            if value and isinstance(value, str) and value.strip():
                return value
        except Exception:
            continue
    return ""


def _prettify(process: str) -> str:
    if not process:
        return ""
    return process.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _clip_tail(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "\u2026" + text[-limit:]

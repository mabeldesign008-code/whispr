"""
WhisprFlow — system-wide AI dictation for Windows.

Pipeline:  hotkey → always-on capture (with pre-roll) → AssemblyAI
           → Groq refinement → hallucination guard → inject at cursor
"""

import asyncio
import ctypes
import logging
import os
import sys
import threading
import time
import traceback
import winsound
from collections import deque
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import Label, messagebox, scrolledtext

import pystray
from dotenv import load_dotenv, set_key
from PIL import Image, ImageDraw
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController

from audio import AudioCapture, process as process_audio
from context import ProfileSet, SnippetSet
from injector import TextInjector
from refine import TONE_NAMES, basic_cleanup, build_instruction, default_tone
from refine.commands import CommandProcessor
from selection import SelectionManager
from stt import DictationClient, UserDictionary, default_config_dir
from ui import theme
from ui.overlay import FloatingPill, PillState

# High-DPI awareness so the overlay renders crisply.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

def resource_path(*parts) -> Path:
    """Locate a bundled data file.

    PyInstaller unpacks datas into sys._MEIPASS at runtime; from source we
    resolve relative to this file. Never relative to CWD -- launching from
    a shortcut with a different working directory would break it.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


CONFIG_DIR = default_config_dir()
ENV_PATH = CONFIG_DIR / ".env"
load_dotenv(ENV_PATH)
load_dotenv()  # also honour a project-local .env during development

DEBUG = os.getenv("WHISPRFLOW_DEBUG", "").lower() in ("1", "true", "yes")

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("whisprflow")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def beep_async(freq: int, ms: int) -> None:
    """Never block the caller. The old code called winsound.Beep inline
    before opening the mic, which is how the first syllable got clipped."""
    threading.Thread(
        target=lambda: _safe_beep(freq, ms), daemon=True
    ).start()


def _safe_beep(freq: int, ms: int) -> None:
    try:
        winsound.Beep(freq, ms)
    except Exception:
        pass


_GTI = None          # lazily built GUITHREADINFO structure type


def _focus_signature():
    """(foreground window, focused control, owning pid) as integers.

    A paste can go wrong in two ways: the user Alt+Tabbed to another app, or
    they clicked into a different field of the same window. The top-level
    handle catches the first and the focused control catches the second, so
    both are sampled. Two documented Windows behaviours shape the code:

      * GetForegroundWindow returns NULL "in certain circumstances, such as
        when a window is losing activation", and GetGUIThreadInfo "may not
        return valid window handles ... when called to retrieve information
        for the foreground thread" (learn.microsoft.com/windows/win32/api/
        winuser/nf-winuser-getforegroundwindow, .../nf-winuser-getguithreadinfo).
        So 0 means *unknown*, never *changed* -- a refusal has to be based on
        two real values, or the app would refuse at random.
      * The same GetGUIThreadInfo page says rcCaret "may not give the correct
        position of the cursor" for an edit control, so the caret rectangle is
        deliberately not part of the signature. hwndFocus is stable and free.

    Cost is two Win32 calls with no allocation and no UI Automation, which is
    what makes it possible to take the sample on the keyboard-hook thread.
    """
    global _GTI
    if sys.platform != 'win32':
        return 0, 0, 0
    try:
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow() or 0)
        if not hwnd:
            return 0, 0, 0

        pid = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        if _GTI is None:
            from ctypes.wintypes import RECT

            class GUITHREADINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_ulong), ("flags", ctypes.c_ulong),
                    ("hwndActive", ctypes.c_void_p),
                    ("hwndFocus", ctypes.c_void_p),
                    ("hwndCapture", ctypes.c_void_p),
                    ("hwndMenuOwner", ctypes.c_void_p),
                    ("hwndMoveSize", ctypes.c_void_p),
                    ("hwndCaret", ctypes.c_void_p),
                    ("rcCaret", RECT),
                ]
            _GTI = GUITHREADINFO

        focus = 0
        if _GTI is not None:
            info = _GTI()
            info.cbSize = ctypes.sizeof(_GTI)
            # idThread 0 == the foreground thread, per the docs.
            if user32.GetGUIThreadInfo(0, ctypes.byref(info)) and info.hwndFocus:
                focus = int(info.hwndFocus)
        return hwnd, focus, int(pid.value or 0)
    except Exception:
        return 0, 0, 0


def _foreground_hwnd() -> int:
    """Just the top-level handle, for callers that only need that."""
    return _focus_signature()[0]


def _foreground_is_ours(sig) -> bool:
    """True when our own overlay is what took focus.

    The pill and the settings window are real Tk windows; if any of them ends
    up in the foreground while the pipeline runs, comparing handles would
    refuse a perfectly good paste. Same-pid means "the user did not change
    app", so the anchor is left alone.
    """
    return bool(sig[2]) and sig[2] == os.getpid()


def _foreground_process() -> str:
    """Process name of the foreground window, e.g. "chrome.exe" ("" on
    non-Windows or any failure). Used only to match formatting profiles.
    psutil was already a dependency for the deleted UIA reader; the one
    syscall is ~1 ms and never fails loudly enough to break a take.
    """
    if sys.platform != "win32":
        return ""
    try:
        import psutil

        pid = _focus_signature()[2]
        if not pid or pid == os.getpid():
            return ""
        return (psutil.Process(pid).name() or "").lower()
    except Exception:
        return ""


class WhisprFlowApp:
    def __init__(self):
        self.dictionary = UserDictionary()
        self.injector = TextInjector()
        # Command Mode is the one feature that still calls an LLM directly;
        # it stays optional (needs a Groq key) because the Dictation API has
        # no transform-on-selection endpoint.
        self.commands = CommandProcessor(api_key=os.getenv("GROQ_API_KEY", ""))
        self.selection = SelectionManager()

        # One client, one request per take: verbatim + cleaned text come
        # back together. This replaced the old streaming/batch STT + Groq
        # refiner + guard stack (see AUDIT_REPORT.md section 5).
        self.stt = DictationClient(api_key=os.getenv("ASSEMBLYAI_API_KEY", ""))

        # Always-on capture. The stream opens once and stays open, so the
        # pre-roll ring already holds the moment before the hotkey fired.
        saved_mic = os.getenv("WHISPRFLOW_MIC_DEVICE", "").strip()
        mic_index = AudioCapture.find_device_by_identifier(saved_mic)
        self.capture = AudioCapture(device=mic_index)
        self._device_map = {}
        self._meter_running = False

        # Context: which app the user is dictating into, hence how the
        # text should be formatted there (Code, Terminal, Chat, Email...).
        self.profiles = ProfileSet(CONFIG_DIR / "profiles.json")
        self.profiles.write_template()
        self.snippets = SnippetSet(CONFIG_DIR / "snippets.json")
        self.snippets.write_template()

        self._profile_instruction = ""
        self._live_process = ""

        self._state_lock = threading.RLock()
        self.is_recording = False
        self.generation = 0          # bumped on cancel to invalidate in-flight work
        self.last_raw_text = ""
        self.last_audio = None
        self.last_text_injected = ""
        self.last_failed = False
        # Window that was focused when the *release* happened. The pipeline
        # runs 1-3 s later, and every one of those seconds is enough for the
        # user to Alt+Tab; without this anchor the text is pasted into
        # whatever is focused at injection time (audit C3).
        self._focus_anchor = (0, 0, 0)
        self._start_time = 0.0

        # Rolling end-to-end latency so the Settings page can say what this
        # machine's round trip actually is instead of claiming "latency low"
        # (audit C3: the status text was hardcoded, which is exactly the kind
        # of reassurance that goes stale without anyone noticing).
        self._latency_ms = deque(maxlen=8)

        # Cleanup tone for the Dictation API rewrite (persisted in .env;
        # legacy values like "auto"/"polished" are mapped in default_tone).
        self.tone = default_tone(os.getenv("WHISPRFLOW_TONE", "general"))

        # Hands-free mode. A quick tap of the hotkey locks recording on;
        # holding it is classic push-to-talk. Both gestures use the same
        # keys, so there is nothing extra to learn.
        self.locked = False
        self._press_time = 0.0

        # Command Mode: Ctrl+Shift+Win selects -> speak -> rewrite in place.
        self.command_combo = {keyboard.Key.ctrl_l, keyboard.Key.shift,
                              keyboard.Key.cmd}
        self.command_mode = False
        self._pending_selection = ""

        self.loop = asyncio.new_event_loop()

        # Hotkeys
        self.hotkey_combo = {keyboard.Key.ctrl_l, keyboard.Key.cmd}
        self.pressed_keys = set()
        self.kb_controller = KeyboardController()

        # Window
        self.root = tk.Tk()
        self.root.title("WhisprFlow")
        self.root.geometry("640x720")
        self.root.configure(bg=theme.HEX_BG_APP)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        self._set_window_icon()

        # Tk variables may only be touched on the main thread, so the
        # pipeline reads this plain mirror instead of the BooleanVar.
        self._cleanup_on = True
        self.cleanup_enabled = tk.BooleanVar(value=True)

        self._build_ui()
        self.pill = FloatingPill(
            self.root,
            get_level=self.capture.get_level,
            on_stop=self.stop_recording,
            on_cancel=self.cancel_recording,
            on_retry=self.retry,
        )
        self.root.withdraw()
        self.tray = self._build_tray()

    # ══════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Scrollable container so all cards fit any screen resolution
        canvas = tk.Canvas(self.root, bg=theme.HEX_BG_APP, highlightthickness=0, borderwidth=0)
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        wrap = tk.Frame(canvas, bg=theme.HEX_BG_APP)

        wrap_id = canvas.create_window((0, 0), window=wrap, anchor="nw")

        def _on_wrap_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(wrap_id, width=event.width)

        wrap.bind("<Configure>", _on_wrap_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            if hasattr(self, "log_area"):
                try:
                    if str(event.widget).startswith(str(self.log_area)):
                        return
                except Exception:
                    pass
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self._canvas = canvas

        inner = tk.Frame(wrap, bg=theme.HEX_BG_APP)
        inner.pack(fill="both", expand=True, padx=24, pady=24)

        Label(inner, text="WhisprFlow", font=(theme.UI_FONT, 22, "bold"),
              fg=theme.HEX_TEXT, bg=theme.HEX_BG_APP).pack(anchor="w")
        Label(inner, text="Hold Ctrl + Win to dictate anywhere",
              font=(theme.UI_FONT, 10), fg=theme.HEX_MUTED,
              bg=theme.HEX_BG_APP).pack(anchor="w", pady=(0, 18))

        # ── Engine ──
        card = self._card(inner, "Transcription")
        info = self.stt.get_info()
        self.engine_label = Label(
            card, text=f"AssemblyAI · {info['model']}",
            font=(theme.UI_FONT, 11, "bold"),
            fg=theme.HEX_SUCCESS if info["configured"] else theme.HEX_WARNING,
            bg=theme.HEX_BG_CARD)
        self.engine_label.pack(anchor="w")
        self.engine_sub = Label(
            card,
            text=("Ready" if info["configured"] else "Add an API key to start"),
            font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD)
        self.engine_sub.pack(anchor="w", pady=(1, 12))

        self.aai_entry = self._key_row(card, "AssemblyAI API key",
                                       os.getenv("ASSEMBLYAI_API_KEY", ""),
                                       self.save_assemblyai_key)
        Label(card, text="One key powers everything below. Groq is only "
                         "needed if you use Command Mode.",
              font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED,
              bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(0, 4))
        self.groq_entry = self._key_row(card, "Groq API key (Command Mode only)",
                                        os.getenv("GROQ_API_KEY", ""),
                                        self.save_groq_key)

        # ── Microphone ──
        card_mic = self._card(inner, "Microphone")
        self.mic_status_label = Label(
            card_mic, text="", font=(theme.UI_FONT, 10, "bold"),
            fg=theme.HEX_TEXT, bg=theme.HEX_BG_CARD)
        self.mic_status_label.pack(anchor="w")

        self.mic_sub = Label(
            card_mic, text="", font=(theme.UI_FONT, 9),
            fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD)
        self.mic_sub.pack(anchor="w", pady=(1, 10))

        row_mic = tk.Frame(card_mic, bg=theme.HEX_BG_CARD)
        row_mic.pack(fill="x", pady=(0, 10))

        self.mic_var = tk.StringVar(value="Default (System Default)")
        self.mic_menu = tk.OptionMenu(
            row_mic, self.mic_var, "Default (System Default)",
            command=self._on_device_selected
        )
        self.mic_menu.config(
            bg=theme.HEX_BG_INPUT, fg=theme.HEX_TEXT,
            activebackground=theme.HEX_BG_HOVER, activeforeground=theme.HEX_TEXT,
            relief="flat", highlightthickness=0, borderwidth=0,
            font=(theme.UI_FONT, 9), padx=10, pady=6, cursor="hand2"
        )
        self.mic_menu["menu"].config(
            bg=theme.HEX_BG_CARD, fg=theme.HEX_TEXT,
            activebackground=theme.HEX_BG_HOVER, activeforeground=theme.HEX_TEXT,
            relief="flat", borderwidth=1, font=(theme.UI_FONT, 9)
        )
        self.mic_menu.pack(side="left", fill="x", expand=True, padx=(0, 8))

        self._button(row_mic, "Refresh", self.refresh_devices, subtle=True).pack(side="left")

        # Live Level Meter
        meter_row = tk.Frame(card_mic, bg=theme.HEX_BG_CARD)
        meter_row.pack(fill="x", pady=(2, 0))

        Label(meter_row, text="Input level", font=(theme.UI_FONT, 8, "bold"),
              fg=theme.HEX_FAINT, bg=theme.HEX_BG_CARD).pack(side="left", padx=(0, 8))

        self.meter_canvas = tk.Canvas(
            meter_row, height=8, bg=theme.HEX_BG_INPUT,
            highlightthickness=0, relief="flat")
        self.meter_canvas.pack(side="left", fill="x", expand=True)

        # ── Cleanup tone ──
        card2 = self._card(inner, "Cleanup")
        tk.Checkbutton(
            card2, text="Clean up transcripts (filler removal, punctuation)",
            variable=self.cleanup_enabled, command=self._on_cleanup_toggle,
            bg=theme.HEX_BG_CARD, fg=theme.HEX_TEXT,
            selectcolor=theme.HEX_BG_INPUT, activebackground=theme.HEX_BG_CARD,
            activeforeground=theme.HEX_TEXT, font=(theme.UI_FONT, 10),
            borderwidth=0, highlightthickness=0,
        ).pack(anchor="w")

        tone_row = tk.Frame(card2, bg=theme.HEX_BG_CARD)
        tone_row.pack(fill="x", pady=(8, 2))
        Label(tone_row, text="Tone", font=(theme.UI_FONT, 9, "bold"),
              fg=theme.HEX_FAINT, bg=theme.HEX_BG_CARD).pack(side="left", padx=(0, 8))
        self.tone_var = tk.StringVar(value=TONE_NAMES[self.tone])
        self.tone_menu = tk.OptionMenu(
            tone_row, self.tone_var, *TONE_NAMES.values(),
            command=self._on_tone_selected)
        self.tone_menu.config(
            bg=theme.HEX_BG_INPUT, fg=theme.HEX_TEXT,
            activebackground=theme.HEX_BG_HOVER, activeforeground=theme.HEX_TEXT,
            relief="flat", highlightthickness=0, borderwidth=0,
            font=(theme.UI_FONT, 9), padx=10, pady=5, cursor="hand2")
        self.tone_menu["menu"].config(
            bg=theme.HEX_BG_CARD, fg=theme.HEX_TEXT,
            activebackground=theme.HEX_BG_HOVER, activeforeground=theme.HEX_TEXT,
            relief="flat", borderwidth=1, font=(theme.UI_FONT, 9))
        self.tone_menu.pack(side="left")

        self.refine_status = Label(card2, text="", font=(theme.UI_FONT, 9),
                                   fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD)
        self.refine_status.pack(anchor="w", pady=(4, 0))

        # ── Dictionary ──
        card3 = self._card(inner, "Dictionary")
        self.dict_label = Label(
            card3, text=f"{len(self.dictionary)} terms",
            font=(theme.UI_FONT, 10), fg=theme.HEX_TEXT, bg=theme.HEX_BG_CARD)
        self.dict_label.pack(anchor="w")
        Label(card3, text="Names and jargon the recogniser should never guess at.",
              font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED,
              bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(1, 8))

        row = tk.Frame(card3, bg=theme.HEX_BG_CARD)
        row.pack(fill="x")
        self.dict_entry = self._entry(row)
        self.dict_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self.dict_entry.bind("<Return>", lambda e: self.add_dictionary_term())
        self._button(row, "Add", self.add_dictionary_term).pack(side="left", padx=(0, 6))
        self._button(row, "Open file", self.open_dictionary,
                     subtle=True).pack(side="left")

        # ── Command Mode ──
        cmd = self._card(inner, "Command Mode")
        Label(cmd, text="Select text anywhere, then Ctrl + Shift + Win",
              font=(theme.UI_FONT, 10), fg=theme.HEX_TEXT,
              bg=theme.HEX_BG_CARD).pack(anchor="w")
        Label(cmd,
              text=("Speak an instruction — \u201cmake this formal\u201d, "
                    "\u201cbullet points\u201d, \u201ctranslate to French\u201d."),
              font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED,
              bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(1, 6))
        self.command_status = Label(cmd, text="", font=(theme.UI_FONT, 9),
                                    fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD)
        self.command_status.pack(anchor="w")

        # ── Snippets ──
        snip = self._card(inner, "Snippets")
        self.snippet_label = Label(
            snip, text=f"{len(self.snippets)} triggers",
            font=(theme.UI_FONT, 10), fg=theme.HEX_TEXT, bg=theme.HEX_BG_CARD)
        self.snippet_label.pack(anchor="w")
        Label(snip, text="Say a phrase, get canned text. No AI, no latency.",
              font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED,
              bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(1, 8))
        self._button(snip, "Edit snippets", self.open_snippets,
                     subtle=True).pack(anchor="w")

        # ── Context ──
        card5 = self._card(inner, "Context")
        Label(card5, text="Formatting profiles per app",
              font=(theme.UI_FONT, 10), fg=theme.HEX_TEXT,
              bg=theme.HEX_BG_CARD).pack(anchor="w")
        Label(card5,
              text=(f"{len(self.profiles.profiles)} profiles — the app in "
                    "focus shapes the cleanup (Code, Terminal, Chat\u2026)"),
              font=(theme.UI_FONT, 9), fg=theme.HEX_MUTED, wraplength=280,
              justify="left",
              bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(1, 8))
        self._button(card5, "Edit profiles", self.open_profiles,
                     subtle=True).pack(anchor="w")

        # ── Activity ──
        card4 = self._card(inner, "Activity", expand=False)
        self.log_area = scrolledtext.ScrolledText(
            card4, height=8, font=(theme.MONO_FONT, 9),
            bg=theme.HEX_BG_APP, fg=theme.HEX_TEXT, relief="flat",
            wrap=tk.WORD, borderwidth=0, insertbackground=theme.HEX_TEXT)
        self.log_area.pack(fill="both", expand=True)
        self.log_area.tag_configure("error", foreground=theme.HEX_DANGER)
        self.log_area.tag_configure("warn", foreground=theme.HEX_WARNING)
        self.log_area.tag_configure("ok", foreground=theme.HEX_SUCCESS)
        self.log_area.tag_configure("dim", foreground=theme.HEX_FAINT)
        self.log_area.config(state="disabled")

        self._populate_mic_menu()
        self._refresh_status()

    def _card(self, parent, title, expand=False):
        Label(parent, text=title.upper(), font=(theme.UI_FONT, 8, "bold"),
              fg=theme.HEX_FAINT, bg=theme.HEX_BG_APP).pack(anchor="w", pady=(8, 5))
        frame = tk.Frame(parent, bg=theme.HEX_BG_CARD)
        frame.pack(fill="both" if expand else "x", expand=expand, pady=(0, 6))
        inner = tk.Frame(frame, bg=theme.HEX_BG_CARD)
        inner.pack(fill="both", expand=True, padx=16, pady=14)
        return inner

    def _entry(self, parent, show=None):
        return tk.Entry(
            parent, font=(theme.UI_FONT, 10), bg=theme.HEX_BG_INPUT,
            fg=theme.HEX_TEXT, insertbackground=theme.HEX_TEXT,
            relief="flat", show=show, borderwidth=0, highlightthickness=0)

    def _key_row(self, parent, label, value, command):
        Label(parent, text=label, font=(theme.UI_FONT, 9),
              fg=theme.HEX_MUTED, bg=theme.HEX_BG_CARD).pack(anchor="w", pady=(6, 3))
        row = tk.Frame(parent, bg=theme.HEX_BG_CARD)
        row.pack(fill="x")
        entry = self._entry(row, show="\u2022")
        entry.insert(0, value)
        entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(0, 8))
        self._button(row, "Save", command).pack(side="left")
        return entry

    def _button(self, parent, text, command, subtle=False):
        bg = theme.HEX_BG_INPUT if subtle else theme.HEX_ACCENT
        fg = theme.HEX_TEXT if subtle else "#0d1220"
        b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                      font=(theme.UI_FONT, 9, "bold"), relief="flat",
                      padx=14, pady=6, cursor="hand2",
                      activebackground=theme.HEX_BG_HOVER,
                      activeforeground=theme.HEX_TEXT,
                      borderwidth=0, highlightthickness=0)
        hover = theme.HEX_BG_HOVER if subtle else "#93b4ff"
        b.bind("<Enter>", lambda e: b.config(bg=hover))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    def _set_window_icon(self):
        try:
            ico = resource_path("assets", "icon.ico")
            if ico.exists():
                self.root.iconbitmap(default=str(ico))
        except Exception:
            pass  # cosmetic only

    def _tray_image(self):
        try:
            png = resource_path("assets", "icon.png")
            if png.exists():
                return Image.open(png).convert("RGBA").resize((64, 64))
        except Exception:
            pass
        img = Image.new("RGB", (64, 64), theme.BG_APP)
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([14, 26, 50, 38], radius=6, fill=theme.ACCENT)
        return img

    def _build_tray(self):
        img = self._tray_image()
        menu = pystray.Menu(
            pystray.MenuItem("Open WhisprFlow", self.show_window, default=True),
            pystray.MenuItem(lambda text: f"Mic: {self._get_mic_summary()}", lambda: self.show_window()),
            pystray.MenuItem("Dictionary", lambda: self.open_dictionary()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit_app),
        )
        return pystray.Icon("WhisprFlow", img, "WhisprFlow", menu)

    # ══════════════════════════════════════════════════════════════════
    #  Settings
    # ══════════════════════════════════════════════════════════════════

    def _persist(self, key: str, value: str) -> None:
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not ENV_PATH.exists():
            ENV_PATH.touch()
        set_key(str(ENV_PATH), key, value)
        os.environ[key] = value

    def save_assemblyai_key(self):
        key = self.aai_entry.get().strip()
        if not key:
            messagebox.showwarning("WhisprFlow", "Enter an AssemblyAI API key.")
            return
        self._persist("ASSEMBLYAI_API_KEY", key)
        self.stt.set_api_key(key)
        self._refresh_status()
        # "If your key is valid, you get a 200 ... If it's missing or wrong,
        # you get a 401. That's your authentication smoke test before you send
        # any audio." -- the save button is the only moment where a wrong key
        # can be reported before it costs the user a dictation.
        asyncio.run_coroutine_threadsafe(self._check_assemblyai_key(), self.loop)

    async def _check_assemblyai_key(self):
        try:
            ok, msg = await self.stt.validate_key()
        except Exception as e:
            ok, msg = None, f"could not check: {e}"
        if ok:
            self.log(f"AssemblyAI key saved. {msg}", "ok")
            self._set_status_text(self.engine_sub, "Key verified \u00b7 " + msg)
            await self.stt.warm()
        elif ok is None:
            self.log("AssemblyAI key saved, but it could not be verified offline.", "warn")
            self._set_status_text(self.engine_sub, "Saved \u00b7 not verified: " + msg)
        else:
            self.log(f"AssemblyAI key rejected: {msg}", "error")
            self._set_status_text(self.engine_sub, "Key rejected: " + msg)
            self.after_ui(lambda: messagebox.showwarning(
                "WhisprFlow",
                "AssemblyAI rejected that key.\n\n" + msg +
                "\n\nNothing was sent for transcription."))

    def _set_status_text(self, widget, text):
        """Apply a label change from the asyncio thread safely."""
        self._ui(lambda: widget.config(text=text))

    def save_groq_key(self):
        key = self.groq_entry.get().strip()
        if not key:
            messagebox.showwarning("WhisprFlow", "Enter a Groq API key.")
            return
        self._persist("GROQ_API_KEY", key)
        self.commands.set_api_key(key)
        self.log("Groq key saved (Command Mode).", "ok")
        self._refresh_status()

    def add_dictionary_term(self):
        term = self.dict_entry.get().strip()
        if not term:
            return
        if self.dictionary.add(term):
            self.log(f"Added \u201c{term}\u201d to dictionary.", "ok")
            self.dict_entry.delete(0, tk.END)
            self._refresh_status()
        else:
            self.log(f"\u201c{term}\u201d not added (duplicate or too long).", "warn")

    def open_snippets(self):
        try:
            self.snippets.save()
            os.startfile(str(self.snippets.path))
        except Exception as e:
            self.log(f"Could not open snippets: {e}", "error")

    def reload_snippets(self):
        self.snippets.load()
        self._refresh_status()

    def open_profiles(self):
        try:
            os.startfile(str(self.profiles.path))
        except Exception as e:
            self.log(f"Could not open profiles: {e}", "error")

    def open_dictionary(self):
        try:
            os.startfile(str(self.dictionary.path))
        except Exception as e:
            self.log(f"Could not open dictionary: {e}", "error")

    def _on_cleanup_toggle(self):
        self._cleanup_on = bool(self.cleanup_enabled.get())
        self.log("Cleanup " + ("enabled." if self.cleanup_enabled.get()
                                 else "disabled — verbatim transcript only."))
        self._refresh_status()

    def _on_tone_selected(self, label):
        inv = {v: k for k, v in TONE_NAMES.items()}
        self.tone = inv.get(label, "general")
        self._persist("WHISPRFLOW_TONE", self.tone)
        self.log(f"Cleanup tone: {label}.")

    def _refresh_status(self):
        info = self.stt.get_info()
        self.engine_label.config(
            text=f"AssemblyAI · {info['model']}",
            fg=theme.HEX_SUCCESS if info["configured"] else theme.HEX_WARNING)
        if not info["configured"]:
            sub = "Add an API key to start"
        else:
            # Saying what happened on the last take is the difference
            # between "it feels slow" being diagnosable and not (audit C12).
            bits = [f"{info.get('requests', 0)} takes"]
            if info.get("fallbacks"):
                bits.append(f"{info['fallbacks']} used local cleanup")
            if info.get("last_request_ms"):
                bits.append(f"server {info['last_request_ms']:.0f} ms")
            if info.get("last_error"):
                bits.append(f"last error: {info['last_error']}")
            if info.get("last_session_id"):
                bits.append(f"request {info['last_session_id']}")
            sub = "\n".join(bits)
        self.engine_sub.config(text=sub, wraplength=280)
        self._key_state = info["configured"]

        if hasattr(self, "mic_status_label"):
            if not self.capture.is_running:
                self.mic_status_label.config(
                    text="Microphone Unavailable", fg=theme.HEX_DANGER)
                self.mic_sub.config(
                    text=self.capture.last_error or "Check microphone permissions or select another device")
            else:
                curr = self.capture.get_current_device_info()
                is_def = (self.capture.device is None)
                def_tag = " (System Default)" if is_def else ""
                name_str = curr.name if curr else "Default Microphone"
                self.mic_status_label.config(
                    text=f"{name_str}{def_tag}", fg=theme.HEX_SUCCESS)
                if self._latency_ms:
                    avg = int(sum(self._latency_ms) / len(self._latency_ms))
                    lat = f"last {self._latency_ms[-1]} ms · avg {avg} ms"
                else:
                    lat = "no dictation yet this session"
                self.mic_sub.config(
                    text=f"16 kHz mono · ready · {lat}")

        if not self.cleanup_enabled.get():
            txt, col = "Off — verbatim transcript", theme.HEX_MUTED
        else:
            # If the server rewrite fails, the pipeline degrades to local
            # cleanup and logs it — silence about degradations was audit C1.
            if info.get("last_error"):
                txt, col = f"Local cleanup: {info['last_error']}", theme.HEX_WARNING
            else:
                txt, col = f"On · tone: {TONE_NAMES[self.tone]}", theme.HEX_SUCCESS
        self.refine_status.config(text=txt, fg=col, wraplength=280)
        self.dict_label.config(text=f"{len(self.dictionary)} terms")
        self.snippet_label.config(text=f"{len(self.snippets)} triggers")

        cs = self.commands.get_stats()
        if not cs["configured"]:
            self.command_status.config(text="Needs a Groq key", fg=theme.HEX_WARNING)
        elif cs["invocations"]:
            self.command_status.config(
                text=f"{cs['invocations']} used \u00b7 {cs['rejections']} rejected",
                fg=theme.HEX_MUTED)
        else:
            self.command_status.config(text=f"Ready ({cs['model']})",
                                       fg=theme.HEX_SUCCESS)

    def _populate_mic_menu(self):
        devices = self.capture.list_devices()
        self._device_map = {}
        default_label = "Default (System Default)"
        self._device_map[default_label] = (None, "default")

        seen_labels = {default_label}
        for d in devices:
            label = d.label
            if label in seen_labels:
                label = f"{d.label} [#{d.index}]"
            seen_labels.add(label)
            self._device_map[label] = (d.index, d.label)

        menu = self.mic_menu["menu"]
        menu.delete(0, "end")
        for label in self._device_map.keys():
            menu.add_command(
                label=label,
                command=lambda val=label: self._on_device_selected(val)
            )

        active_label = default_label
        if self.capture.device is not None:
            for lbl, (idx, _) in self._device_map.items():
                if idx == self.capture.device:
                    active_label = lbl
                    break
        self.mic_var.set(active_label)

    def _on_device_selected(self, chosen_label: str):
        if chosen_label not in self._device_map:
            return
        idx, identifier = self._device_map[chosen_label]
        if idx == self.capture.device:
            self.mic_var.set(chosen_label)
            return

        self.mic_var.set(chosen_label)
        self.log(f"Switching microphone to: {chosen_label}\u2026")
        ok = self.capture.set_device(idx)
        if ok:
            self._persist("WHISPRFLOW_MIC_DEVICE", identifier)
            self.log(f"Microphone set to {chosen_label}.", "ok")
        else:
            err = self.capture.last_error or "Failed to open device"
            self.log(f"Could not switch microphone: {err}", "error")
            messagebox.showerror("Microphone Error", f"Could not open {chosen_label}:\n\n{err}")
            # Revert dropdown selection to whatever capture ended up on
            active_label = "Default (System Default)"
            for lbl, (d_idx, _) in self._device_map.items():
                if d_idx == self.capture.device:
                    active_label = lbl
                    break
            self.mic_var.set(active_label)
        self._refresh_status()

    def refresh_devices(self):
        self._populate_mic_menu()
        self._refresh_status()
        self.log(f"Audio devices refreshed ({len(self._device_map) - 1} input devices detected).", "dim")

    def _draw_meter(self, level: float):
        if not hasattr(self, "meter_canvas"):
            return
        try:
            w = self.meter_canvas.winfo_width()
            h = self.meter_canvas.winfo_height()
            if w <= 1 or h <= 1:
                return
            self.meter_canvas.delete("all")
            # Scaled for responsive audio meter visualization
            scaled = min(1.0, max(0.0, level * 3.0))
            if scaled > 0.01:
                fill_w = max(3, int(w * scaled))
                color = theme.HEX_SUCCESS if scaled > 0.06 else theme.HEX_ACCENT
                self.meter_canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        except Exception:
            pass

    def _update_meter(self):
        if not self._meter_running:
            return
        try:
            if self.root.winfo_viewable():
                level = self.capture.get_level() if self.capture.is_running else 0.0
                self._draw_meter(level)
                self.root.after(40, self._update_meter)
            else:
                self._meter_running = False
        except Exception:
            self._meter_running = False

    def _get_mic_summary(self) -> str:
        if not self.capture.is_running:
            return "Unavailable"
        curr = self.capture.get_current_device_info()
        if not curr:
            return "Default"
        name = curr.name
        return name[:25] + "\u2026" if len(name) > 25 else name

    # ══════════════════════════════════════════════════════════════════
    #  Logging
    # ══════════════════════════════════════════════════════════════════

    # ── thread-safe UI calls ──────────────────────────────────────────
    #
    # Tkinter is not thread-safe: every widget call must happen on the
    # thread running mainloop(). The pipeline runs on the asyncio thread,
    # so all pill/window updates are marshalled through root.after(0, ...).
    # Calling directly raised "main thread is not in main loop".

    def _ui(self, fn, *args):
        try:
            self.root.after(0, lambda: _safe_call(fn, *args))
        except Exception:
            pass

    def pill_state(self, state, status: str = ""):
        self._ui(self.pill.set_state, state, status)

    def pill_success(self, message: str = ""):
        self._ui(self.pill.flash_success, message)

    def pill_error(self, message: str = ""):
        self._ui(self.pill.flash_error, message)

    def pill_locked(self, locked: bool):
        self._ui(self.pill.set_locked, locked)

    def pill_command(self, active: bool):
        self._ui(self.pill.set_command_mode, active)

    def refresh_status(self):
        self._ui(self._refresh_status)

    def log(self, message: str, tag: str = "dim"):
        try:
            self.root.after(0, self._log_ui, message, tag)
        except Exception:
            logger.info(message)

    def _log_ui(self, message: str, tag: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, f"{ts}  ", "dim")
        self.log_area.insert(tk.END, f"{message}\n", tag)
        # Keep the buffer bounded; transcripts are sensitive and unbounded
        # growth was a slow leak in the old version.
        if int(self.log_area.index("end-1c").split(".")[0]) > 300:
            self.log_area.delete("1.0", "100.0")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    # ══════════════════════════════════════════════════════════════════
    #  Recording
    # ══════════════════════════════════════════════════════════════════

    def start_recording(self):
        with self._state_lock:
            if self.is_recording:
                return
            if not self.capture.is_running:
                self.log("Microphone unavailable.", "error")
                self.pill_error("No microphone")
                return
            self.is_recording = True
            self.generation += 1
            gen = self.generation

        self._start_time = time.monotonic()
        self.capture.begin()
        beep_async(880, 60)
        self.pill_state(PillState.RECORDING)

        # Resolve the formatting profile now — while the user speaks,
        # never in the 0.5 s after release where it would sit on the
        # critical path.
        self._grab_context()

        if self.stt.is_configured and not self.command_mode:
            # Complete DNS/TCP/TLS (+H2) while audio accumulates; by the
            # time the key comes up the POST starts sending immediately.
            asyncio.run_coroutine_threadsafe(self.stt.warm(), self.loop)

        threading.Thread(target=self._watch_duration, args=(gen,),
                         daemon=True).start()

    def _watch_duration(self, gen: int):
        """Stop automatically at the buffer cap.

        A locked recording the user forgets about would otherwise keep
        going until the ring silently truncated the start of it.
        """
        # The Dictation API rejects clips over 120 s (the request fails
        # after billing nothing), so recording stops ahead of that rather
        # than at the local memory cap.
        limit = min(self.capture.max_seconds, 110)
        warned = False
        while True:
            time.sleep(0.5)
            with self._state_lock:
                if gen != self.generation or not self.is_recording:
                    return
            elapsed = time.monotonic() - self._start_time
            if not warned and elapsed > limit - 30:
                warned = True
                self.log(f"30 seconds left of the {limit / 60:.0f} minute limit.",
                         "warn")
            if elapsed >= limit:
                self.log("Reached the take limit — transcribing what you have.",
                         "warn")
                self.stop_recording()
                return

    def _grab_context(self):
        """Resolve the app-specific profile for this take (fast, sync).

        The old UI-Automation reader ran un-CoInitialize'd on a fresh
        thread, threw on most apps, and fed a prompt field that had to be
        kept field-content-free. The foreground process name is one
        syscall and is all the profiles ever used.
        """
        try:
            process = _foreground_process()
        except Exception:
            process = ""
        self._live_process = process or ""
        profile = self.profiles.resolve(self._live_process)
        self._profile_instruction = profile.instruction if profile else ""
        if DEBUG and profile and profile.name != "Default":
            self.log(f"[profile: {profile.name}]")

    def _stt_prompt(self) -> str:
        """A short description of the audio for the Dictation `stt_prompt`
        field. It REPLACES the managed default prompt, so keep it purely
        descriptive of the audio -- never instruction-like, never field
        content."""
        bits = ["A single speaker dictating into a Windows app"]
        if self._live_process:
            bits.append(f"(in {self._live_process})")
        return " ".join(bits)

    def start_command_mode(self):
        """Grab the selection, then record the instruction to apply to it."""
        if not self.commands.is_configured:
            self.log("Command Mode needs a Groq API key.", "error")
            self.pill_error("Add a Groq key")
            return

        sel = self.selection.capture()
        if not sel.ok or sel.is_empty:
            self.log(f"Command Mode: {sel.error or 'nothing selected'}", "warn")
            self.pill_error(sel.error or "Select text first")
            return

        with self._state_lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.command_mode = True
            self.generation += 1
            gen = self.generation

        self._pending_selection = sel.text
        self._start_time = time.monotonic()
        # on_release only acts when _press_time is set. Without this the
        # keys could be released and nothing would happen -- Command Mode
        # would record until the duration cap.
        self._press_time = time.monotonic()
        self.capture.begin()
        beep_async(1320, 55)

        preview = sel.text.strip().replace("\n", " ")
        self.log(f"Command Mode — {len(sel.text)} chars selected: "
                 f"{preview[:50]}{'…' if len(preview) > 50 else ''}")
        self.pill_state(PillState.RECORDING, "command")
        self.pill_command(True)

        threading.Thread(target=self._grab_context, daemon=True).start()
        threading.Thread(target=self._watch_duration, args=(gen,),
                         daemon=True).start()

    async def _run_command(self, audio, gen: int, selected: str):
        """Transcribe the spoken instruction and apply it to the selection."""
        try:
            cleaned = await asyncio.to_thread(
                process_audio, audio, self.capture.sample_rate)
            if self._stale(gen):
                return

            # Verbatim transcription of the spoken instruction — the command
            # must be heard exactly, with no cleanup rewrite second-guessing
            # it (and no llm_instruction sent, so none is run).
            result = await self.stt.transcribe(
                cleaned.samples, cleaned.sample_rate,
                keyterms=self.dictionary.as_keyterms())
            if self._stale(gen):
                return

            if not result.ok:
                self.log(f"Command failed: {result.error}", "error")
                self.pill_error(_short_error(result.error))
                return

            instruction = result.text.strip()
            if not instruction:
                self.log("No instruction heard.", "warn")
                self.pill_state(PillState.IDLE)
                return

            self.log(f"Command: \u201c{instruction}\u201d")

            outcome = await self.commands.apply(
                selected, instruction,
                dictionary_terms=self.dictionary.as_keyterms(),
                app_context=self._live_process)

            if self._stale(gen):
                return

            if not outcome.ok:
                self.log(f"Command rejected: {outcome.error}", "error")
                self.pill_error(_preview(outcome.error or "Failed", 22))
                return

            if not await self._anchor_ok(outcome.text, require_selection=selected):
                return

            # Still paste rather than type: the selection has to be replaced
            # in one operation, and `replace()` restores the clipboard.
            if not await asyncio.to_thread(self.selection.replace, outcome.text):
                self.last_text_injected = ''
                self.last_failed = True
                self.log('Could not replace the selection.', 'error')
                self.pill_error('Replace failed')
                self.pill_state(PillState.IDLE)
                return
            self.last_text_injected = outcome.text

            self.log(f"\u2713 {_preview(outcome.text)}", "ok")
            self.pill_success(_preview(outcome.text, 22))
            self.refresh_status()

        except Exception as e:
            self.log(f"Command failed: {e}", "error")
            self.pill_error("Something went wrong")
            if DEBUG:
                traceback.print_exc()

    def stop_recording(self):
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.locked = False
            was_command = self.command_mode
            self.command_mode = False
            gen = self.generation
            self._focus_anchor = _focus_signature()
        self.pill_locked(False)
        self.pill_command(False)

        beep_async(660, 60)
        audio = self.capture.end()
        duration = time.monotonic() - self._start_time

        if audio is None or duration < 0.25:
            self.pill_state(PillState.IDLE)
            self.log("Too short.", "warn")
            return

        self.pill_state(PillState.PROCESSING)

        if was_command:
            selected, self._pending_selection = self._pending_selection, ""
            asyncio.run_coroutine_threadsafe(
                self._run_command(audio, gen, selected), self.loop)
            return

        asyncio.run_coroutine_threadsafe(self._pipeline(audio, gen), self.loop)

    def cancel_recording(self):
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self.locked = False
            self.command_mode = False
            self._pending_selection = ""
            self.generation += 1     # invalidates any in-flight pipeline

        self.capture.discard()
        beep_async(440, 90)
        self.pill_state(PillState.IDLE)
        self.log("Cancelled.", "dim")

    def _stale(self, gen: int) -> bool:
        """True if the user cancelled or started a new take since `gen`."""
        with self._state_lock:
            return gen != self.generation

    # ══════════════════════════════════════════════════════════════════
    #  Pipeline
    # ══════════════════════════════════════════════════════════════════

    async def _pipeline(self, audio, gen: int):
        try:
            cleaned = await asyncio.to_thread(process_audio, audio, self.capture.sample_rate)
            if self._stale(gen):
                return

            self.last_audio = (cleaned.samples, cleaned.sample_rate)

            # A local silence gate saves a billable round trip; catching
            # coughs and key bumps here costs nothing.
            if not cleaned.speech_detected:
                self.log("No speech detected.", "warn")
                self.pill_state(PillState.IDLE)
                return

            # One request: verbatim text AND the cleaned version come back
            # together. The instruction carries the cleanup rules for this
            # take (tone + app profile); it is skipped on very short takes
            # where the server rewrite is blunt anyway.
            want_cleanup = bool(self._cleanup_on)
            instruction = ""
            if want_cleanup:
                instruction = build_instruction(self.tone, self._profile_instruction)

            result = await self.stt.transcribe(
                cleaned.samples, cleaned.sample_rate,
                instruction=instruction,
                prompt=self._stt_prompt(),
                keyterms=self.dictionary.as_keyterms(),
            )
            if self._stale(gen) or result is None:
                return

            if result.latency_ms:
                self._latency_ms.append(int(result.latency_ms))

            if not result.ok:
                self.last_failed = True
                self.log(f"Transcription failed: {result.error}", "error")
                self.pill_error(_short_error(result.error))
                return

            raw = result.text
            raw, used = self.snippets.expand(raw)
            if used:
                self.log(f"Snippet: {', '.join(used)}")
            self.last_raw_text = raw

            if not raw.strip():
                self.log("Nothing transcribed.", "warn")
                self.pill_state(PillState.IDLE)
                return

            if DEBUG:
                self.log(f"[dictation {result.latency_ms}ms "
                         f"conf={result.confidence:.2f} kind={result.kind}] {raw}")

            final = self._choose_final(raw, result, snippets_used=bool(used))
            if self._stale(gen):
                return

            if not await self._anchor_ok(final):
                return
            if not await self._inject(final):
                return

            self.last_failed = False
            self.log(_preview(final), "ok")
            self.pill_success(_preview(final, 22))
            self.refresh_status()

        except Exception as e:
            self.last_failed = True
            self.log(f"Failed: {e}", "error")
            self.pill_error("Something went wrong")
            if DEBUG:
                traceback.print_exc()

    def _choose_final(self, raw: str, result, snippets_used: bool) -> str:
        """Decide what gets injected: the server's rewrite when it exists,
        the deterministic local cleanup otherwise.

        Degradations are logged visibly -- the audit showed a quiet ✓ on a
        degraded take is worse than an honest warning (audit C1/C3).
        """
        if not self._cleanup_on:
            return raw

        if result.kind == "enhanced" and not snippets_used:
            return result.clean
        if result.kind == "fallback":
            # Server-side rewrite failed (llm_error). Local cleanup is the
            # floor, and the user hears about it rather than seeing ✓.
            self.log("Server cleanup failed (%s) — applied local cleanup."
                     % (result.llm_error or "error"), "warn")
        return basic_cleanup(raw)

    async def _anchor_ok(self, text: str, require_selection: str = "") -> bool:
        """Is it still safe to write, and if not, salvage the text?

        Two things have to be true after the 1-3 s the pipeline takes: the
        window the user was dictating into is still focused, and -- in Command
        Mode -- the selection we were given is still selected. Refusing is
        cheap here because `stage()` leaves the text on the clipboard, so
        "nothing was pasted" never means "the dictation is lost" (audit C3).
        """
        anchor = self._focus_anchor
        if anchor[0]:
            current = _focus_signature()
            moved = bool(current[0]) and current[0] != anchor[0]
            if _foreground_is_ours(current):
                moved = False
            # A different control in the same window is only a mismatch when
            # both sides actually reported one; 0 means unknown (docs, above).
            reentered = (not moved and anchor[1] and current[1]
                         and current[1] != anchor[1])
            if moved or reentered:
                self.log('Focus moved to a different control'
                         if reentered else 'Focus moved to another window', 'dim')
                staged = await asyncio.to_thread(self.injector.stage, text)
                self.last_text_injected = ''      # nothing was written: no undo
                self.last_failed = True
                self.log('Focus changed while processing — '
                         + ('text is on the clipboard, paste it yourself.'
                            if staged else
                            'nothing was pasted, and it could not be copied either.'),
                         'warn')
                self.pill_error('Focus changed' if not staged else 'Copied — paste it')
                self.pill_state(PillState.IDLE)
                return False

        if require_selection:
            # A click, an arrow key or an Escape in the target app drops the
            # highlight while the LLM runs; pasting then *inserts* the rewrite
            # beside the original instead of replacing it.
            now = await asyncio.to_thread(self.selection.capture)
            if not now.ok or now.text != require_selection:
                staged = await asyncio.to_thread(self.injector.stage, text)
                self.last_text_injected = ''
                self.last_failed = True
                self.log('The selection changed — nothing was replaced'
                         + ('; the rewrite is on the clipboard.' if staged else '.'),
                         'warn')
                self.pill_error('Selection changed')
                self.pill_state(PillState.IDLE)
                return False

        return True

    async def _inject(self, text: str) -> bool:
        """Inject, and believe the result.

        `inject()` returns a bool that main.py used to discard, so a failed
        clipboard write or paste still set `last_text_injected` and the next
        Ctrl+Alt+Z undid whatever the user had really done last.
        """
        if not await asyncio.to_thread(self.injector.inject, text):
            self.last_text_injected = ''
            self.last_failed = True
            self.log('Injection failed — the paste did not happen.', 'error')
            self.pill_error('Paste failed')
            self.pill_state(PillState.IDLE)
            return False
        self.last_text_injected = text
        self.last_failed = False
        return True

    def retry(self):
        if self.last_audio is None:
            self.log("Nothing to retry.", "dim")
            return
        with self._state_lock:
            self.generation += 1
            gen = self.generation
        samples, sr = self.last_audio
        self.pill_state(PillState.PROCESSING)
        self.log("Retrying\u2026")
        asyncio.run_coroutine_threadsafe(self._retry_from(samples, sr, gen), self.loop)

    async def _retry_from(self, samples, sr, gen):
        want_cleanup = bool(self._cleanup_on)
        instruction = ""
        if want_cleanup:
            instruction = build_instruction(self.tone, self._profile_instruction)
        result = await self.stt.transcribe(
            samples, sr,
            instruction=instruction,
            prompt=self._stt_prompt(),
            keyterms=self.dictionary.as_keyterms())
        if self._stale(gen):
            return
        if not result.ok:
            self.log(f"Retry failed: {result.error}", "error")
            self.pill_error(_short_error(result.error))
            return
        raw, used = self.snippets.expand(result.text)
        final = self._choose_final(raw, result, snippets_used=bool(used))
        if self._stale(gen):
            return
        # The old retry bypassed the focus anchor (audit M-list): same
        # safety as the normal path.
        if not await self._anchor_ok(final):
            return
        if not await self._inject(final):
            return
        self.log(_preview(final), "ok")
        self.pill_success(_preview(final, 22))

    # ══════════════════════════════════════════════════════════════════
    #  Hotkeys
    # ══════════════════════════════════════════════════════════════════

    # Below this, a press counts as a "tap" (lock on) rather than a hold.
    TAP_SECONDS = 0.35

    _KEY_ALIASES = None

    @classmethod
    def _key_aliases(cls) -> dict:
        """Right-hand modifiers mapped onto their left-hand twins.

        Built with getattr and cached, because which members exist is a
        property of the pynput backend (and of test stubs), not of this app.
        In pynput `Key.ctrl is Key.ctrl_l` already, so only the _r variants
        need folding; a missing member is simply not aliased.
        """
        if cls._KEY_ALIASES is None:
            K = keyboard.Key
            pairs = (("ctrl_r", "ctrl_l"), ("cmd_r", "cmd"),
                     ("shift_r", "shift_l"), ("alt_r", "alt_l"))
            alias = {}
            for right, left in pairs:
                r = getattr(K, right, None)
                l = getattr(K, left, None) or getattr(K, right[:-2], None)
                if r is not None and l is not None and r is not l:
                    alias[r] = l
            cls._KEY_ALIASES = alias
        return cls._KEY_ALIASES

    @classmethod
    def _canonical_key(cls, key):
        """Fold right-hand modifiers onto their left-hand twins.

        `hotkey_combo` is {ctrl_l, cmd}, so a user whose hands are on the
        right Ctrl, or who uses cmd_r, could never trigger the app at all --
        measured against the real methods: ctrl_r+cmd and ctrl_l+cmd_r both
        do nothing. Folding keeps the *exact-set* semantics that stop
        Ctrl+Win+D from firing a phantom take, while accepting either side of
        the keyboard.
        """
        return cls._key_aliases().get(key, key)

    def on_press(self, key):
        key = self._canonical_key(key)
        # Esc cancels a recording -- with hands free, reaching for the pill
        # with the mouse is the wrong reflex. Handled *before* the key joins
        # pressed_keys, so a stray Esc can never pollute the set and break the
        # exact-match hotkey test below (this is what the audit's C4 claimed
        # was missing; the guard was already correct, see the retraction in
        # AUDIT_PERF_UI_ACCURACY_2026-09-13.md). It used to require `locked`,
        # which meant the panic key did nothing during an ordinary hold.
        if key == keyboard.Key.esc:
            if self.is_recording:
                self.cancel_recording()
            return

        self.pressed_keys.add(key)

        # Command Mode: capture the selection first, then record an
        # instruction to apply to it.
        if self.pressed_keys == self.command_combo and not self.is_recording:
            self.start_command_mode()
            return

        # Exact match only. `issubset` meant Ctrl+Win+D (new desktop) and
        # Ctrl+Win+arrows all started phantom recordings.
        if self.pressed_keys != self.hotkey_combo:
            return

        if self.locked:
            # Already hands-free: this press is the user finishing up.
            self._press_time = 0.0
            self.stop_recording()
        elif not self.is_recording:
            self._press_time = time.monotonic()
            self.start_recording()

    def on_release(self, key):
        key = self._canonical_key(key)
        was_hotkey = key in self.hotkey_combo or key in self.command_combo
        self.pressed_keys.discard(key)
        if not (was_hotkey and self.is_recording and self._press_time):
            return

        # Anything still held that is not part of either combo means the
        # chord was really Ctrl+Win+<something>. That is Windows' own
        # vocabulary -- Ctrl+Win+D makes a new desktop, Ctrl+Win+arrows switch
        # between them -- and the user was not dictating. Starting on such a
        # chord is already refused; the *stop* was not, so the system chime and
        # whatever the user said next became a take and got pasted (audit C11).
        # Cancel, do not transcribe.
        combo = self.command_combo if self.command_mode else self.hotkey_combo
        extras = {k for k in (self.pressed_keys - combo) if k != keyboard.Key.esc}
        if extras:
            self.cancel_recording()
            self.log('Ignored: ' + ', '.join(getattr(k, 'name', str(k)) for k in extras)
                     + ' was held, so this was not a dictation.', 'dim')
            return

        held = time.monotonic() - self._press_time
        self._press_time = 0.0

        if held < self.TAP_SECONDS:
            # Quick tap -> stay recording until they tap again.
            self.locked = True
            self.log("Locked — tap Ctrl + Win again to finish, Esc to cancel.", "ok")
            self.pill_locked(True)
            beep_async(1180, 45)
        else:
            self.stop_recording()

    def undo(self):
        """Ctrl+Z is safer than replaying N backspaces: most apps coalesce a
        paste into one undo step, and blind backspaces eat real text when
        the caret has moved."""
        if not self.last_text_injected:
            self.log("Nothing to undo.", "dim")
            return
        self.kb_controller.press(keyboard.Key.ctrl)
        self.kb_controller.press("z")
        self.kb_controller.release("z")
        self.kb_controller.release(keyboard.Key.ctrl)
        self.last_text_injected = ""
        self.log("Undo sent.", "dim")

    # ══════════════════════════════════════════════════════════════════
    #  Window / lifecycle
    # ══════════════════════════════════════════════════════════════════

    def hide_window(self):
        self._meter_running = False
        self.root.withdraw()

    def show_window(self, icon=None, item=None):
        def _show():
            self.root.deiconify()
            self.root.lift()
            if not self._meter_running:
                self._meter_running = True
                self._update_meter()
        self.root.after(0, _show)

    def quit_app(self, icon=None, item=None):
        self._meter_running = False
        self.log("Shutting down\u2026")
        try:
            self.tray.stop()
        except Exception:
            pass
        self.capture.stop_stream()
        for coro in (self.stt.close(), self.commands.close()):
            try:
                asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=3)
            except Exception:
                pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        try:
            self.pill.destroy()
            self.root.quit()
        except Exception:
            pass
        os._exit(0)

    def run(self):
        threading.Thread(
            target=lambda: (asyncio.set_event_loop(self.loop), self.loop.run_forever()),
            daemon=True, name="asyncio").start()

        if self.capture.start_stream():
            curr = self.capture.get_current_device_info()
            dev_name = curr.name if curr else "Default"
            self.log(f"Microphone ready: {dev_name}.", "ok")
        else:
            self.log(f"Microphone failed: {self.capture.last_error}", "error")

        # Pre-warm the TLS session so the first dictation doesn't pay for it.
        asyncio.run_coroutine_threadsafe(self.stt.warm(), self.loop)

        keyboard.Listener(on_press=self.on_press, on_release=self.on_release).start()
        keyboard.GlobalHotKeys({"<ctrl>+<alt>+z": self.undo}).start()
        threading.Thread(target=self.tray.run, daemon=True, name="tray").start()

        if not self.stt.is_configured:
            self.log("No AssemblyAI key — open settings to add one.", "warn")
            self.root.after(400, self.show_window)
        else:
            self.log("Ready. Hold Ctrl + Win to dictate.", "ok")

        self.root.mainloop()


def _safe_call(fn, *args):
    """Run a UI callback, swallowing errors from a torn-down window."""
    try:
        fn(*args)
    except Exception:
        pass


def _preview(text: str, limit: int = 60) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _short_error(error: str) -> str:
    e = (error or "").lower()
    if "key" in e or "401" in e or "unauthor" in e:
        return "Check API key"
    if "timed out" in e or "timeout" in e:
        return "Timed out"
    if "network" in e or "connect" in e:
        return "No connection"
    if "insufficient" in e or "credit" in e:
        return "Out of credit"
    return "Failed \u21ba"


def _fatal(title: str, message: str) -> None:
    """Show a readable error instead of PyInstaller's raw traceback dialog.

    A frozen windowed app has no console, so an unhandled exception
    surfaces as an unreadable wall of text. Worse, that dialog keeps the
    process alive -- which is how a broken v1.0.0 passed its own CI check.
    """
    sys.stderr.write(f"{title}\n\n{message}\n")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        pass


def _check_runtime() -> None:
    """Verify the compiled dependencies loaded before touching the UI."""
    try:
        import numpy as np
        np.zeros(8, dtype=np.float32).sum()
    except Exception as e:
        _fatal(
            "WhisprFlow — missing components",
            "NumPy failed to load, so this build is incomplete.\n\n"
            f"{type(e).__name__}: {e}\n\n"
            "Please report this at:\n"
            "https://github.com/mabeldesign008-code/flow/issues",
        )
        raise SystemExit(1)

    try:
        import sounddevice as _sd  # noqa: F401
        _ = _sd.__name__
    except Exception as e:
        _fatal(
            "WhisprFlow — audio unavailable",
            "The audio library could not be loaded.\n\n"
            f"{type(e).__name__}: {e}\n\n"
            "On Windows this usually means the PortAudio DLL is missing "
            "from the build. Please report this at:\n"
            "https://github.com/mabeldesign008-code/flow/issues",
        )
        raise SystemExit(1)


if __name__ == "__main__":
    _check_runtime()
    try:
        app = WhisprFlowApp()
        app.run()
    except KeyboardInterrupt:
        os._exit(0)
    except Exception as exc:
        logger.exception("Fatal error during startup")
        _fatal(
            "WhisprFlow — startup failed",
            f"{type(exc).__name__}: {exc}\n\n"
            + "".join(traceback.format_exc().splitlines(True)[-6:])
            + "\nPlease report this at:\n"
            "https://github.com/mabeldesign008-code/flow/issues",
        )
        raise SystemExit(1)

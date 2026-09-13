"""Hotkey state machine and paste-safety tests.

These drive main.py's *real* on_press / on_release / _anchor_ok / _inject on a
bare WhisprFlowApp instance, with only the peripheral libraries stubbed out.
That matters: the audit's C4 was written from a probe that re-implemented the
methods instead of calling them, and the bug it described does not exist. A
test that fakes the code under test proves nothing, so the stubs here stop at
the module boundary (keyboard, audio, clipboard, tray) and the logic under
test is the shipped logic.

Covers: C4 (retracted -- Esc must not pollute pressed_keys, and does not),
C11 (extra keys held at release), U6 (right-side modifiers never fired),
C3 (focus anchor, selection anchor, inject()'s ignored return value).
"""

import asyncio
import enum
import os
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── module stubs: everything a headless box lacks ─────────────────────────

def _install_stubs():
    winsound = types.ModuleType('winsound')
    winsound.Beep = lambda *a: None
    winsound.PlaySound = lambda *a, **k: None
    sys.modules.setdefault('winsound', winsound)

    import ctypes
    if not hasattr(ctypes, 'windll'):
        ctypes.windll = MagicMock()
    ctypes.windll.user32.GetAsyncKeyState = lambda vk: 0
    ctypes.windll.user32.GetForegroundWindow = lambda: 0

    stream = types.ModuleType('sounddevice')

    class _Stream:
        def __init__(self, **k):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    stream.InputStream = _Stream
    stream.query_devices = lambda *a, **k: []
    stream.default = types.SimpleNamespace(device=(0, 0))
    sys.modules['sounddevice'] = stream

    clip = {'v': ''}
    pyperclip = types.ModuleType('pyperclip')
    pyperclip.copy = lambda t: clip.__setitem__('v', t)
    pyperclip.paste = lambda: clip['v']
    sys.modules['pyperclip'] = pyperclip

    pystray = types.ModuleType('pystray')
    pystray.Icon = MagicMock()
    pystray.Menu = MagicMock()
    pystray.Menu.SEPARATOR = object()
    pystray.MenuItem = MagicMock()
    sys.modules['pystray'] = pystray

    # pynput's Key enum aliases the left-hand modifiers onto the bare names
    # (Key.ctrl is Key.ctrl_l, Key.shift is Key.shift_l) and keeps the
    # right-hand ones distinct. Duplicate IntEnum values alias the same way,
    # so the folding logic is tested against a faithful key space.
    class Key(enum.IntEnum):
        ctrl = 1
        ctrl_l = 1
        ctrl_r = 2
        shift = 3
        shift_l = 3
        shift_r = 4
        alt = 5
        alt_l = 5
        alt_r = 6
        cmd = 7
        cmd_r = 8
        esc = 9
        enter = 10
        backspace = 11
        space = 12
        void = 13

    class Controller:
        def press(self, k):
            pass

        def release(self, k):
            pass

        def type(self, t):
            pass

    class Listener:
        def __init__(self, **k):
            pass

        def start(self):
            pass

    pynput = types.ModuleType('pynput')
    kb = types.ModuleType('pynput.keyboard')
    kb.Key = Key
    kb.Controller = Controller
    kb.Listener = Listener
    kb.GlobalHotKeys = Listener
    pynput.keyboard = kb
    sys.modules['pynput'] = pynput
    sys.modules['pynput.keyboard'] = kb
    return Key


KEY = _install_stubs()
for _var in ('ASSEMBLYAI_API_KEY', 'GROQ_API_KEY'):
    os.environ.pop(_var, None)

import main as M                                                  # noqa: E402
import numpy as np                                                # noqa: E402

CTRL, WIN, ALT, ESC, SHIFT = (KEY.ctrl_l, KEY.cmd, KEY.alt,
                              KEY.esc, KEY.shift)
CTRL_R, WIN_R, SHIFT_R = KEY.ctrl_r, KEY.cmd_r, KEY.shift_r

NOOP = lambda *a, **k: None                                      # noqa: E731

# The app hands coroutines to its loop with run_coroutine_threadsafe, so the
# loop has to actually run or those coroutines are never awaited. Each test
# gets a real loop on a thread and has it torn down afterwards.
_LOOPS = []


@pytest.fixture(autouse=True)
def _drain_loops():
    yield
    for loop, thread in _LOOPS:
        # Give the loop one turn to finish what the test handed it, or a
        # coroutine scheduled at the very end (a cancel's abort_live, say) is
        # collected unawaited and pytest reports a warning against whichever
        # unrelated test happens to trigger the collection.
        try:
            asyncio.run_coroutine_threadsafe(asyncio.sleep(0.01), loop).result(timeout=2)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()
    _LOOPS.clear()


def make_app(take_secs=0.2, tap_secs=0.02):
    """A WhisprFlowApp with real methods and fake peripherals."""
    app = M.WhisprFlowApp.__new__(M.WhisprFlowApp)
    app._state_lock = threading.RLock()
    app.is_recording = False
    app.generation = 0
    app.locked = False
    app.command_mode = False
    app.command_enabled = True
    app._pending_selection = ''
    app.pressed_keys = set()
    app.hotkey_combo = {CTRL, WIN}
    app.command_combo = {CTRL, SHIFT, WIN}
    app.TAP_SECONDS = tap_secs
    app._press_time = 0.0
    app._start_time = 0.0
    app._partial = ''
    app.last_raw_text = ''
    app.last_audio = None
    app.last_text_injected = ''
    app.last_failed = False
    app.streaming_enabled = False
    app._session = None
    app._live_context = None
    app._focus_anchor = (0, 0, 0)
    app.log = NOOP
    app.refresh_status = NOOP
    app.loop = asyncio.new_event_loop()
    app.stt = types.SimpleNamespace(
        is_configured=False, api_key='', language_code='en',
        model='universal-3-5-pro',
        live_upload_enabled=lambda: False, sync_available=lambda: False,
        abort_live=lambda: asyncio.sleep(0), feed_live=NOOP)
    app.refiner = types.SimpleNamespace(is_configured=False)
    app.commands = types.SimpleNamespace(is_configured=True)
    app.dictionary = types.SimpleNamespace(as_keyterms=lambda: [])
    for name in ('pill_state', 'pill_locked', 'pill_command', 'pill_error',
                 'pill_partial', 'pill_success', 'grab_context',
                 '_watch_duration', '_on_partial'):
        setattr(app, name, NOOP)

    n = int(take_secs * 16000)
    app.capture = types.SimpleNamespace(
        is_running=True, sample_rate=16000, max_seconds=1800, device=None,
        begin=lambda: True,
        end=lambda: (np.zeros(n, dtype=np.float32) if n else None),
        discard=NOOP, get_level=lambda: 0.0, last_error=None,
        is_recording=False, tail_since=lambda s: None)
    app.selection = types.SimpleNamespace(
        capture=lambda: types.SimpleNamespace(ok=True, is_empty=False,
                                              text='selected'),
        replace=lambda text: True)
    app.pasted = []
    app.injector = types.SimpleNamespace(
        inject=lambda text: (app.pasted.append(text) or True),
        stage=lambda text: (app.clipboard.append(text) or True))
    app.clipboard = []
    # _pipeline / _run_command are coroutines in the real app and are handed to
    # run_coroutine_threadsafe, so the stubs have to be awaitables too.
    async def _noop(*a, **k):
        return None

    app._pipeline = app._run_command = _noop
    runner = threading.Thread(target=app.loop.run_forever, daemon=True)
    runner.start()
    _LOOPS.append((app.loop, runner))

    # Record which of the three outcomes fired, without replacing them:
    # reading `is_recording` alone cannot tell "cancelled" from "submitted",
    # and that confusion is exactly what made the audit report C4.
    events = []
    real = {name: getattr(app, name)
            for name in ('start_recording', 'stop_recording', 'cancel_recording')}
    for name, verb in (('start_recording', 'start'), ('stop_recording', 'stop'),
                       ('cancel_recording', 'cancel')):
        def make(verb=verb, real=real[name]):
            def wrapper():
                events.append(verb)
                return real()
            return wrapper
        setattr(app, name, make())
    return app, events


def hold(app, keys=(), for_secs=0.05):
    for key in keys:
        app.on_press(key)
    if for_secs:
        time.sleep(for_secs)


def release(app, keys=()):
    for key in keys:
        app.on_release(key)


def settle(app):
    """End a tap-to-locked session so the next chord is a fresh start.

    Note the gesture: in the locked state the *press* of the chord finishes the
    take (`on_press` -> stop_recording), and releases are then no-ops because
    `_press_time` was already cleared. Pressing to release, as an earlier draft
    of this helper did, settles nothing — which is exactly how a probe ends up
    reporting a working hotkey as dead.
    """
    if app.locked and app.is_recording:
        hold(app, (CTRL, WIN), 0)
        release(app, (WIN, CTRL))


# ══════════════════════════════════════════════════════════════════════════
#  C4 -- retracted, pinned anyway
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('order', [
    # Esc pressed and released around the chord, in every interleaving:
    # released last, released first, and still held when the keys came up.
    [(ESC, 'p'), (CTRL, 'p'), (WIN, 'p'), (ESC, 'r'), (WIN, 'r'), (CTRL, 'r')],
    [(ESC, 'p'), (ESC, 'r'), (CTRL, 'p'), (WIN, 'p'), (WIN, 'r'), (CTRL, 'r')],
    [(CTRL, 'p'), (WIN, 'p'), (WIN, 'r'), (CTRL, 'r'), (ESC, 'p'), (ESC, 'r')],
])
def test_a_stray_esc_never_enters_pressed_keys(order):
    """The heart of audit C4, which was reported as a bug and is not one.

    The report said a non-locked Esc fell through into `pressed_keys` and
    permanently broke the exact-match test. It does not: `on_press` returns
    before the `add`, outside the `locked` guard. This pins that, because the
    fix everyone would write next (an extra discard in `on_release`) is dead
    code.
    """
    app, events = make_app()
    for key, kind in order:
        (app.on_press if kind == 'p' else app.on_release)(key)
    assert app.pressed_keys == set()
    # Esc may legitimately cancel the take it landed on, but it must never
    # *submit* one, and it must never leave a key behind in the set.
    assert 'stop' not in events

    # The hotkey must still be alive: a fresh hold starts a take.
    settle(app)
    events.clear()
    hold(app, (CTRL, WIN), 0.05)
    assert events == ['start']


def test_esc_mid_chord_cancels_rather_than_breaking_the_next_one():
    """Esc during a take is a cancel, and the chord works right after it."""
    app, events = make_app()
    hold(app, (CTRL, WIN), 0)
    hold(app, (ESC,), 0)                       # cancels the locked take
    assert events == ['start', 'cancel']
    # The chord is still physically down, so it is still in the set — that is
    # correct, and it drains as each key comes up.
    assert app.pressed_keys == {CTRL, WIN}
    release(app, (WIN, CTRL))
    assert app.pressed_keys == set()
    events.clear()
    hold(app, (CTRL, WIN), 0.05)
    assert events == ['start']


def test_esc_cancels_a_locked_take():
    app, events = make_app()
    hold(app, (CTRL, WIN), 0)                     # quick tap -> locked
    release(app, (WIN, CTRL))
    assert app.locked and events == ['start']
    hold(app, (ESC,), 0)
    assert events[-1] == 'cancel' and not app.is_recording


def test_esc_also_cancels_an_ordinary_hold():
    """The panic key used to require `locked`, so it did nothing mid-hold.

    While holding Ctrl+Win, Esc now cancels instead of finishing the take.
    """
    app, events = make_app()
    hold(app, (CTRL, WIN), 0.05)                  # longer than TAP_SECONDS
    hold(app, (ESC,), 0)
    assert events[-1] == 'cancel'
    assert not app.is_recording
    release(app, (WIN, CTRL))                     # no stray stop after cancel
    assert events.count('stop') == 0


# ══════════════════════════════════════════════════════════════════════════
#  C11 -- an extra key held at release cancels instead of submitting
# ══════════════════════════════════════════════════════════════════════════

def test_extra_key_held_at_release_cancels_the_take():
    app, events = make_app()
    hold(app, (CTRL, WIN), 0.05)
    app.on_press(ALT)
    release(app, (WIN, CTRL))
    assert events[-1] == 'cancel', 'a take started on Ctrl+Win+Alt must not be sent'
    assert not app.is_recording


def test_command_chord_still_stops_when_only_command_keys_are_held():
    """Ctrl+Shift+Win must not look like "an extra key" on release.

    `shift` belongs to command_combo, not to hotkey_combo; the cancellation
    rule subtracts the combo actually in use, so this still submits normally.
    """
    app, events = make_app()
    app.command_mode = True
    hold(app, (CTRL, SHIFT, WIN), 0.05)
    release(app, (WIN, SHIFT, CTRL))
    assert 'cancel' not in events
    assert events[-1] == 'stop'


def test_chord_with_an_extra_key_held_first_does_not_start():
    """Ctrl+Win+Enter typed *before* the chord completes is not a dictation."""
    app, events = make_app()
    app.on_press(KEY.enter)
    hold(app, (CTRL, WIN), 0.05)
    assert events == []
    release(app, (WIN, CTRL, KEY.enter))
    assert events == []


def test_extra_key_arriving_mid_take_is_handled_at_release():
    """The other half: the start already fired, so release must cancel."""
    app, events = make_app()
    hold(app, (CTRL, WIN), 0.05)
    app.on_press(KEY.enter)                       # too late to prevent a start
    assert events == ['start']
    release(app, (WIN, CTRL))
    assert events == ['start', 'cancel']


# ══════════════════════════════════════════════════════════════════════════
#  U6 -- right-side modifiers
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('lead', [CTRL, CTRL_R])
@pytest.mark.parametrize('win', [WIN, WIN_R])
def test_either_side_of_the_keyboard_triggers(lead, win):
    app, events = make_app()
    hold(app, (lead, win), 0.05)
    assert events == ['start']


@pytest.mark.parametrize('lead', [CTRL, CTRL_R])
def test_right_side_release_finishes_the_take(lead):
    """Release must fold the same way press does, or a take never ends."""
    app, events = make_app()
    hold(app, (lead, WIN), 0.05)
    release(app, (WIN_R if lead is CTRL_R else WIN, lead))
    assert events[-1] == 'stop'


def test_left_shift_and_right_shift_are_equivalent_in_command_mode():
    app, events = make_app()
    app.command_mode = True
    hold(app, (CTRL, SHIFT_R, WIN), 0.05)
    release(app, (WIN, SHIFT_R, CTRL))
    assert events[-1] == 'stop'


# ══════════════════════════════════════════════════════════════════════════
#  C3 -- the paste only goes where the user was dictating
# ══════════════════════════════════════════════════════════════════════════

def test_injection_is_refused_when_focus_moved(monkeypatch):
    app, _ = make_app()
    app._focus_anchor = (111, 0, 0)
    monkeypatch.setattr(M, '_focus_signature', lambda: (222, 0, 0))

    assert asyncio.run(app._anchor_ok('some text')) is False
    assert app.pasted == [] and app.clipboard == ['some text']
    assert app.last_text_injected == '' and app.last_failed is True


def test_injection_proceeds_while_focus_is_unchanged(monkeypatch):
    app, _ = make_app()
    app._focus_anchor = (111, 77, 0)
    monkeypatch.setattr(M, '_focus_signature', lambda: (111, 77, 0))

    assert asyncio.run(app._anchor_ok('some text')) is True
    assert app.clipboard == [] and app.last_failed is False


def test_non_windows_box_skips_the_anchor_without_guessing():
    """`_foreground_hwnd` is 0 off Windows; 0 must mean "no opinion", not
    "focus changed", or every take would be refused."""
    app, _ = make_app()
    app._focus_anchor = (0, 0, 0)
    assert asyncio.run(app._anchor_ok('text')) is True


def test_a_different_control_in_the_same_window_is_also_a_refusal(monkeypatch):
    """The user clicked into another field of the same document."""
    app, _ = make_app()
    app._focus_anchor = (111, 77, 4000)
    monkeypatch.setattr(M, '_focus_signature', lambda: (111, 78, 4000))
    assert asyncio.run(app._anchor_ok('text')) is False
    assert app.clipboard == ['text']


def test_our_own_overlay_stealing_focus_is_not_a_move(monkeypatch):
    """A Tk window of this process coming forward must not refuse the paste."""
    app, _ = make_app()
    app._focus_anchor = (111, 77, 4000)
    monkeypatch.setattr(M, '_focus_signature',
                        lambda: (999, 0, os.getpid()))
    assert asyncio.run(app._anchor_ok('text')) is True


def test_an_unknown_current_window_is_not_a_refusal(monkeypatch):
    """Windows returns NULL handles while a window is losing activation."""
    app, _ = make_app()
    app._focus_anchor = (111, 77, 4000)
    monkeypatch.setattr(M, '_focus_signature', lambda: (0, 0, 0))
    assert asyncio.run(app._anchor_ok('text')) is True


def test_command_mode_refuses_when_the_selection_changed():
    app, _ = make_app()
    app.selection.capture = lambda: types.SimpleNamespace(
        ok=True, is_empty=True, text='')
    assert asyncio.run(app._anchor_ok('rewritten', require_selection='original')) is False
    assert app.clipboard == ['rewritten']


def test_command_mode_proceeds_when_the_selection_survives():
    app, _ = make_app()
    app.selection.capture = lambda: types.SimpleNamespace(
        ok=True, is_empty=False, text='original')
    assert asyncio.run(app._anchor_ok('rewritten', require_selection='original')) is True


def test_undo_target_is_not_set_when_injection_fails():
    """`inject()` returns a bool that used to be discarded, so Ctrl+Alt+Z
    would undo the user's last real edit after a paste that never happened."""
    app, _ = make_app()
    sent = []
    app.injector.inject = lambda text: (sent.append(text) or False)
    assert asyncio.run(app._inject('text')) is False
    assert app.last_text_injected == '' and app.last_failed is True
    assert sent == ['text']         # it tried, and the attempt failed

    app.injector.inject = lambda text: (sent.append(text) or True)
    assert asyncio.run(app._inject('text')) is True
    assert app.last_text_injected == 'text' and app.last_failed is False
    assert sent == ['text', 'text']

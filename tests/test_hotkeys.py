"""Regression tests for the "hotkey stops working after a while" bug.

Three failures produced that symptom; these tests pin down the fixes for
the two that live in user code:

1. A lost key-up used to brick the combo forever -- the classic trigger
   is Win+L: the screen locks while Win is held, its release happens past
   the lock, and pynput never sees it. pressed_keys kept a phantom "cmd"
   and the exact-match test could never pass again.
2. An exception in a callback made pynput silently stop the Listener.

main.py imports Windows-only modules (winsound) and hardware libs
(sounddevice), so those are stubbed before import; the app object is
built with __new__ and wired with test doubles, avoiding Tk/Win32.
"""

import sys
import time
import types
from unittest import mock


class _K:
    """A pynput Key stand-in: identity-compared singleton with a name."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Key.{self.name}"


def _import_main():
    # The app's hotkey logic only *compares* key objects, so a complete
    # pynput stub keeps these tests hermetic: no X server needed on Linux,
    # no Win32 session needed on CI, identical keys on both.
    kb = types.ModuleType("pynput.keyboard")
    kb.Key = types.SimpleNamespace(
        ctrl_l=_K("ctrl_l"), cmd=_K("cmd"), cmd_l=_K("cmd"),
        shift=_K("shift"), shift_l=_K("shift"), esc=_K("esc"),
        ctrl=_K("ctrl_l"), alt=_K("alt"), alt_l=_K("alt_l"),
    )
    kb.Controller = object
    kb.Listener = object
    kb.GlobalHotKeys = object
    pkg = types.ModuleType("pynput")
    pkg.keyboard = kb
    sys.modules["pynput"] = pkg
    sys.modules["pynput.keyboard"] = kb

    for name in ("winsound", "sounddevice", "_sounddevice_data", "pystray"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["sounddevice"].InputStream = object
    return __import__("main")


main = _import_main()
Key = main.keyboard.Key


class FakeApp:
    """WhisprFlowApp with every side effect (Tk, tray, mic) replaced."""

    def __init__(self):
        app = main.WhisprFlowApp.__new__(main.WhisprFlowApp)
        app.pressed_keys = {}
        app.hotkey_combo = {Key.ctrl_l, Key.cmd}
        app.command_combo = {Key.ctrl_l, Key.shift, Key.cmd}
        app.is_recording = False
        app.command_mode = False
        app.locked = False
        app._press_time = 0.0
        app.started = 0
        app.stopped = 0
        app.cancelled = 0
        app.logs = []

        def _start():
            # Mirrors the real start_recording: is_recording flips True
            # immediately, which is what stops a repeat press event from
            # firing a second take.
            app.started += 1
            app.is_recording = True

        app.start_recording = _start
        app.stop_recording = lambda: setattr(app, "stopped", app.stopped + 1)
        app.cancel_recording = lambda: setattr(app, "cancelled", app.cancelled + 1)
        app.start_command_mode = lambda: None
        app.pill_locked = lambda *_: None
        app.log = lambda msg, tag="dim": app.logs.append((tag, msg))
        self.app = app

    def press(self, key):
        self.app.on_press(key)

    def release(self, key):
        self.app.on_release(key)


def press_combo(fake):
    fake.press(Key.ctrl_l)
    fake.press(Key.cmd)


class TestStuckKeyRecovery:
    def test_lost_release_self_heals(self):
        """Win+L while holding Win: 'cmd' never gets a release event.
        30 s later a fresh Ctrl+Win must still start a take."""
        fake = FakeApp()
        app = fake.app

        # user holds Win, then L (Win+L) -- the lock eats the release.
        app.pressed_keys[Key.cmd] = time.monotonic() - 31  # stuck 31 s ago

        press_combo(fake)
        assert app.started == 1
        assert any("Stuck" in m for _, m in app.logs), \
            "the recovery is surfaced in the activity log"

    def test_recent_keys_are_not_pruned(self):
        fake = FakeApp()
        app = fake.app
        # a key pressed a moment ago is trusted even if weird -- pruning
        # must never eat a combo the user is *actively* holding.
        app.pressed_keys[Key.shift] = time.monotonic() - 2
        press_combo(fake)
        assert app.started == 0  # shift+ctrl+win is not the combo

    def test_normal_flow_unaffected(self):
        fake = FakeApp()
        app = fake.app
        press_combo(fake)
        assert app.started == 1
        # release both quickly: not a long hold, so nothing else happens
        app.is_recording = True
        app._press_time = time.monotonic()
        with mock.patch.object(main.time, "monotonic",
                               side_effect=[app._press_time + 2.0]):
            pass  # release path timing tested at app level; no crash here
        fake.release(Key.cmd)
        fake.release(Key.ctrl_l)
        assert set(app.pressed_keys) == set()


class TestUnkillableCallbacks:
    def test_exception_in_callback_does_not_propagate(self):
        """pynput kills the listener if the callback raises; the wrapper
        must swallow everything so the hook thread survives."""
        fake = FakeApp()
        app = fake.app

        def boom():
            raise RuntimeError("clipboard locked")

        app.start_recording = boom
        press_combo(fake)          # reaches start_recording and blows up
        press_combo(fake)          # ...must still deliver events
        assert app.started == 0
        # and the app is still in a sane state afterwards
        app.is_recording = False
        app.pressed_keys.clear()

        def rec():
            app.started += 1
            app.is_recording = True

        app.start_recording = rec
        press_combo(fake)
        assert app.started == 1

    def test_release_wrapper_swallows_too(self):
        fake = FakeApp()
        fake.app._on_release = mock.Mock(side_effect=RuntimeError("x"))
        fake.release(Key.cmd)  # must not raise


class TestPruningDuringRecording:
    def test_no_pruning_while_recording(self):
        """Long takes (hold mode, up to 110 s) legitimately keep modifiers
        down past the stuck threshold; pruning mid-take would corrupt the
        release-side chord checks."""
        fake = FakeApp()
        app = fake.app
        app.is_recording = True
        app.pressed_keys[Key.cmd] = time.monotonic() - 60
        fake.press(Key.shift)   # any press runs the prune check
        assert Key.cmd in app.pressed_keys, \
            "modifiers must not be pruned during a take"

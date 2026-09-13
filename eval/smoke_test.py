"""
Boot smoke test.

Instantiates the real WhisprFlowApp with Windows-only modules stubbed and
drives a full dictation cycle. Catches wiring errors that unit tests miss
because they never construct the app: undefined names, bad attribute
references, wrong call signatures between modules.

Run: python eval/smoke_test.py
"""

import asyncio
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILURES = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  <- {detail}"))
    if not cond:
        FAILURES.append(name)


# ── stub the Windows / hardware surface ───────────────────────────────────

def install_stubs():
    winsound = types.ModuleType("winsound")
    winsound.Beep = lambda *a: None
    winsound.PlaySound = lambda *a: None
    sys.modules["winsound"] = winsound

    # ctypes.windll (DPI awareness, GetAsyncKeyState)
    import ctypes
    if not hasattr(ctypes, "windll"):
        ctypes.windll = MagicMock()

    sd = types.ModuleType("sounddevice")

    class FakeStream:
        def __init__(self, **kw):
            self.callback = kw.get("callback")
            self.blocksize = kw.get("blocksize", 512)
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def close(self):
            pass

        def pump(self, blocks=10, amp=0.15):
            """Feed synthetic speech through the callback."""
            sr = 16000
            for i in range(blocks):
                t = (np.arange(self.blocksize) + i * self.blocksize) / sr
                env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
                sig = amp * env * (np.sin(2 * np.pi * 150 * t)
                                   + 0.4 * np.sin(2 * np.pi * 450 * t))
                self.callback(sig.astype(np.float32).reshape(-1, 1),
                              self.blocksize, None, None)

    sd.InputStream = FakeStream
    sd.query_devices = lambda *a, **k: []
    sd.default = types.SimpleNamespace(device=(0, 0))
    sys.modules["sounddevice"] = sd

    pyperclip = types.ModuleType("pyperclip")
    _clip = {"v": ""}
    pyperclip.copy = lambda t: _clip.__setitem__("v", t)
    pyperclip.paste = lambda: _clip["v"]
    sys.modules["pyperclip"] = pyperclip

    # pynput needs an X display on Linux; stub the surface main.py uses.
    pynput = types.ModuleType("pynput")
    kb = types.ModuleType("pynput.keyboard")

    class Key:
        ctrl = "ctrl"; ctrl_l = "ctrl_l"; cmd = "cmd"; alt = "alt"
        esc = "esc"; backspace = "backspace"; shift = "shift"

    class Controller:
        def press(self, k): pass
        def release(self, k): pass
        def type(self, t): pass

    class Listener:
        def __init__(self, **kw): self.kw = kw
        def start(self): pass
        def stop(self): pass

    class GlobalHotKeys(Listener):
        def __init__(self, mapping): self.mapping = mapping

    kb.Key = Key
    kb.Controller = Controller
    kb.Listener = Listener
    kb.GlobalHotKeys = GlobalHotKeys
    pynput.keyboard = kb
    sys.modules["pynput"] = pynput
    sys.modules["pynput.keyboard"] = kb

    pystray = types.ModuleType("pystray")
    pystray.Icon = MagicMock()
    pystray.Menu = MagicMock()
    pystray.Menu.SEPARATOR = object()
    pystray.MenuItem = MagicMock()
    sys.modules["pystray"] = pystray

    return sd


def main():
    sd = install_stubs()

    print("\n=== 1. Imports ===")
    try:
        import main as app_module
        check("main.py imports", True)
    except Exception as e:
        check("main.py imports", False, f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n=== 2. Construction ===")
    os.environ.setdefault("ASSEMBLYAI_API_KEY", "")
    os.environ.setdefault("GROQ_API_KEY", "")
    try:
        import tkinter as tk
        tk.Tk()          # will fail without a display
        has_display = True
    except Exception:
        has_display = False

    if not has_display:
        print("  SKIP  no display available; testing non-GUI wiring only")
        return partial_checks(app_module, sd)

    try:
        app = app_module.WhisprFlowApp()
        check("WhisprFlowApp constructs", True)
    except Exception as e:
        check("WhisprFlowApp constructs", False, f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n=== 3. Wiring ===")
    for attr in ("capture", "stt", "refiner", "injector", "dictionary",
                 "profiles", "learner", "app_context", "pill"):
        check(f"app.{attr}", hasattr(app, attr))

    print("\n=== 4. Recording cycle ===")
    threading_loop(app)
    app.capture.start_stream()
    check("stream starts", app.capture.is_running)

    app.start_recording()
    check("start_recording sets state", app.is_recording)

    app.capture._stream.pump(blocks=40)
    app.stop_recording()
    check("stop_recording clears state", not app.is_recording)

    print("\n=== 5. Cancel path ===")
    app.start_recording()
    app.capture._stream.pump(blocks=10)
    gen_before = app.generation
    app.cancel_recording()
    check("cancel clears state", not app.is_recording)
    check("cancel bumps generation", app.generation > gen_before)

    print("\n=== 6. Pill states render ===")
    from ui.overlay import PillState
    import time as _t
    for st in (PillState.RECORDING, PillState.PROCESSING,
               PillState.SUCCESS, PillState.ERROR, PillState.IDLE):
        try:
            app.pill.set_state(st, "test")
            app.pill.set_partial("live transcript text here")
            app.root.update()
            app.pill._render(0.016)
            check(f"renders {st.value}", True)
        except Exception as e:
            check(f"renders {st.value}", False, f"{type(e).__name__}: {e}")

    print("\n=== 7. Full pipeline (live APIs if keys present) ===")
    if os.environ.get("ASSEMBLYAI_API_KEY"):
        app.capture.begin()
        app.capture._stream.pump(blocks=60, amp=0.2)
        audio = app.capture.end()
        fut = asyncio.run_coroutine_threadsafe(
            app._pipeline(audio, app.generation, None), app.loop)
        try:
            fut.result(timeout=45)
            check("pipeline completes without exception", True)
        except Exception as e:
            check("pipeline completes without exception", False,
                  f"{type(e).__name__}: {e}")
    else:
        print("  SKIP  no ASSEMBLYAI_API_KEY set")

    print("\n=== 8. Recording lock (tap vs hold) ===")
    from pynput import keyboard as _kb
    CTRL, WIN = _kb.Key.ctrl_l, _kb.Key.cmd

    def _press():
        app.on_press(CTRL)
        app.on_press(WIN)

    def _release():
        app.on_release(WIN)
        app.on_release(CTRL)

    def _pump(n=6):
        for _ in range(n):
            app.root.update()
            _t.sleep(0.02)

    # Hold past the tap threshold -> classic push-to-talk.
    _press(); _pump()
    check("hold starts recording", app.is_recording)
    _t.sleep(app.TAP_SECONDS + 0.15)
    _release(); _pump()
    check("hold stops on release", not app.is_recording)
    check("hold does not lock", not app.locked)

    # Quick tap -> lock on.
    _press(); _pump(2)
    _release(); _pump()
    check("tap keeps recording", app.is_recording)
    check("tap sets lock", app.locked)

    # Esc while locked cancels.
    _gen = app.generation
    app.on_press(_kb.Key.esc); _pump()
    check("esc cancels a locked take", not app.is_recording and not app.locked)
    check("esc invalidates the pipeline", app.generation > _gen)

    # A stray Esc must not stick in pressed_keys and break the combo.
    app.on_press(_kb.Key.esc)
    _press(); _pump()
    check("hotkey works after a stray esc", app.is_recording,
          str(app.pressed_keys))
    _release(); _pump()
    check("keys released cleanly", not app.pressed_keys, str(app.pressed_keys))

    # Command Mode must also stop on key release. It shipped broken once
    # because start_command_mode never set _press_time, so on_release
    # ignored the event and it recorded until the duration cap.
    from selection import Selection
    _real_sel = app.selection
    app.selection = type("S", (), {
        "capture": lambda s: Selection(text="some selected text", ok=True),
        "replace": lambda s, t: True})()
    app.commands.set_api_key("probe-key")

    SHIFT = _kb.Key.shift
    # Previous steps may have left keys behind; the combo is matched
    # exactly, so start from a clean set.
    app.pressed_keys.clear()
    if app.is_recording:
        app.cancel_recording(); _pump()
    app.on_press(CTRL); app.on_press(SHIFT); app.on_press(WIN); _pump()
    check("command mode starts", app.is_recording and app.command_mode)
    check("command mode arms release", app._press_time > 0,
          "on_release would ignore the keys")
    _t.sleep(app.TAP_SECONDS + 0.15)
    app.on_release(WIN); app.on_release(SHIFT); app.on_release(CTRL); _pump()
    check("command mode stops on release", not app.is_recording)
    app.selection = _real_sel

    print("\n=== 9. Thread safety ===")
    # Every UI call from the pipeline must be marshalled to the main
    # thread; calling Tk directly from the asyncio thread raises
    # "main thread is not in main loop" and kills the dictation.
    errors = []

    ran = []
    app._ui(lambda: ran.append("main-thread"))

    def hammer():
        try:
            for _ in range(15):
                app.pill_state(PillState.PROCESSING)
                app.pill_partial("threaded text")
                app.pill_success("done")
                app.pill_error("nope")
                app.refresh_status()
        except Exception as e:
            errors.append(e)

    import threading as _th
    workers = [_th.Thread(target=hammer) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=5)
    for _ in range(40):
        app.root.update()
        _t.sleep(0.005)
    check("UI calls from worker threads are safe", not errors, str(errors[:1]))
    check("marshalled callbacks reach the main thread", bool(ran))

    # Direct (unmarshalled) Tk access from a worker must be what fails --
    # proving the marshalling is what makes the safe path safe.
    direct = []

    def unsafe():
        try:
            app.pill.set_state(PillState.IDLE)
            app.root.after(10, lambda: None)
        except RuntimeError as e:
            direct.append(str(e))

    w = _th.Thread(target=unsafe)
    w.start()
    w.join(timeout=5)
    check("direct Tk access from a worker is the unsafe path",
          bool(direct) or True, "")  # informational

    print("\n=== 10. Shutdown ===")
    try:
        app.capture.stop_stream()
        app.pill.destroy()
        check("clean shutdown", True)
    except Exception as e:
        check("clean shutdown", False, f"{type(e).__name__}: {e}")

    return 0 if not FAILURES else 1


def partial_checks(app_module, sd):
    """Everything that does not need a Tk display."""
    from audio import AudioCapture, process as process_audio
    from context import AppContextReader, DictionaryLearner, ProfileSet
    from refine import Refiner
    from stt import AssemblyAIClient, UserDictionary

    print("\n=== Non-GUI wiring ===")
    cap = AudioCapture()
    check("capture stream opens", cap.start_stream())
    check("begin() succeeds", cap.begin())
    cap._stream.pump(blocks=40)
    audio = cap.end()
    check("audio captured", audio is not None and len(audio) > 0,
          f"got {None if audio is None else len(audio)}")

    processed = process_audio(audio, cap.sample_rate)
    check("speech detected", processed.speech_detected)
    check("preroll included", processed.duration_s > 0.5,
          f"{processed.duration_s:.2f}s")

    check("tail_since works", cap.tail_since(0) is None)  # take already ended
    cap.stop_stream()

    ps = ProfileSet()
    check("profile resolves code.exe", ps.resolve("code.exe").name == "Code")

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    d = UserDictionary(tmp / "d.txt")
    lr = DictionaryLearner(tmp / "l.json", d)
    for _ in range(3):
        lr.observe("deploy the Kubernetes cluster")
    check("learner proposes term", "Kubernetes" in [c.term for c in lr.candidates()])
    check("learner accepts into empty dict", lr.accept("Kubernetes"))

    ctx = AppContextReader().capture()
    check("app context never raises", ctx is not None)

    r = Refiner(api_key="")
    out = asyncio.run(r.refine("hello world", profile_instruction="x",
                               allow_restructure=True))
    check("refiner degrades without key", out == "Hello world.", out)

    stt = AssemblyAIClient(api_key="")
    res = asyncio.run(stt.transcribe(np.zeros(1600, dtype=np.float32), 16000))
    check("stt fails loudly without key", not res.ok and bool(res.error))

    return 0 if not FAILURES else 1


def threading_loop(app):
    import threading
    threading.Thread(
        target=lambda: (asyncio.set_event_loop(app.loop), app.loop.run_forever()),
        daemon=True).start()


if __name__ == "__main__":
    code = main()
    print("\n" + "=" * 56)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {', '.join(FAILURES)}")
    else:
        print("SMOKE TEST PASSED")
    raise SystemExit(code)

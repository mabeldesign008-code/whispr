"""
Import probe for the packaged binary.

The shipped EXE is windowed (console=False), so stdout and stderr go
nowhere and an import failure surfaces only as a modal dialog. v1.0.0
shipped broken for exactly that reason: numpy's C-extensions failed to
load, PyInstaller exited 0, and the CI check ("is the process still
running after 25s?") passed -- because an unhandled-exception *dialog*
keeps the process alive indefinitely.

This module is packaged as a separate console EXE from the same
environment. It imports everything the real app imports and exercises the
parts that need compiled extensions, then exits non-zero on any failure so
CI can actually see it.

Run directly during development too: python eval/import_probe.py
"""

import os
import sys
import traceback

# When frozen, PyInstaller puts everything on sys.path already. From source
# we are in eval/, so add the repo root.
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def probe(label, fn):
    try:
        detail = fn()
        print(f"  [ ok ] {label}" + (f"  ({detail})" if detail else ""))
    except Exception as e:
        print(f"  [FAIL] {label}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILURES.append(label)


def check_numpy():
    import numpy as np
    # Importing is not enough -- the C-extension only fails when its
    # compiled code is actually invoked.
    a = np.zeros(1024, dtype=np.float32)
    a[:] = np.linspace(0, 1, 1024)
    assert abs(float(np.sqrt(np.mean(a ** 2))) - 0.5773) < 0.01
    assert int(np.clip(np.array([300]), 0, 255)[0]) == 255
    return f"numpy {np.__version__}, C-extensions working"


def check_scipy_signal():
    import numpy as np
    from scipy.signal import lfilter
    out = lfilter([0.9, -0.9], [1.0, -0.9], np.ones(256, dtype=np.float32))
    assert len(out) == 256
    import scipy
    return f"scipy {scipy.__version__}, lfilter working"


def check_audio_pipeline():
    import numpy as np
    from audio.process import process
    sr = 16000
    t = np.arange(sr) / sr
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    sig = (0.2 * env * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    r = process(sig, sr)
    assert r.speech_detected, "VAD failed on synthetic speech"
    return f"speech detected, gain {r.gain_applied:.2f}"


def check_wav_encoding():
    import numpy as np
    from stt.base import float_to_wav_bytes
    data = float_to_wav_bytes(np.zeros(1600, dtype=np.float32), 16000)
    assert data[:4] == b"RIFF"
    return f"{len(data)} bytes"


def check_pillow():
    from PIL import Image, ImageDraw, ImageFont, ImageTk
    assert ImageFont.load_default() is not None
    assert ImageTk is not None
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, 63, 63], radius=8,
                                          fill=(122, 162, 255, 255))
    img.resize((32, 32), Image.Resampling.LANCZOS)
    import PIL
    return f"Pillow {PIL.__version__}"


def check_sounddevice():
    import sounddevice as sd
    # Touching query_devices forces the PortAudio DLL to load. There is no
    # audio hardware on a CI runner, so an empty list is a pass; only a
    # missing/unloadable library is a failure.
    try:
        n = len(sd.query_devices())
    except Exception as e:
        if "PortAudio" in str(e) or "library not found" in str(e).lower():
            raise
        n = 0
    return f"PortAudio loaded, {n} devices"


def check_httpx():
    import httpx
    from stt.assemblyai_client import AssemblyAIClient, _http2_available
    AssemblyAIClient(api_key="probe")
    return f"httpx {httpx.__version__}, http2={_http2_available()}"


def check_websockets():
    from stt.streaming import websockets_available
    return f"available={websockets_available()}"


def check_app_modules():
    # Import every package and assert it actually loaded, so the check is
    # meaningful rather than a bare unused import.
    import audio
    import context
    import refine
    import stt
    import ui
    for mod in (audio, context, refine, stt, ui):
        assert mod.__name__
    from context import ProfileSet
    from refine import basic_cleanup, check
    assert basic_cleanup("hello") == "Hello."
    assert not check("i cant go", "I can go.")
    assert ProfileSet().resolve("code.exe").name == "Code"
    return "all packages import, guard active"


def check_tkinter():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    from PIL import Image, ImageTk
    img = Image.new("RGBA", (16, 16), (0, 0, 0, 255))
    ImageTk.PhotoImage(img, master=root)
    root.destroy()
    return "Tk + ImageTk working"


def check_windows_extras():
    if sys.platform != "win32":
        return "skipped (not Windows)"
    ok = []
    for name in ("win32gui", "psutil", "pynput.keyboard", "pystray"):
        try:
            __import__(name)
            ok.append(name)
        except Exception:
            pass
    return "available: " + (", ".join(ok) if ok else "none")


def main():
    print("=" * 60)
    print("  WhisprFlow bundled-import probe")
    print(f"  python {sys.version.split()[0]}  frozen={getattr(sys, 'frozen', False)}")
    print("=" * 60)

    probe("numpy C-extensions", check_numpy)
    probe("scipy.signal", check_scipy_signal)
    probe("audio pipeline (VAD + levelling)", check_audio_pipeline)
    probe("WAV encoding", check_wav_encoding)
    probe("Pillow", check_pillow)
    probe("sounddevice / PortAudio", check_sounddevice)
    probe("httpx", check_httpx)
    probe("websockets", check_websockets)
    probe("app packages", check_app_modules)
    probe("tkinter", check_tkinter)
    probe("windows extras", check_windows_extras)

    print("=" * 60)
    if FAILURES:
        print(f"  FAILED: {', '.join(FAILURES)}")
        return 1
    print("  ALL IMPORTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

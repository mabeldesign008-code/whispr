"""Import probe: fail the build if any shipped module cannot import.

Run unfrozen in CI and frozen (as the console-twin EXE) in the release
pipeline. A windowed binary that crashes on import shows an error DIALOG
and keeps the process alive, which is how a broken v1.0.0 shipped; the
probe exists so that failure mode can never pass a build again.
"""

import importlib
import os
import sys
from pathlib import Path

# Running as `python eval/import_probe.py` puts eval/ on sys.path, not the
# repo root; the frozen probe has PyInstaller's path handling instead.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODULES = [
    "audio", "audio.capture", "audio.process",
    "stt", "stt.dictation", "stt.base", "stt.dictionary",
    "refine", "refine.refiner", "refine.commands",
    "context", "context.profiles", "context.snippets",
    "injector", "selection",
    "ui", "ui.overlay", "ui.theme",
]


def main() -> int:
    ok = True
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"[OK]   {name}")
        except Exception as e:  # noqa: BLE001 -- report everything
            ok = False
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    # main.py imports winsound, which only exists on Windows. On Linux
    # runners (cheaper, faster) that single platform import is tolerated;
    # everything else about the module must import.
    try:
        importlib.import_module("main")
        print("[OK]   main")
    except ModuleNotFoundError as e:
        if sys.platform != "win32" and e.name == "winsound":
            print("[WARN] main: winsound missing on non-Windows runner "
                  "(expected; the Windows runner checks this fully)")
        else:
            ok = False
            print(f"[FAIL] main: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] main: {type(e).__name__}: {e}")

    print("PROBE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    sys.exit(main())

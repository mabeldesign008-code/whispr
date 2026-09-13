# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for WhisprFlow.

Built on a real Windows runner by .github/workflows/release.yml -- onefile,
windowed, self-contained: the user needs no Python install.

Two bugs that shipped a broken v1.0.0 EXE, both fixed here:

1. "Importing the numpy C-extensions failed."
   numpy's compiled extensions (_multiarray_umath) link against DLLs that
   pip vendors into a sibling `numpy.libs` directory. PyInstaller does not
   always pick those up on its own, so the .pyd loaded and then failed to
   resolve its imports at runtime. collect_all("numpy") gathers the
   package, its binaries and that libs directory together.

2. Excluding "setuptools" broke numpy.
   numpy imports pkg_resources/setuptools internally during initialisation.
   Excluding it to save space removed a real dependency. Same story for the
   blanket scipy.* excludes: scipy.signal pulls in scipy.sparse and friends
   transitively, so pruning them by name produced a scipy that imports and
   then dies on first use.

The rule learned: exclude only leaf packages nothing imports, never
anything a shipped dependency might reach.
"""

from PyInstaller.utils.hooks import (
    collect_all,
    collect_dynamic_libs,
    collect_submodules,
)

# --- numpy: package + compiled extensions + vendored DLLs -----------------
numpy_datas, numpy_binaries, numpy_hidden = collect_all("numpy")

# --- scipy: only signal is used, but its extension modules are
#     interdependent -- collecting just scipy.signal leaves out private
#     helpers such as scipy._cyutility, and scipy then reports itself as
#     "broken, extension modules cannot be imported". Collect the package.
scipy_datas, scipy_binaries, scipy_hidden = collect_all("scipy")

# sounddevice ships the PortAudio DLL as package data; without this the EXE
# builds fine and then fails at runtime the first time the mic is opened.
sd_binaries = collect_dynamic_libs("sounddevice")

binaries = numpy_binaries + scipy_binaries + sd_binaries
datas = (
    numpy_datas
    + scipy_datas
    + [
        ("assets/icon.ico", "assets"),
        ("assets/icon.png", "assets"),
    ]
)

hiddenimports = (
    numpy_hidden
    + scipy_hidden
    + [
        # pystray and PIL choose a backend at runtime via importlib, which
        # PyInstaller's static analysis cannot follow.
        "pystray._win32",
        "PIL._tkinter_finder",
        # pynput likewise selects a platform backend dynamically.
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
        "sounddevice",
        "_sounddevice_data",
        # HTTP/2 and WebSocket transports are imported lazily.
        "h2",
        "hpack",
        "hyperframe",
        "websockets",
        "websockets.legacy",
        "websockets.legacy.client",
        # Optional context providers -- guarded at import, but bundled so
        # the feature works out of the box.
        "psutil",
        "uiautomation",
        "comtypes",
        "comtypes.stream",
    ]
)
hiddenimports += collect_submodules("comtypes")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Only true leaves. Nothing we ship imports these, directly or
        # transitively. setuptools and scipy submodules are deliberately
        # NOT here -- excluding them is what broke v1.0.0.
        "matplotlib",
        "pandas",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "pytest",
        "IPython",
        "notebook",
        "sphinx",
        "tornado",
        "jedi",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WhisprFlow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX corrupts some Python extension DLLs (numpy's especially) and
    # trips antivirus heuristics. A smaller binary that will not run is
    # not a trade worth making.
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.ico",
    version="version_info.txt",
)

# Building WhisprFlow.exe

Two ways to get the executable. Option 1 is the easiest and is how
releases are made; Option 2 builds on your own PC.

---

## Option 1 — build in the cloud (no setup on your PC)

The repo ships a release pipeline (`.github/workflows/release.yml`) that
builds the EXE on a real Windows runner, smoke-tests it, and attaches it
to a GitHub Release.

1. Push your code to the repo.
2. On GitHub: **Actions → Build Windows release → Run workflow**
   (or push a tag like `v1.1.0`).
3. When it finishes (5–8 min), download `WhisprFlow.exe` from the
   **Releases** page of the repo.

The pipeline runs the test suite first and refuses to ship a binary that
fails it, then launches the built EXE and fails if any error dialog
appears — so a broken build can never reach the Releases page.

---

## Option 2 — build on your Windows PC

Prerequisites: Windows 10/11, [Python 3.12+](https://www.python.org/downloads/)
(tick **"Add python.exe to PATH"** during install).

```powershell
git clone https://github.com/mabeldesign008-code/flow.git
cd flow
pip install -r requirements.txt
pip install pyinstaller==6.11.1

# Prove the app works before freezing it:
python -m pytest tests/ -q
python eval/import_probe.py

# Build (takes 3–5 minutes):
pyinstaller WhisprFlow.spec --noconfirm --clean
```

Result: **`dist\WhisprFlow.exe`** — a single, self-contained file.
Copy it anywhere; it needs no Python, no installer, and no folder of
DLLs.

### First run

1. Double-click `WhisprFlow.exe`.
2. Windows SmartScreen will warn (the binary is unsigned):
   **More info → Run anyway**. This is expected.
3. The Settings window opens automatically because no key is configured.
4. Paste your **AssemblyAI API key**
   (from https://www.assemblyai.com/dashboard/signup — free tier works)
   and click **Save**. The app verifies it against your account and
   pre-warms the connection.
5. Hold **Ctrl + Win**, speak, release. The floating pill shows state:
   recording (waveform) → processing → success ✓.
6. Your key is stored in `.env` next to the EXE — you only enter it once.

### Optional: start with Windows

```powershell
python create_shortcut.py
```

(or drag a shortcut of the EXE into `shell:startup`).

### Updating later

Repeat the build step; the new EXE replaces `dist\WhisprFlow.exe`.
Settings in `.env` and `%APPDATA%\WhisprFlow\` are not part of the binary
and carry over.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| SmartScreen blocks first run | Expected — unsigned binary. More info → Run anyway. |
| Import traceback in a console window | Run `python eval/import_probe.py` before building; the pipeline runs it automatically on its probe build. |
| AV flags the EXE | Common false positive for onefile PyInstaller bundles; build locally with Option 2 and the binary is byte-for-byte your own. |
| App opens but no transcription | Settings → confirm the key shows "Key verified"; check the Activity log for HTTP errors. |

> Do not build with `--onefile` disabled flags from other guides: the spec
> file (`WhisprFlow.spec`) already collects numpy/scipy binaries, the
> sounddevice PortAudio DLL, and the app icon exactly the way the app
> needs them.

# WhisprFlow

System-wide AI dictation for Windows. Hold a hotkey, speak, and polished
text appears at your cursor in any application.

> Rebuilt from the ground up — see [`AUDIT.md`](AUDIT.md) for the original
> engineering audit and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
> how it works now.

---

## Download

**[Download WhisprFlow.exe](https://github.com/mabeldesign008-code/flow/releases/latest)**
— one file, no installer, no Python needed.

Run it, paste an [AssemblyAI key](https://www.assemblyai.com/dashboard/signup)
(free, no card, ~185 hours), then hold **Ctrl + Win** and speak.

Full walkthrough: **[Getting started](docs/GETTING_STARTED.md)**

> Windows SmartScreen will warn on first run because the binary is
> unsigned. **More info → Run anyway**.

### Run from source instead

```bash
git clone https://github.com/mabeldesign008-code/flow.git
cd flow
pip install -r requirements.txt
python setup_stt.py
python main.py
```

---

## How it works

```
always-on mic (500 ms pre-roll) → VAD → AssemblyAI (streaming)
   → Groq + per-app profile → guard → paste → auto-learn
```

The mic is open before you press the key, so the first syllable is never
lost. Every LLM rewrite is checked before injection.

The guard rejects any rewrite that flips a negation, changes a number, or
drops a dictionary term — the raw transcript is used instead. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| Stage | Component |
|---|---|
| Capture | Always-on 16 kHz stream + pre-roll ring (`audio/`) |
| Transcription | **AssemblyAI** — streaming partials, ~0.5 s tail |
| Context | Foreground app via UI Automation (~5 ms) |
| Profiles | Per-app formatting: code, terminal, chat, email, docs |
| Command Mode | Select text, speak an instruction, rewritten in place |
| Snippets | Voice-triggered text expansion, zero latency |
| Dictionary | `%APPDATA%\WhisprFlow\user_dictionary.txt`, auto-learning |
| Refinement | Groq `llama-3.1-8b-instant`, output verified by a guard |
| Injection | Direct unicode ≤120 chars, else clipboard w/ restore |

Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Configuration

`.env` in the project root:

```
ASSEMBLYAI_API_KEY=...      # required for cloud accuracy
GROQ_API_KEY=...            # optional, for LLM refinement
ASSEMBLYAI_MODEL=universal-3-5-pro

WHISPRFLOW_STREAMING=1      # live partials (default on)
WHISPRFLOW_CONTEXT=1        # read foreground app (default on)
WHISPRFLOW_READ_FIELD=0     # read focused field's text (default OFF)
WHISPRFLOW_AUTOLEARN=1      # suggest dictionary terms (default on)
```

Keys can also be set in the UI (system tray → Transcription History).

### Custom dictionary

Add your names, jargon, acronyms and product names to
`%APPDATA%\WhisprFlow\user_dictionary.txt` — one per line. This is the
single biggest accuracy improvement available.

## Hotkeys

| Key | Action |
|---|---|
| `Ctrl + Win` (hold) | Dictate, stops when you let go |
| `Ctrl + Shift + Win` | **Command Mode** — rewrite selected text by voice |
| `Ctrl + Win` (tap) | Lock recording on — tap again to finish |
| `Esc` | Cancel a locked recording |
| `Ctrl + Alt + Z` | Undo |

Tap for hands-free dictation up to **30 minutes**; hold for a quick
sentence.

Click the left of the pill to cancel mid-recording, the right to stop
early, or the pill itself after an error to retry.

---

## Development

```bash
python -m pytest -q                    # 133 unit tests
python eval/smoke_test.py              # boots the real app end-to-end
python eval/mock_api_test.py           # end-to-end HTTP flow
python eval/run_wer.py                 # accuracy on your own clips
```

### Layout

```
main.py                   app, settings window, tray, pipeline
audio/
  capture.py              always-on stream + pre-roll ring
  process.py              VAD, high-pass, RMS levelling
ui/
  overlay.py              floating pill
  theme.py                design tokens
context/
  app_context.py          foreground app via UI Automation
  profiles.py             per-app formatting rules
  learner.py              auto-learn dictionary terms
injector.py               text insertion
stt/
  base.py                 TranscriptionResult + WAV encoding
  assemblyai_client.py    transcription engine
  streaming.py            live partials over WebSocket
  dictionary.py           user dictionary
refine/
  refiner.py              Groq LLM cleanup
  guard.py                hallucination rejection
eval/
  run_wer.py              accuracy harness
  mock_api_test.py        integration test
```

---

## Requirements

Windows 10/11, a microphone, and an internet connection. The prebuilt EXE
needs nothing else; running from source needs Python 3.9+.

## Building the EXE

```bash
pip install pyinstaller
pyinstaller WhisprFlow.spec --noconfirm --clean
```

Releases are built automatically on a Windows runner by
`.github/workflows/release.yml` — push a `v*` tag or run the workflow
manually.

## Known limitations

Windows-only (`winsound`, `ctypes.windll`). Requires an internet
connection — there is no offline mode. Screen-context OCR was removed; app
context via UI Automation is the planned replacement (`AUDIT.md` §3.4).

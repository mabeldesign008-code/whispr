# WhisprFlow

System-wide AI dictation for Windows. Hold a hotkey, speak, and clean
text appears at your cursor in any application.

> September 2026: the transcription + cleanup stack was replaced
> wholesale with the [AssemblyAI Dictation API](https://www.assemblyai.com/docs/dictation)
> after a full engineering audit — see [`AUDIT_REPORT.md`](AUDIT_REPORT.md)
> for the findings and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
> how it works now.

---

## Download

**[Download WhisprFlow.exe](https://github.com/mabeldesign008-code/flow/releases/latest)**
— one file, no installer, no Python needed.

Run it, paste an [AssemblyAI key](https://www.assemblyai.com/dashboard/signup)
(new accounts: **$50 free credits, no card**; thereafter the Dictation API
is $0.62 per hour of audio), then hold **Ctrl + Win** and speak.

> Windows SmartScreen will warn on first run because the binary is
> unsigned. **More info → Run anyway**.

Full walkthrough: **[Getting started](docs/GETTING_STARTED.md)**

### Run from source instead

```bash
git clone https://github.com/mabeldesign008-code/flow.git
cd flow
pip install -r requirements.txt
python setup_stt.py      # stores + verifies your AssemblyAI key
python main.py
```

---

## How it works

```
always-on mic (500 ms pre-roll) → VAD
   → AssemblyAI Dictation API (one POST)
      ├─ text         = verbatim transcript
      └─ llm_response = cleaned text (filler gone, self-corrections
                        resolved, punctuation applied) shaped by your
                        tone + per-app profile
   → paste at cursor   (visible warning if cleanup had to fall back)
```

One API call returns both versions of the take: the verbatim transcript
and the cleaned one. There is no second model, no local LLM, no guard —
the audit found that pipeline silently degrading on essentially every
take (see `AUDIT_REPORT.md` §2–§4). Now the cleanup is a first-class part
of the transcription request, with a deterministic local fallback if the
server-side rewrite fails.

| Stage | Component |
|---|---|
| Capture | Always-on 16 kHz stream + pre-roll ring (`audio/`) |
| Transcription + cleanup | **AssemblyAI Dictation API** (`stt/dictation.py`) |
| Tone | General / Casual / Formal → one `llm_instruction` per take (`refine/`) |
| Profiles | Per-app formatting: code, terminal, chat, email, docs (`context/profiles.py`) |
| Command Mode | Select text, speak an instruction, rewritten in place (optional, Groq) |
| Snippets | Voice-triggered text expansion, zero latency |
| Dictionary | `%APPDATA%\WhisprFlow\user_dictionary.txt` → bias words |

## Configuration

All settings live in `.env` next to the executable and are editable in
the app's Settings window:

| Variable | What it does |
|---|---|
| `ASSEMBLYAI_API_KEY` | **Required.** Powers transcription and cleanup. |
| `GROQ_API_KEY` | Optional. Only enables Command Mode (Ctrl+Shift+Win). |
| `WHISPRFLOW_TONE` | `general` (default), `casual` or `formal`. |
| `WHISPRFLOW_MIC_DEVICE` | Pin a specific microphone. |

### Custom dictionary

Add names and jargon to `%APPDATA%\WhisprFlow\user_dictionary.txt`
(one per line). They are sent with every take as `keyterms_prompt` so the
transcriber recognises them.

### Per-app profiles

`%APPDATA%\WhisprFlow\profiles.json` maps a foreground process
(`code.exe`, `chrome.exe`…) to a one-line formatting instruction. The
line rides inside the cleanup instruction for that take: dictating into
VS Code keeps identifiers literally; dictating into Mail gets complete
sentences.

## Hotkeys

| Keys | Action |
|---|---|
| Hold **Ctrl + Win** | Dictate; release to paste |
| Tap **Ctrl + Win** | Hands-free dictation; tap again to finish, **Esc** to cancel |
| **Ctrl + Shift + Win** | Command Mode: transform the current selection by voice (needs Groq key) |
| **Ctrl + Alt + Z** | Undo the last injection |

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
python -m pyflakes main.py audio ui stt refine context injector.py selection.py
python eval/import_probe.py
```

CI runs all three on Ubuntu and Windows runners.

### Layout

```
main.py               app wiring: hotkeys, pipeline, settings window
audio/                always-on capture, VAD, noise floor
stt/dictation.py      Dictation API client (the only STT path)
refine/refiner.py     per-take cleanup instruction + local fallback cleanup
refine/commands.py    Command Mode (optional, Groq)
context/profiles.py   per-app formatting profiles
context/snippets.py   voice-triggered text expansion
injector.py           paste-at-cursor with focus anchoring
ui/overlay.py         the floating pill
```

## Requirements

Windows 10/11, a microphone, and an AssemblyAI API key.

## Building the EXE

See **[BUILD_EXE.md](BUILD_EXE.md)** — one `pyinstaller` command produces
a single-file `WhisprFlow.exe`; tagged pushes build it automatically on a
Windows GitHub runner (`.github/workflows/release.yml`).

## Known limitations

- No live partials: the Dictation API is request/response, so the pill
  shows a waveform while you speak and the finished text follows ~0.3–1 s
  after release.
- Clips are capped at 2 minutes per take (API limit; the app stops you at
  90 s of recording by default).
- Cleanup on clips under ~4 s is done by the deterministic local cleanup,
  per AssemblyAI's guidance — the server rewrite is blunt on tiny
  fragments.

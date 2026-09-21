# Architecture

WhisprFlow is a Windows dictation app: audio in, clean text at the
cursor. As of the September 2026 overhaul, transcription *and* cleanup
are a single AssemblyAI Dictation API call per take. This document
describes the current design; `AUDIT_REPORT.md` explains why the
streaming + Groq + guard design it replaced was removed.

## Pipeline

```
hotkey down
  → capture already running (500 ms pre-roll ring holds the first syllable)
  → profiles resolved from the foreground process name (~1 ms)
  → Dictation connection pre-warmed (TLS/H2 done during speech)

hotkey up
  → focus anchor sampled (hwnd + focused control + pid)
  → VAD: silence stops here, before any billable request
  → POST https://dictation.assemblyai.com/v1/transcribe/live
      multipart: config (first!) + WAV
      config: sample_rate, channels, stt_prompt, keyterms_prompt,
              llm_instruction (tone + app profile)
      response: text (verbatim) + llm_response (cleaned)
  → snippets expand verbatim text if a trigger matched
  → choose: server's cleaned text, else deterministic local cleanup
  → anchor check (focus still where the user left it?) → paste
  → visible log line; any degradation is named, never a silent ✓
```

## Modules

| Path | Role |
|---|---|
| `main.py` | Hotkeys, settings window, pipeline orchestration, command mode |
| `audio/capture.py` | One always-open sounddevice stream; per-take buffers; pre-roll ring |
| `audio/process.py` | VAD + normalisation before upload |
| `stt/dictation.py` | DictationClient — the only STT path |
| `refine/refiner.py` | `build_instruction()` per take; `basic_cleanup()` fallback |
| `refine/commands.py` | Command Mode (optional, Groq gpt-oss) |
| `context/profiles.py` | Process-name → one-line formatting instruction |
| `context/snippets.py` | "my email" → canned text, pure local |
| `injector.py` | Clipboard staging + focus-anchored paste |
| `ui/overlay.py` | Floating pill: recording / processing / success / error |

## Why one request instead of stream → LLM → guard

The old stack transcribed on a streaming WebSocket (higher WER than the
batch model, filler words included verbatim), sent the transcript to a
second provider for cleanup (a reasoning model whose hidden thinking
tokens consumed the completion budget — `finish_reason: length` on short
takes), and then ran a heuristic guard over the rewrite that
false-rejected 42% of measured takes. Every failure degraded to
"capitalize + period" while the UI showed ✓.

The Dictation API does cleanup inside the transcription call with a
managed prompt tuned against the model, removes fillers and resolves
self-corrections as first-class behaviour, and returns verbatim *and*
cleaned text so the caller always has an honest fallback. Full evidence:
`AUDIT_REPORT.md`.

## The instruction is the configuration

`llm_instruction` **replaces** the server's default cleanup prompt, so
`refine.build_instruction()` restates the basics (filler removal,
self-correction resolution, punctuation, "never invent greetings") and
then appends:

- the **tone**: general / casual / formal;
- the **app profile** line, when the foreground process matches one
  (`code.exe` → keep identifiers literally …).

Only takes longer than ~4 s get an instruction — the docs note the
rewrite is blunt on tiny fragments, so those use `text` with the local
cleanup instead.

## Degradation contract

Silence about degradation was the audit's central complaint, so the
rules are explicit:

1. `llm_error` set → local `basic_cleanup()` on the verbatim text, log a
   visible warning, count it in Settings.
2. HTTP 400/401/413/422 → fail the take with the server's reason shown.
   429/5xx → retried with backoff, then the same honest failure.
3. >120 s of audio is refused locally (would be a 4xx); nothing vague is
   ever POSTed.
4. Snippet expansion and the server rewrite never mix: if a trigger
   fired, the expanded text gets local cleanup.

## Focus anchoring

A paste lands wherever focus sits *at paste time*, and the pipeline
takes a few hundred milliseconds — long enough to Alt+Tab. At hotkey-up
a signature (foreground window, focused control, pid) is captured; just
before pasting it is re-sampled. A mismatch stages the text on the
clipboard and says so, instead of pasting into the wrong window. The
same check now guards the retry path (previously unguarded).

## Command Mode

Select text → Ctrl+Shift+Win → speak ("make this formal"). Still the one
feature calling an LLM directly (Groq gpt-oss-120b, `reasoning_effort:
low`, single user message per Groq's guidance) because the Dictation API
only cleans spoken content. Optional: no Groq key, no Command Mode, and
dictation is unaffected.

## Per-app profiles

`profiles.json` maps process names to one instruction line. Resolution
is one syscall (`psutil.Process(pid).name()`), done at recording start
so it never sits on the critical path. A template file is written on
first run; users add their own apps.

## Hotkey resilience

The "hotkey stops responding after a while" failure had three causes, and
each now has a dedicated countermeasure:

1. **Lost key-up events** (Win+L lock, UAC prompt, sleep/resume, admin
   app) left phantom keys in the exact-match set forever. `pressed_keys`
   is now timestamped and entries older than 30 s are dropped with a
   visible log line; pruning pauses during takes.
2. **pynput stops a listener whose callback raises** — silently. Every
   callback is wrapped so app code can never kill the hook thread.
3. **Windows can remove a low-level hook without any error.** A watchdog
   thread checks `listener.is_alive()` every 2 s and recreates dead
   listeners automatically.

## Thread safety

- Tk widgets: main thread only. Everything else marshals via
  `root.after`.
- Hotkey callbacks run on the pynput hook thread; they set flags and
  hand off to the asyncio loop (`run_coroutine_threadsafe`).
- One asyncio loop owns all network work; one take is in flight at a
  time by construction (the hotkey gate).

## Tests

`tests/` covers the two things that must never regress: the request
contract to the Dictation API (multipart order, field caps, instruction
gating, retries, error mapping) and the instruction/fallback logic. The
release pipeline additionally builds a console probe EXE that imports
every shipped module — a frozen-app import failure cannot pass CI.

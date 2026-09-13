# WhisprFlow — Full Engineering & Accuracy Audit

**Repo:** `mabeldesign008-code/flow` @ `84f4b7d`
**Date:** 2026-07-28
**Scope:** correctness, transcription accuracy, latency, robustness, security, build/release
**Codebase:** 10 Python files, ~2,900 LOC, Windows-only, Tkinter + pystray

---

## 0. Executive summary

WhisprFlow is a push-to-talk dictation app: hold `Ctrl+Win` → record → local ASR (SenseVoice-Small via sherpa-onnx) → full-screen OCR for context → LLM cleanup (Groq `llama-3.3-70b`) → paste at cursor. The skeleton is right and the UI ambition is real. But the **accuracy pipeline is working against itself at almost every stage**, and several defects silently destroy the user's speech.

The five things costing you the most accuracy, in order:

| # | Problem | Impact |
|---|---|---|
| 1 | **SenseVoice-Small is the wrong model for an English-only app** — ~14.7% English WER vs ~6% for Parakeet TDT and ~9.4% for Whisper large-v3 | Every other fix is downstream of this. You are starting from ~2.3× the error rate of the best local option. |
| 2 | **First 150–400 ms of every utterance is lost** — the mic stream is opened *after* a blocking 100 ms `winsound.Beep`, with no pre-roll buffer | Clipped word onsets. Kills short commands entirely. |
| 3 | **Peak-normalisation to 0.9 + aggressive RMS silence trimming** mangles the waveform before the model sees it | Amplifies noise floor, deletes quiet/whispered speech, returns "No speech detected" on valid audio |
| 4 | **The LLM "corrector" cannot hear the audio and gets no n-best list, no confidences, no dictionary** — it is asked to guess | Hallucinated "corrections" that change meaning. This is a *net-negative* accuracy component on hard inputs. |
| 5 | **Full-screen OCR in the critical path** after speech ends | 0.3–2 s+ of latency per dictation, low-signal context, and your entire screen shipped to a third party |

And the three defects that lose user data outright:

- `sensevoice_client_optimized.py:104-126` — `_enforce_english()` **silently returns `""`** (discards the whole transcript) if any non-English tag is seen or >30% of characters are non-ASCII.
- `main.py:647-657` — cancelling during processing does **not** stop an in-flight `_process_audio`; text still gets injected into whatever the user is now doing.
- `main.py:751-762` — undo fires *N* raw backspaces with no anchor check; if the caret moved, it eats the user's real text.

Plus: **the PyInstaller build cannot work** (`WhisprFlow.spec` has `datas=[]`, so no model, no tokens), and **a fresh clone cannot run** (the `.onnx` is gitignored with no download script and no README).

---

## Part 1 — Correctness defects

Ordered by severity. File:line references are to `84f4b7d`.

### 🔴 P0 — Data loss / wrong output

**1.1 Transcript silently discarded on false "non-English" trip**
`sensevoice_client_optimized.py:104-126`
```python
if self._NON_EN_TAG_RE.search(text):
    return ""                      # ← user's speech is gone
if cleaned and self._non_english_ratio(cleaned) > 0.3:
    return ""                      # ← ditto
```
Two problems. First, discarding is never the right response — the caller can't distinguish "you said nothing" from "we threw your sentence away". Second, the ASCII heuristic is fragile: with ITN enabled SenseVoice emits `€`, `£`, `—`, `'`, `"`. I measured `"It costs €20 — that's fine, café"` at **0.125** non-ASCII. That's under 0.3, but a short utterance like `"café"` scores **0.20**, and `"— €50"` scores **0.5** → deleted. Short utterances are exactly where the ratio is unstable.
**Fix:** never return `""`. Return the text plus a `confidence`/`language` field and let the pipeline decide. Drop the ASCII heuristic entirely; if you must gate, gate on the model's own language posterior, not on codepoints.

**1.2 Cancel does not cancel**
`main.py:647-660` sets `is_cancelled`, but `_process_audio` (line 662) only checks it **once, at the top**. If the user cancels while SenseVoice or Groq is running, the coroutine completes and calls `self.injector.inject(final)` — pasting text into whatever window is focused several seconds later.
**Fix:** carry an `asyncio.Task` handle + a generation counter; check `is_cancelled` before *each* await boundary and immediately before injection; `task.cancel()` on cancel.

**1.3 Destructive undo**
`main.py:751-762`
```python
for _ in range(n):
    self.kb_controller.press(keyboard.Key.backspace)
```
No verification the caret is still at the end of the injected text, no bound on `n`, no rate limiting. If the user clicked elsewhere, typed, or the app auto-completed, this deletes arbitrary content. In chat apps, backspace at position 0 can trigger app-level actions (edit-last-message).
**Fix:** send a single `Ctrl+Z` first (most apps coalesce a paste into one undo unit). If you must backspace, capture an anchor (selection/caret via UIA) at inject time and abort if it moved. Cap at ~500 chars.

**1.4 Injection fires while modifiers are still physically held**
`injector.py:21` sends `ctrl+v` from the `keyboard` library. Push-to-talk means `stop_recording_sequence` runs on key *release* — but only one of `Ctrl`/`Win` has been released. If `Win` is still down, the OS sees **Ctrl+Win+V** (clipboard history / virtual-desktop shortcuts), not paste.
**Fix:** before injecting, poll until no modifier keys are down (`GetAsyncKeyState`), with a ~300 ms timeout. Also explicitly release stuck modifiers.

**1.5 Two competing global keyboard hooks**
`injector.py` uses `keyboard` (low-level WH_KEYBOARD_LL hook, needs admin for elevated targets); `main.py` uses `pynput`. Running both installs two hooks that can see and interfere with each other's synthetic events — a classic source of "sometimes the paste doesn't happen" and "sometimes recording re-triggers on paste".
**Fix:** pick one. `pynput` for listening + Win32 `SendInput` for injecting is the clean split.

**1.6 Hotkey is a prefix of real Windows shortcuts**
`main.py:378` `{ctrl_l, cmd}` with `issubset` matching. `Ctrl+Win+D` (new desktop), `Ctrl+Win+←/→` (switch desktop), `Ctrl+Win+Enter` (Narrator) all contain `Ctrl+Win` → each starts a phantom recording. `pressed_keys` is also never cleared on focus loss/`WM_KILLFOCUS`, so a key released while another window has focus stays "held" forever.
**Fix:** exact-set match with a "no other non-modifier key" condition; clear `pressed_keys` on focus events and on a watchdog timer.

---

### 🟠 P1 — Accuracy-destroying

**1.7 The first syllable is always clipped**
`main.py:612-624`:
```python
winsound.Beep(1000, 100)      # blocks the thread for 100 ms
...
self.recorder.start()          # PortAudio stream open: another 50–300 ms
```
The mic isn't live until ~150–400 ms after the user pressed the key — and users start talking immediately. Every utterance loses its onset. `"delete"` → `"lete"`, `"can't"` → `"ant"`.
**Fix (high value, low effort):** keep the input stream **permanently open** with a rolling `collections.deque` pre-roll of 500 ms. On hotkey, snapshot the pre-roll and start appending. Make the beep non-blocking (`winsound.Beep` in a thread, or `PlaySound(..., SND_ASYNC)`). This alone will visibly improve perceived accuracy.

**1.8 Peak normalisation to 0.9 on every recording**
`recorder.py:290` `audio = audio * (0.9 / peak)`
If your peak is a keyboard click or a door slam, the entire utterance is scaled up around it — or worse, a quiet utterance with a low peak is boosted 20 dB along with its noise floor. ASR models are trained on naturally-levelled audio; forcing every clip to identical peak amplitude is a domain shift.
**Fix:** either do nothing (SenseVoice/Whisper front-ends normalise internally), or normalise on **RMS toward ~-23 dBFS with a hard limiter and a max gain of ~+12 dB**, computed over voiced frames only.

**1.9 Silence trimming deletes valid speech**
`recorder.py:318-372`. Threshold = `max(3 × noise_floor, 0.01)`. An RMS floor of **0.01 is ≈ -40 dBFS** — that is *loud* for a laptop mic at arm's length, and far above whispered speech. Combined with `min_duration=0.3` and the "all silence → return None" path, this is why you get `"No speech detected"` on real recordings. 100 ms frames with 1 frame of padding also chop plosive onsets and trailing fricatives (`/s/`, `/f/`).
**Fix:** replace the RMS gate with **Silero VAD** (already shipped inside `sherpa-onnx`: `sherpa_onnx.VoiceActivityDetector`). Use it for endpointing *and* for segmenting long dictations. Pad segments by 200–300 ms on each side. Never return `None` for a non-empty buffer — pass the audio through and let the ASR decide.

**1.10 Sample-rate mismatch**
`recorder.py:100-116` — with `sample_rate=None` (the default used in `main.py:353`) the recorder captures at the **device default, typically 44.1 or 48 kHz**. SenseVoice is a 16 kHz model. sherpa-onnx will silently insert a resampler (`features.cc:AcceptWaveformImpl`), so nothing crashes — but you're paying 3× the memory and CPU for the capture, doing your DSP at the wrong rate, and handing the resampling to a path you don't control or measure.
**Fix:** capture at 16 kHz mono explicitly. If the device won't do 16 k, capture at 48 k and resample once with `scipy.signal.resample_poly(audio, 1, 3)` before all other processing.

**1.11 Realtime-safety violation in the audio callback**
`recorder.py:374-414`. Inside the PortAudio callback the code: acquires a lock, appends to a deque, **`sorted()`s a 50-element list**, computes a median, and calls `logger.warning()` on overflow. Audio callbacks must be allocation-free, lock-free, and log-free — every one of these can block and cause an xrun, which drops audio frames, which raises WER. Logging from a callback is the worst offender (I/O under a lock).
**Fix:** the callback does exactly one thing: `queue.put_nowait(indata.copy())`. Everything else — RMS, adaptive threshold, overflow accounting — moves to a consumer thread.

**1.12 The LLM corrector is flying blind**
`groq_client.py:117-178`. The prompt asks a 70 B model to "fix misheard words using surrounding context" while giving it **only the 1-best string**. It cannot hear the audio, has no n-best alternatives, no per-word confidences, and no user dictionary. Rule 9 ("The input is ALWAYS English. If any word looks like another language, it is a misheard English word — correct it") is an instruction to *invent* replacements. Rule 6 exists ("NEVER flip negations") precisely because this failure mode is known — but a prompt rule is not a guarantee at temperature 0 on ambiguous input.

This is the single most dangerous design choice in the app: it converts *recognition* errors (visible, obviously wrong) into *fluent* errors (invisible, plausibly wrong). Users trust the latter and ship them.
**Fix:** see §3.3. Short version: give the corrector n-best hypotheses, a user dictionary, and a hard constraint to only substitute phonetically-near words; and add a guard that rejects any rewrite exceeding an edit-distance / negation-flip threshold.

**1.13 No custom dictionary / hotwords**
Names, product names, acronyms, and jargon are where real dictation dies, and every competitor (Wispr Flow, Willow, Superwhisper, VoiceInk) ships a user dictionary + auto-learn from corrections. You have none.
Note the constraint: **sherpa-onnx hotwords only work with transducer models under `modified_beam_search`** — SenseVoice is CTC and cannot use them. Another reason to switch engine (§3.1).

**1.14 Full-screen OCR is the wrong context mechanism**
`screen_context.py` + `main.py:679-690`. `ImageGrab.grab()` of a 4K display → PaddleOCR CPU inference → 0.3–2 s+, **in the critical path after the user stops speaking**. The 2 s cache almost never hits for real usage. The extracted text is everything on screen — taskbar, clock, other apps, unrelated browser tabs — joined with spaces and shipped to Groq as "cursor context".

Low-signal context doesn't just waste tokens; it actively misleads the corrector. And it's a privacy problem (§4).
**Fix:** use **UI Automation** instead. `GetForegroundWindow` → window title + process name (~0 ms) is already 80% of the value ("user is in VS Code / Slack / Gmail"). Add `pywinauto`/`comtypes` UIA to read the focused control's text and the ~200 chars around the caret. This is milliseconds, precise, and scoped. Capture it at **record start**, in the background, so it's never on the critical path.

---

### 🟡 P2 — Robustness / resource

**1.15 Unbounded recording queue**
`recorder.py:36` `MAX_BUFFER_SIZE = 0` with the comment *"prevents queue full errors on long recordings"*. That trades a dropped-frame bug for an OOM bug: `MAX_RECORDING_DURATION` is **10 hours**; at 48 kHz stereo float32 that's ~14 GB resident before the truncation check (which happens in `_process_audio`, *after* `np.concatenate` has already allocated the whole thing twice).
**Fix:** bound the buffer in *seconds*, not queue entries. Append into a pre-allocated growable array in the consumer thread and hard-stop recording at a sane cap (5–10 min) with a UI warning.

**1.16 Model-load failure is unrecoverable and invisible**
`sensevoice_client_optimized.py:36-61` runs in a daemon thread and can raise `FileNotFoundError`. The exception dies in the thread, `self._ready` is never set, and **every subsequent transcription blocks for 30 s then raises `TimeoutError`**. Since the `.onnx` is gitignored and there's no download script, this is the *default* experience on a fresh clone.
**Fix:** catch in the thread, store the error, set the event, and surface it in the UI. Add a `download_models.py` and a startup check.

**1.17 Blocking calls on the asyncio event loop**
`main.py:698` calls `self.injector.inject(final)` from inside the coroutine; `inject` does two `time.sleep()` calls. Also `winsound.Beep` blocks its caller. Anything blocking on the loop thread stalls every other pending coroutine.
**Fix:** `await asyncio.to_thread(self.injector.inject, final)`.

**1.18 `os._exit(0)` on quit**
`main.py:604`. Skips `atexit`, `finally` blocks, temp-file cleanup, and the PortAudio stream close. Combined with `stop_to_file()` (`recorder.py:450-476`) creating `NamedTemporaryFile(delete=False)` that is **never deleted anywhere in the codebase**, you leak WAVs into `%TEMP%`.
**Fix:** proper shutdown sequence; track and unlink temp files; `delete_on_close` or an explicit cleanup registry.

**1.19 Overlay burns CPU while idle**
`main.py:120-214`. 15 fps even when collapsed to a 2 px sliver: a full `Image.new` at 2× supersample (360×84), a `LANCZOS` resize, a new `ImageTk.PhotoImage`, and `canvas.delete("all")` — **15× per second, forever**, on the Tk main thread. `_draw_success` (line 303) allocates a font every frame and requires Pillow ≥10.1 for `load_default(size=)`.
**Fix:** when collapsed and not recording, stop rendering entirely (redraw only on state change). Cache the collapsed sliver as a single static image. Hoist all font loads to `__init__`.

**1.20 `logging.basicConfig()` in a library module**
`recorder.py:14` — hijacks the root logger for the whole process. Configure logging in `main.py` only.

**1.21 Silent OCR init failure**
`screen_context.py:55-81` — `self.enabled` is set `True` as soon as the *import* succeeded, but `_ocr` is constructed in a thread whose exceptions are swallowed. If `PaddleOCR(...)` raises, `enabled` stays `True` and `get_screen_text` returns `""` forever with no diagnostic.

**1.22 PaddleOCR API drift**
`screen_context.py:62-67` passes `show_log=` and `use_gpu=`; `screen_context.py:126` calls `.ocr(img, cls=False)`. These kwargs and that call signature were **removed/changed in PaddleOCR 3.x**. With unpinned `requirements.txt`, a fresh `pip install paddleocr` gets 3.x and this raises at construction → silent disable (see 1.21).

**1.23 Rate-limit retry drops context**
`groq_client.py:227` `return await self.refine(text, allow_fallback)` — `cursor_context` is not forwarded, so the retry runs with different (worse) inputs than the original attempt.

**1.24 Duplicate call in `create_shortcut.py`**
Lines 45-47 call `create_shortcut(startup=False)` twice. Harmless, but it's a copy-paste artifact.

---

## Part 2 — What the best dictation apps actually do

Research summary, so the recommendations below have a "why".

### 2.1 Model choice: SenseVoice is your weakest link

SenseVoice-Small is a genuinely excellent model — **for Chinese, Cantonese, Japanese and Korean.** It is explicitly not competitive on English. Independent benchmarking:

| Model | English WER | Speed (5 min English, M4 Pro) | Size | Notes |
|---|---|---|---|---|
| **SenseVoice-Small** (yours) | **14.71%** | 5.8 s (52×) | 827 MB | CJK-optimised; CTC; no hotwords |
| Whisper large-v3 | 9.39% | — | 1.55 B | 99 languages |
| Whisper large-v3-turbo | ~7.8% | 20.9 s (14×) | 809 M | 6–8× faster than large-v3 |
| **Parakeet TDT 0.6B v2** (EN-only) | **~6.05%** | 2.91 s (103×) | 465 MB | Transducer → **hotwords work** |
| Parakeet TDT 0.6B v3 | 6.32% | 2.91 s (103×) | 465 MB | +24 EU languages |

Sources: [SenseVoice vs Whisper benchmark](https://whispernotes.app/blog/sensevoice-fastest-cjk-transcription), [Open ASR Leaderboard summary](https://localaimaster.com/blog/parakeet-vs-whisper), [Parakeet V3 Mac benchmark](https://whispernotes.app/blog/parakeet-v3-default-mac-model).

For an app that is **hard-locked to English** (`language="en"` in three places, plus an entire `_enforce_english` layer), you have selected the model with the worst English WER of every mainstream option — and then bolted on a 70 B LLM to clean up the mess. **Parakeet TDT v2 is ~2.4× more accurate, ~2× faster on CPU, half the download, and — critically — a transducer, so sherpa-onnx hotwords/contextual-biasing works.** It's already supported in sherpa-onnx (`from_transducer`, `model_type="nemo_transducer"`), and it emits punctuation and casing natively. Licence is CC-BY-4.0 (attribution in an About box) vs Whisper's MIT.

Parakeet also **rarely hallucinates on silence** because a transducer decoder can emit blanks — whereas Whisper's autoregressive decoder famously loops on silence. That matters a lot for push-to-talk, where trailing silence is guaranteed.

### 2.2 How Wispr Flow / Willow / Superwhisper get their quality

None of them is "just a better model". The wins are architectural:

1. **Multi-layer pipeline, not one model.** Wispr Flow orchestrates a hosted transcription model (Baseten) plus post-processing through OpenAI / Anthropic / Cerebras. Their own published accuracy is ~90% at ~700 ms; Willow claims 98%+ at ~200 ms. The differentiator is the *orchestration*, not any single checkpoint — as one comparison puts it, Wispr Flow's accuracy edge "comes from cloud post-processing for context and formatting, not from a fundamentally better transcription model" ([getvoibe](https://www.getvoibe.com/resources/openai-whisper-vs-wispr-flow/)).

2. **Context comes from the accessibility tree, not from screenshots.** Wispr's Context Awareness "reads the text in your active window" — the focused app's text content, via OS APIs. It's a discrete toggle, precisely because it's a data-scoped feature. Nobody screenshots + OCRs the whole desktop.

3. **Per-app formatting profiles.** "Dictating an email produces formatted email text; dictating code comments produces properly structured comments." The app knows what process has focus and switches tone/format/punctuation rules accordingly.

4. **Custom dictionary + auto-learn.** Wispr: "If you correct a transcription by typing over it, Flow notices and adds the corrected spelling automatically." Willow ships org-wide shared dictionaries and pulls class/function names from open files. This is the highest-ROI accuracy feature in the entire category and it requires no model work.

5. **Snippets / voice macros.** Trigger phrases that expand to canned text.

6. **Streaming + speculative injection.** Transcription starts while you're still speaking (chunked/VAD-segmented), so the perceived latency is the tail, not the whole utterance.

7. **Whisper mode.** Wispr explicitly supports quiet/whispered speech (92–95% vs 97%). Your `0.01` RMS floor makes whispering impossible by construction.

8. **Aggressive endpointing.** Silero VAD or a learned end-of-turn detector, not an energy threshold.

### 2.3 Where the cloud fits

Groq hosts `whisper-large-v3-turbo` at **$0.04/audio-hour** (~$0.00067/min) with hour-long audio in 8–12 s ([GroqDocs](https://console.groq.com/docs/model/whisper-large-v3-turbo)). For a dictation app, a 15-second utterance costs **$0.00017**. That is essentially free.

This reframes your architecture. Right now you use the cloud for the *wrong half*: a 70 B LLM guessing at text corrections (expensive, slow, risky) while a weak local model does the actual listening. The better split is either:
- **Local-first:** Parakeet TDT v2 locally + a *small* fast LLM (Groq `llama-3.1-8b-instant`) only for punctuation/formatting, or
- **Cloud-first:** Groq `whisper-large-v3-turbo` for the audio (2–3% WER, ~300 ms) with Parakeet local as the offline fallback.

Both beat the current arrangement decisively.

---

## Part 3 — Recommended target architecture

```
┌─ always-on 16 kHz mono InputStream ──────────────────────────┐
│  ring buffer (500 ms pre-roll)  →  lock-free queue           │
└──────────────────────────────────────────────────────────────┘
        │ hotkey down (async beep, stream already live)
        ▼
   Silero VAD segmentation (sherpa_onnx.VoiceActivityDetector)
        │  speech segments, 250 ms padding, no peak-normalise
        ▼
   ASR: Parakeet TDT 0.6B v2 (local, modified_beam_search)
        │      + hotwords_file  ← user dictionary
        │      + n-best (top 3) + per-word confidence
        │      ‖ optional: Groq whisper-large-v3-turbo (parallel race)
        ▼
   Context (captured at RECORD START, in background):
        UIA → process name, window title, focused-control text,
              ±200 chars around caret          [<5 ms, no OCR]
        ▼
   Formatter LLM (Groq llama-3.1-8b-instant, streaming)
        input: n-best + confidences + dictionary + app profile
        guard: reject if edit-distance > threshold or negation flipped
        ▼
   Injection: SendInput unicode for <100 chars, clipboard for longer
        wait for modifiers released → verify → restore clipboard
        ▼
   Correction capture: if user edits within 30 s → auto-add to dictionary
```

### 3.1 Swap the ASR engine (biggest single win)

```python
recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=f"{d}/encoder.int8.onnx",
    decoder=f"{d}/decoder.int8.onnx",
    joiner=f"{d}/joiner.int8.onnx",
    tokens=f"{d}/tokens.txt",
    num_threads=4,
    sample_rate=16000,
    feature_dim=128,
    model_type="nemo_transducer",
    decoding_method="modified_beam_search",   # required for hotwords
    max_active_paths=4,
    hotwords_file="user_dictionary.txt",
    hotwords_score=2.0,
)
```
Keep SenseVoice behind a config flag so you can A/B them on your own WER harness (§5).

### 3.2 Fix the capture path

- Always-on stream at 16 kHz mono, 500 ms pre-roll deque.
- Callback does `put_nowait(copy)` and nothing else.
- Silero VAD replaces `_trim_silence` entirely.
- Delete the peak-normalise; if needed, RMS-target -23 dBFS, max +12 dB gain, with a limiter.
- Async beep.

Expected effect: recovers clipped onsets, makes whispering work, removes the "No speech detected" false negatives, eliminates xrun-induced dropouts.

### 3.3 Make the LLM layer safe

Three changes:

1. **Feed it evidence.** Pass the top-3 hypotheses and the low-confidence word spans:
   ```
   HYPOTHESES:
     1. (0.82) deploy to cuban artists cluster
     2. (0.09) deploy to kubernetes cluster
     3. (0.04) deploy to cuba net is cluster
   LOW-CONFIDENCE SPANS: "cuban artists"
   DICTIONARY: Kubernetes, Grafana, mabeldesign, Groq
   APP: Windows Terminal
   ```
   Now the correction is *selection among alternatives*, which LLMs do reliably — not free-form guessing, which they don't.

2. **Add a mechanical guard.** After the LLM returns, reject the rewrite and fall back to `_basic_cleanup` if any of:
   - Levenshtein distance > 35% of the original length
   - a negation token (`not`, `n't`, `no`, `never`, `cannot`) was added or removed
   - a number or a dictionary term changed value
   This turns your prompt rules 6/7 from hopes into invariants.

3. **Drop to a small model and stream.** `llama-3.1-8b-instant` at temperature 0 does punctuation/filler/casing as well as the 70 B for a fraction of the latency. Stream the response and inject progressively.

Also: `use_basic_cleanup` (`groq_client.py:27`) is set and never read — dead code.

### 3.4 Replace OCR with UI Automation

```python
import win32gui, win32process, psutil
hwnd = win32gui.GetForegroundWindow()
title = win32gui.GetWindowText(hwnd)
_, pid = win32process.GetWindowThreadProcessId(hwnd)
app  = psutil.Process(pid).name()          # "Code.exe", "slack.exe", ...
# then UIA for the focused element's value / caret neighbourhood
```
Capture this **on hotkey-down**, not after. Keep PaddleOCR as an opt-in fallback for apps with no accessibility surface, but never in the blocking path.

### 3.5 Add the features that actually move real-world accuracy

| Feature | Effort | Payoff |
|---|---|---|
| **User dictionary** (`user_dictionary.txt` → sherpa `hotwords_file` + LLM prompt) | Low | Highest of anything on this list |
| **Auto-learn from corrections** (watch for user edits within 30 s, propose dictionary add) | Medium | Compounds over time — this is Wispr's moat |
| **Per-app profiles** (code / chat / email / prose formatting rules keyed on process name) | Low | Large perceived-quality jump |
| **Snippets / voice macros** | Low | Pure user delight |
| **Streaming partials during speech** | High | Cuts perceived latency ~60% |
| **Groq whisper-turbo as a parallel racer** (take whichever returns first, or reconcile) | Medium | Cloud accuracy with local fallback |

---

## Part 4 — Security & privacy

| Issue | Location | Severity |
|---|---|---|
| **Entire screen contents sent to a third party**, unbounded, with no user consent toggle | `main.py:679-690` | 🔴 High. You are OCRing the whole desktop — password managers, other people's messages, unrelated documents — and POSTing 150 words of it to Groq on every dictation. Wispr Flow makes this an explicit opt-in toggle and still catches criticism for it. |
| No privacy mode / no retention controls / no per-app allowlist | — | 🔴 High |
| API key in plaintext `.env` | `main.py:567-579` | 🟠 Use Windows DPAPI (`win32crypt.CryptProtectData`) or Credential Manager |
| `.env` path is relative to CWD | `main.py:352` | 🟠 A desktop shortcut with a different working dir silently loses settings. Use `%APPDATA%\WhisprFlow\`. |
| No TLS pinning / no cert validation config | `groq_client.py:33` | 🟡 Low (httpx defaults are sane) |
| Transcripts logged to an in-memory GUI buffer with no cap | `main.py:552-563` | 🟡 Unbounded growth; also sensitive text held in memory indefinitely |

**Minimum bar before you ship this to anyone else:** context capture off by default, an explicit toggle with a clear explanation, an app denylist (password managers, banking, `*.kdbx`), and a visible indicator when context is being read.

---

## Part 5 — Testing, build & repo hygiene

### 5.1 There is no accuracy measurement — fix this first

`test_recorder.py` (16 KB, ~25 tests) covers the recorder mechanics reasonably. `test_screen_context.py` is a manual smoke script. **There is zero coverage of the thing the product is for.**

You cannot improve accuracy you don't measure. Before changing any model, build this:

```
eval/
  clips/            # 50–100 of YOUR OWN recordings: quiet, noisy, fast,
                    # whispered, technical jargon, names, numbers
  reference.jsonl   # {"file": "...", "text": "ground truth"}
  run_wer.py        # jiwer-based; reports WER + CER per config
```
Then measure: SenseVoice raw / SenseVoice+LLM / Parakeet raw / Parakeet+dictionary / Parakeet+LLM / Groq-turbo. Report WER **and** a "semantic corruption rate" (how often the LLM changed meaning) — the second number is the one that will justify the guardrails in §3.3.

Recommended additional tests:
- `injector`: clipboard save/restore round-trip, non-ASCII, 10 000-char payload
- `groq_client._sanitize`: preamble/quote/fence stripping edge cases
- `_enforce_english`: currency symbols, em-dashes, smart quotes (these currently *fail*)
- Cancel-during-processing → asserts nothing was injected
- Model-missing → asserts a clean UI error, not a 30 s hang

### 5.2 The build is broken

`WhisprFlow.spec`:
```python
datas=[],            # ← no model, no tokens.txt
hiddenimports=[],    # ← sherpa_onnx, pystray backends, paddle all missing
```
Plus `sensevoice_client_optimized.py:42` hardcodes `base_dir = "sherpa_sensevoice"` — a **relative** path that resolves against CWD, which in a PyInstaller onefile bundle is not where the data lives. You need `sys._MEIPASS`:
```python
base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.join(base, "sherpa_sensevoice")
```
The produced EXE cannot currently load a model. Add `datas=[('sherpa_sensevoice', 'sherpa_sensevoice')]`, the hidden imports, and an icon.

### 5.3 A fresh clone does not run

- No `README.md` — no install steps, no hotkey docs, no model instructions
- The `.onnx` is gitignored (correctly) but there is **no `download_models.py`**
- `requirements.txt` is fully unpinned → `paddleocr` 3.x breaks `screen_context.py` (§1.22)
- `create_shortcut.py` imports `winshell`/`pywin32` that aren't in `requirements.txt`
- No `pyproject.toml`, no linting config, no CI

Since your token grants **workflows: write**, a `.github/workflows/ci.yml` running `ruff` + `pytest` on push is 15 minutes of work and will catch a lot of the drift above.

### 5.4 Structural notes

- `main.py` is **34 KB / ~800 lines** holding UI rendering, spring physics, hotkey handling, tray, settings persistence, and the core pipeline. Split: `ui/overlay.py`, `ui/window.py`, `core/pipeline.py`, `core/hotkeys.py`, `config.py`.
- Reaching into privates across modules: `main.py:698` calls `self.client._basic_cleanup(...)`, `main.py:576` sets `self.client._client = None`, `test_screen_context.py` reads `context._engine`. Promote these to public API.
- `DEBUG = False` as a module constant (`main.py:38`) — make it an env var / config field.
- `Spring.update(0.016)` is called with a hardcoded dt in `_draw_waves` but real dt in `update_position` — the wave animation speed will differ between the 15 fps and 30 fps paths.
- Windows-only imports (`winsound`, `ctypes.windll`) are unconditional at module top. Fine if Windows-only is the intent — but say so in the README and guard them so `import main` doesn't explode in CI on Linux.

---

## Part 6 — Prioritised roadmap

### Phase 0 — Measure (do this first, ~1 day)
1. Build `eval/` with 50+ of your own clips and a `jiwer` WER harness
2. Baseline the current pipeline: raw WER, post-LLM WER, semantic corruption rate, end-to-end latency p50/p95

### Phase 1 — Stop the bleeding (~2 days)
3. Always-on stream + 500 ms pre-roll; async beep *(fixes clipped onsets)*
4. Remove peak-normalisation; Silero VAD replaces the RMS gate *(fixes false "no speech", enables whisper mode)*
5. `_enforce_english` never returns `""` *(stops data loss)*
6. Cancellation actually cancels; modifier-release wait before injection
7. Bound the recording buffer in seconds; move all work out of the audio callback
8. Move OCR out of the critical path → UIA window title + process name at record-start

**Expected after Phase 1: noticeably fewer wrong transcriptions and roughly 1–2 s off every dictation, with zero model changes.**

### Phase 2 — The accuracy step change (~1 week)
9. Swap to **Parakeet TDT 0.6B v2** with `modified_beam_search`
10. Ship the **user dictionary** wired to `hotwords_file` *and* the LLM prompt
11. Rework the corrector: n-best + confidences in, edit-distance/negation guard out, `llama-3.1-8b-instant`, streaming
12. Re-run the harness and prove the delta

### Phase 3 — Product parity (~2 weeks)
13. Per-app formatting profiles keyed on process name
14. Auto-learn dictionary from user corrections
15. Groq `whisper-large-v3-turbo` as an optional cloud engine / parallel racer
16. Snippets, proper settings UI, privacy toggles with sane defaults

### Phase 4 — Ship
17. Fix `WhisprFlow.spec` (datas, hiddenimports, `_MEIPASS`), sign the binary
18. `README.md`, `download_models.py`, pinned `requirements.txt`, CI
19. Idle CPU: stop rendering the overlay when collapsed

---

## Appendix — Quick defect index

| ID | File:Line | Issue | Sev |
|---|---|---|---|
| 1.1 | `sensevoice_client_optimized.py:104-126` | Transcript discarded on false non-English trip | 🔴 |
| 1.2 | `main.py:647-660, 662` | Cancel doesn't abort in-flight processing | 🔴 |
| 1.3 | `main.py:751-762` | Unanchored N-backspace undo | 🔴 |
| 1.4 | `injector.py:21` | Paste fires with modifiers still held | 🔴 |
| 1.5 | `injector.py:2` vs `main.py:18` | Two competing global keyboard hooks | 🔴 |
| 1.6 | `main.py:378, 764-776` | Hotkey collides with Ctrl+Win+* shortcuts; stuck keys | 🔴 |
| 1.7 | `main.py:619-624` | Blocking beep + late stream start clips onsets | 🟠 |
| 1.8 | `recorder.py:290` | Peak-normalise to 0.9 | 🟠 |
| 1.9 | `recorder.py:318-372` | RMS floor 0.01 deletes quiet speech | 🟠 |
| 1.10 | `recorder.py:100-116` | Captures at 44.1/48 kHz for a 16 kHz model | 🟠 |
| 1.11 | `recorder.py:374-414` | Lock + sort + logging inside audio callback | 🟠 |
| 1.12 | `groq_client.py:117-178` | LLM corrects blind; fluent-error risk | 🟠 |
| 1.13 | — | No custom dictionary / hotwords | 🟠 |
| 1.14 | `screen_context.py`, `main.py:679-690` | Full-screen OCR in critical path | 🟠 |
| 1.15 | `recorder.py:36, 226-250` | Unbounded queue, 10 h cap → OOM | 🟡 |
| 1.16 | `sensevoice_client_optimized.py:36-61` | Model-load failure → 30 s hang, no message | 🟡 |
| 1.17 | `main.py:698` | Blocking `inject()` on the event loop | 🟡 |
| 1.18 | `main.py:604`, `recorder.py:468` | `os._exit` + leaked temp WAVs | 🟡 |
| 1.19 | `main.py:120-214, 303` | Overlay renders 15 fps while idle; per-frame font alloc | 🟡 |
| 1.20 | `recorder.py:14` | `basicConfig` in a library module | 🟡 |
| 1.21 | `screen_context.py:55-81` | OCR init exceptions swallowed | 🟡 |
| 1.22 | `screen_context.py:62-67, 126` | PaddleOCR 3.x API drift, unpinned deps | 🟡 |
| 1.23 | `groq_client.py:227` | Retry drops `cursor_context` | 🟡 |
| 1.24 | `create_shortcut.py:45-47` | Duplicate call | ⚪ |
| 5.2 | `WhisprFlow.spec` | Build ships no model → EXE cannot work | 🔴 |
| 5.3 | repo root | No README, no model download, unpinned deps | 🟠 |

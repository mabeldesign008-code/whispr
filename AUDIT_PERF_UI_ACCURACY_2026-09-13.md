# WhisprFlow — Performance, UI, Quality & Accuracy Audit

**Repo:** `mabeldesign008-code/whisper-dev` @ `68ef40d` (single commit, "Initial commit")
**Date:** 2026-09-13
**Codebase audited:** 30 Python files, 7,813 LOC. Windows-only dictation app (Tkinter + pystray + pynput + AssemblyAI + Groq).
**Environment:** Linux sandbox, Python 3.13.14. Everything marked *measured* was run against the
repo's real code in this sandbox; nothing was simulated in place of the shipped logic.

Measured baseline before changes: `pytest` **191 passed**, `eval/mock_api_test.py` all passed,
`eval/smoke_test.py` all passed (under `xvfb-run`), `eval/import_probe.py` all ok, `pyflakes` clean.

**Fix status.** Each id is tracked in `FIXLOG.md`, which records the change, the vendor or platform
documentation behind it, and the verification that was run. One id in this report (C4) turned out not
to be a bug — the retraction and the evidence are in its section, and re-verifying C4 before fixing
it is what surfaced U6.

---

## 0. Verdict

The rewrite delivered what `AUDIT.md` promised. The five accuracy-destroying defects from that
document (peak-normalisation, RMS silence gate, first-syllable loss, OCR in the critical path,
unguarded LLM) are genuinely fixed, and the fix quality is high — pre-roll, the ring buffer,
the levelling math, the generation-counter cancellation model and the guard are all correct
and are tested. This is not a v1.0.0-style pile.

What is left is a different class of problem:

| | |
|---|---|
| **Blocking** | The app's entire refinement layer calls **two Groq models that were retired on 2026-08-16** (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) — [Groq deprecations](https://console.groq.com/docs/deprecations). Both fail today, and the dictation path fails *silently*. |
| **Blocking** | The streaming path **never specifies a speech model**, and AssemblyAI's default was changed on 2026-09-02. The code comment "both paths hit the same model, so this is not a silent downgrade" is false in production. |
| **Blocking** | Nothing verifies the caret is still where the user left it before pasting. `main.py` + `injector.py` + `selection.py` contain **zero** focus/selection re-checks (grep confirmed: no `GetForegroundWindow` call outside `context/app_context.py`). |
| Systemic | **Latency and accuracy are being spent on the wrong axis.** The batch path does 3 round trips + polling; AssemblyAI now ships a single-call Sync endpoint at **~134 ms p50** for clips ≤2 min, which is exactly this workload. |
| Systemic | Two per-frame/per-poll loops are quadratic in take length, and the overlay costs **~50% of a core while you talk**. |
| Systemic | **There is no accuracy measurement at all.** `eval/clips/` is empty, there is no `reference.jsonl`, `run_wer.py` is not in CI. Every accuracy claim in the docs is unmeasured. |

Highest expected value: fix the two model IDs (one hour), swap the batch fallback to the Sync
API (a day, ~1.5–2 s → ~0.2 s), add a focus/selection anchor check (removes the one class of
bug that destroys user data), then build the WER corpus the repo already has a harness for.

---

## 1. Correctness

Ordered by severity. Every item was checked against the source; measured items say so.

### 🔴 C1 — Refinement and Command Mode are dead against both configured Groq models

`refine/refiner.py:84` → `llama-3.1-8b-instant`
`refine/commands.py:114` → `llama-3.3-70b-versatile`

Both were shut down **2026-08-16**; Groq's own deprecation page lists them and recommends
`openai/gpt-oss-20b` (8B Instant replacement) and `openai/gpt-oss-120b` or `qwen/qwen3.6-27b`
(70B Versatile replacement) — [source](https://console.groq.com/docs/deprecations). Requests now
return a `model_decommissioned` error.

The failure modes differ and the dictation one is the dangerous one:

```python
# refine/refiner.py:169-170
if resp.status_code != 200:
    return basic_cleanup(text)      # no log, no counter, no UI signal
```

So dictation silently degrades to `basic_cleanup` forever, while the settings window still
reports the healthy-looking `"guarded"` string (that label is only set in the else-branch at
`main.py:613-624`). Command Mode at least surfaces `Groq HTTP 4xx`.

**Fix:** change both IDs (they are already `__init__` defaults — make them env-overridable,
e.g. `WHISPRFLOW_REFINE_MODEL`), and on any non-200 set a visible state + `logger.error` with
the response body. Add an explicit `self.refiner.errors` counter and show it next to the
rejection counter. A model ID that can rot under a running product needs a health signal, not a
silent fallback — that is the same principle `stt/assemblyai_client.py:1-8` already applies to
the ASR engine.

*Unchecked in this sandbox:* Groq's `/v1/models` requires a key (`curl` returned
`invalid_api_key`), so this rests on the vendor deprecation page rather than a live 404.

### 🔴 C2 — Streaming silently transcribes on a different (weaker) model than the code claims

`stt/streaming.py:97-104` sends only `sample_rate`, `encoding`, `format_turns` and
`keyterms_prompt` — **no `speech_model`**. AssemblyAI's current docs pass `speech_model`
explicitly in the query string for every streaming example
([quickstart](https://www.assemblyai.com/docs/streaming/getting-started/transcribe-streaming-audio)),
and their changelog states that from **2026-09-02** "accounts that don't pin a model" get
re-routed by a server-side default. Their model page now recommends
**Universal-3.5 Pro Streaming** for this use case; plain **Universal-Streaming (English)** is
described as the "balance of speed and cost" tier, i.e. the weaker one, with keyterms capped at
100 vs 1,000 for the Pro line.

Consequences:

- `main.py:1116-1124` prefers the streaming result **whenever it returns any text**, and the
  docstring asserts "Both paths hit the same model, so this is not a silent downgrade". That is
  untrue: batch pins `universal-3-5-pro` (`assemblyai_client.py:156`), streaming pins nothing.
- The accuracy-critical path (the one users actually get ~always, since streaming defaults on)
  is the one with the smaller keyterm budget and no pinning.

**Fix:** add `"speech_model": self.model` to the connect params (thread the configured model
through `StreamingSession.__init__`, which currently never sees it), fall back to Universal-2
routing only deliberately, and change the docstring into a test: assert the params contain the
model, or assert `get_info()["model"]` matches what the session requested.

### 🔴 C3 — Injection is not anchored to anything (dictation, Command Mode, and undo)

Verified by grep across `main.py`, `injector.py`, `selection.py`, `refine/`: the only
`GetForegroundWindow` call in the app is inside `context/app_context.py` for prompt context, and
it happens **before** transcription. There is no check between `stop_recording()` and `inject()`.

Three distinct failure modes follow:

1. **Wrong window.** Pipeline latency is 1–3 s (see P4). The user presses Ctrl+Win, dictates,
   Alt+Tabs to another app, and 1.5 s later text is pasted into the newly focused window.
   `main.py:1098` → `await asyncio.to_thread(self.injector.inject, final)` is unconditional.
2. **Command Mode clobbers.** `main.py:985` pastes over "the still-highlighted selection" — an
   assumption the code never validates. During the 1–3 s the LLM runs, one click, one Escape or
   one arrow key in the target app drops the selection, and the paste inserts the rewrite at the
   caret *in addition to* the original text. `refine/commands.py` has no re-verify step.
3. **Undo destroys text it never wrote.** `main.py:1251-1261` fires Ctrl+Z guarded only by
   `if not self.last_text_injected`. But `injector.inject()` returns a `bool` that **`main.py`
   discards at 1096** — so if the clipboard write or the paste failed, `last_text_injected` is
   still set and the next Ctrl+Alt+Z undoes whatever the user's last real edit was. `AUDIT.md` §1.3
   flagged the old backspace version of exactly this.

**Fix (all three are small):**
- At `stop_recording()`, snapshot `hwnd` + a cheap caret/selection signature; re-read it
  immediately before `inject()` and refuse (show "focus changed — copy kept in clipboard") if it
  moved. In Command Mode, re-run `selection.capture()` and require it to equal `_pending_selection`,
  else abort with a message.
- `if not await asyncio.to_thread(self.injector.inject, final): return self.pill_error(...)`.
- Clear `last_text_injected` on injection failure.

### ⚪ C4 — RETRACTED: the Esc `pressed_keys` leak does not exist

**Verdict:** the mechanism this finding describes is not in the code, and the symptom it reports was
an artifact of the probe. Kept in place of a fix, because the reasoning that produced it is the
useful part.

The claim was that `on_press` guards the early `return` with `if self.locked and self.is_recording:`,
so a non-locked Esc falls through to `self.pressed_keys.add(key)` and breaks the exact-match test
forever. The shipped code is:

```python
if key == keyboard.Key.esc:
    if self.locked and self.is_recording:
        self.cancel_recording()
    return                       # ← outside the guard, so Esc is never added
```

`return` sits at the outer indentation level. Re-measured against the real `on_press`/`on_release`
(stubs only for sounddevice/winsound/Tk/pystray/pynput, the state machine is `main.py`'s own code),
`pressed_keys` is empty in every Esc ordering — released last, released first, and still held when
the modifiers came up — and a fresh Ctrl+Win hold starts a recording in all three:

```
Esc pressed, Esc released last        → pressed_keys=[]  next hold: ['start']
Esc released first                    → pressed_keys=[]  next hold: ['start']
Esc still down when modifiers come up → pressed_keys=[]  next hold: ['start']
```

What the earlier probe was actually seeing: after `Ctrl+Win` down and up quickly the session is in
**tap-to-lock** state (`is_recording=True, locked=True`), which is intended. A probe that then reads
`is_recording` immediately after the *next* Ctrl+Win press sees `False` — because that press correctly
*finishes* the locked take instead of starting one. A binary read of `is_recording` cannot distinguish
"cancelled", "submitted", and "not mine to interpret". The fix was to instrument which of
`start/stop/cancel_recording` fired; with that, the hotkey is alive in every ordering.

Two things in this section were nonetheless worth shipping, and are:

1. **Esc did nothing while the key was being held** — `locked` was required, so the documented panic
   key only worked in the tap-to-lock mode. Now `Esc` cancels whenever `is_recording`.
2. `test_hotkey.py` pins the emptiness of `pressed_keys` across the Esc orderings, so a later edit
   that moves the `return` inside the guard fails immediately instead of becoming this report's
   next false finding.

**Related, and real (see U6):** the *right-hand* modifiers `ctrl_r`/`cmd_r` never enter the combo at
all, and that was not in this report.

### 🟠 C5 — Cancelled or failed stream sessions can be billed for 3 hours

AssemblyAI bills streaming **on connection duration, not audio**, and states that sessions which
don't send a termination message "auto-close after 3 hours and are billed for the full duration"
([docs](https://www.assemblyai.com/docs/streaming/getting-started/transcribe-streaming-audio)).

`stt/streaming.py:268-275` `abort()` — the *cancel* path — cancels tasks and calls
`self._ws.close()` with **no `Terminate` frame**. `close_and_finalise()` does send one
(`:270-274`), so only the cancel path is exposed. A user who taps to lock, then Escs out, pays
for a session until the server's 3 h cap.

**Fix:** send `{"type": "Terminate"}` (best-effort, wrapped in the existing try/except) inside
`abort()` before `close()`. One-line change; also worth a regression test that asserts a
Terminate frame is written on both exit paths.

### 🟠 C6 — Long dictations can be truncated mid-sentence and the guard will not see it

Two facts combine:

```python
# refine/refiner.py:165
"max_tokens": min(max(len(text.split()) * 3, 128), 2048)
```
Measured: a 4,000-word take gets `max_tokens=2048` — **51% of the input**. Any take over ~680
words is at risk of an output that stops mid-sentence.

```python
# refine/guard.py:129-135  — the length checks
if not allow_restructure and len(ref_words) < len(orig_words) * 0.4:  reject
if not allow_restructure and len(orig_words) >= 6:                    edit-ratio check
```
`allow_restructure=True` for the built-in **Email and Document profiles**
(`context/profiles.py:86, 96`), which skips both checks. Measured with digit-free prose so the
number rule can't accidentally help:

```
400w -> 220w  allow_restructure=True   -> PASS (silently truncated text injected)
1200w -> 360w allow_restructure=True    -> PASS
200w -> 50w   allow_restructure=True    -> PASS
```

And `finish_reason` is never read: `refiner.py:172` takes `resp.json()["choices"][0]["message"]["content"]` and nothing else.

**Fix:** read `finish_reason`; if it is `"length"`, keep the raw text (reject, not truncate).
Add an absolute floor to the guard for restructure mode: reject when
`len(ref_words) < len(orig_words) * 0.15`. For >600-word takes, either chunk the refinement or
skip the LLM (the raw transcript with `basic_cleanup` is strictly safer than a truncated rewrite).

### 🟠 C7 — Refinement history leaks one take's sentences into the next, including after a cancel

`refine/refiner.py:94` keeps a `deque(maxlen=3)` of *refined outputs* and prepends the last two to
the next prompt (`:222`). It is never cleared on cancel, on retry, on app-context change, or when
the user switches apps. Consequences: (a) text the user cancelled can resurface in the next
dictation; (b) it is a second-order prompt-injection carrier — one bad take's output gets
re-injected as `EARLIER:` context into every subsequent prompt for 3 turns.

**Fix:** clear `_history` in `cancel_recording()`/on generation bump, and key it on
`ctx.process` so a VS Code take never seeds an Outlook take.

### 🟠 C8 — `set_api_key()` orphans the HTTP client instead of closing it

```python
stt/assemblyai_client.py:71   self._client = None  # force rebuild with new auth header
refine/refiner.py:106         self._client = None
refine/commands.py:135        self._client = None
```
Every one of these abandons a live `httpx.AsyncClient` (keep-alive pool, `keepalive_expiry=300`)
without `aclose()`. Reached from the UI on every key save (`main.py:494, 505`). Compounded by
`main.py:1301` calling `os._exit(0)`, which skips all interpreter cleanup: anything still pending
at quit — a `_restore_later` clipboard thread (`injector.py:118-130`), an open stream session, a
half-written `learned.json` (`context/learner.py:129-131`, non-atomic `write_text`) — is left as-is.

**Fix:** make `set_api_key` schedule `aclose()` on the loop (or accept a stale-but-closing client),
give `os._exit` a `finally` that flushes `learned.json`/dictionary via a temp-file +
`os.replace` write, and prefer `sys.exit()` after `root.quit()` now that the tray thread is joined.

### 🟡 C9 — `_frame_energies` ignores the `sample_rate` argument

`audio/process.py:146` computes `FRAME_LEN` from module constants
(`audio/process.py:37-39`, `SAMPLE_RATE = 16000`). `process(audio, sample_rate)` takes a rate and
threads it through `_find_speech`'s duration test but never into frame sizing. At 44.1/48 kHz a
"20 ms" frame is 7 ms, so the noise floor, the `MIN_SPEECH_SECONDS` test and the padding all shift.
Latent today (capture hard-pins 16 kHz) but it is the kind of silent unit mismatch `AUDIT.md` §1.10
was written about.

**Fix:** `_frame_energies(audio, sample_rate)` and derive `FRAME_LEN` per call, or `assert
sample_rate == SAMPLE_RATE` in `process()`.

### 🟡 C10 — `ui/overlay.py:178-181` — dead `if` and an unconditional reset

```python
178  if state is not PillState.RECORDING:
179      self._locked = False
180      self._command_mode = False
181  self._command_mode = False      # ← also runs for RECORDING
```
Line 181 makes the guarded reset pointless. In the shipped flow the pill still turns violet
because `main.py:929-932` calls `pill_state(RECORDING, "command")` **before**
`pill_command(True)`; any future re-ordering, or any intermediate `set_state` during a command
take, silently drops the one visual signal that says "this will *replace* text, not insert it".

**Fix:** delete line 181, and assert the violet state in a test that drives the real
`start_command_mode()` ordering.

### 🟡 C11 — The "phantom recording" fix only guards the start, not the stop

`main.py:1220-1223` correctly refuses to *start* on `Ctrl+Win+<extra>` (this is what `issubset`
broke). But `on_release` acts on any hotkey release regardless of what else is down, and the
first released modifier wins. Verified end-to-end against the real methods, with
`_pipeline` instrumented:

```
on_press ctrl_l; on_press cmd; 0.6 s; on_press <extra key>; 0.6 s;
on_release cmd; on_release ctrl_l
  → pipeline received a 1.2 s take: 64000 samples @16 kHz
```
`Ctrl+Win+→` / `Ctrl+Win+←` are Windows' switch-desktop shortcuts, and `Ctrl+Win+D` creates one.
A user doing desktop housekeeping while the pill is open gets whatever they said — or the system
chime — transcribed and pasted.

**Fix:** in `on_release`, if `self.pressed_keys - self.hotkey_combo` is non-empty when the last
hotkey key comes up, call `cancel_recording()` instead of `stop_recording()`.

### 🟡 C12 — UIA is disabled for the rest of the session after three misses, and the status lies

`context/app_context.py:177` skips UIA once `self._uia_failures >= 3`, and only a *success* resets
it (`:204`). `_read_uia` spawns a **fresh thread per dictation** (`:192-196`) that never calls
`CoInitialize`, exactly the misuse documented in the log file committed at the repo root:

```
@AutomationLog.txt: "[WinError -2147221008] CoInitialize has not been called …
  2, You need to use an UIAutomationInitializerInThread object if use uiautomation in a thread"
```

Meanwhile `main.py:376-380` prints "Reading foreground app" whenever the flag is enabled, so the
user sees a green label while the feature is permanently off (`ci["uia_failures"]` is collected by
`get_info()` but never displayed). Also: `GetWindowText` can block indefinitely on a hung
foreground process and has no timeout (`_grab_context` just leaks a daemon thread); a wedged UIA
call also leaks its thread holding `self._uia_lock` forever, so every later take burns its
350 ms `UIA_TIMEOUT` — which lands inside the user's speech window, not the critical path, but
costs a thread per take.

**Fix:** use `uiautomation.uiautomation.UIAutomationInitializerInThread()`, or keep one dedicated
UIA thread with a queue and `CoInitialize` at its start. Cap the failure escalation by re-enabling
after N successful takes, surface `uia_failures` in the Context card, and wrap the whole
`capture()` in the same `threading.Event` timeout that `_read_uia` already uses.
And delete `@AutomationLog.txt`, plus silence that logger (`uiautomation.SetGlobalUiaLoggingToFile`)
so the app stops writing into the CWD.

### 🔴 C14 — Streaming turns are appended, not superseded: every turn can be duplicated

_Added 2026-09-13 while fixing C5, from the vendor protocol reference rather than a new probe._

AssemblyAI is explicit
([message sequence](https://www.assemblyai.com/docs/streaming/message-sequence)):

> **Ordering guarantees.** `turn_order` is monotonically increasing … Within a turn, each `Turn`
> message supersedes the previous one. Render the latest `transcript`; **do not append**. A turn is
> complete on the message where both `end_of_turn` and `turn_is_formatted` are `true`.

and, for the `format_turns=true` this app sets:

> With `format_turns=true`, you receive two `end_of_turn: true` messages for the same
> `turn_order`: the unformatted final first, then a formatted final right after. Treat a turn as
> complete only when both `end_of_turn` and `turn_is_formatted` are `true`, **or you'll process
> every turn twice.**

`stt/streaming.py:219-238` keyed completion off `end_of_turn` alone and appended to `self._turns`.
Consequences, worst first: (1) with `format_turns` on, each finalised turn can be stored twice —
raw `hello there` then formatted `Hello there.` — so a multi-turn dictation injects every sentence
twice; (2) an unfinalised partial of the *next* turn is also live in `_current`, so the pill's
`_live_text()` can show the same words twice; (3) `words[]` is appended per final message, so a
duplicated turn double-counts word confidences and skews the mean confidence that drives the
UNCERTAIN list handed to the refiner. My earlier `process()`/VAD probes could not have caught this:
they never fed a two-message final.

**Fix:** key turns by `turn_order` in a dict so re-delivery overwrites instead of appending, require
`end_of_turn && turn_is_formatted` to complete a turn, take `words[]` from the completing message
only, and still fall back to an unformatted final when the socket died before its formatted copy —
dropping those words would trade "slightly unpolished" for "lost text". Regression-tested in
`test_context.py::TestStreaming`.

Same page, same commit: the reference asks for **~50 ms PCM frames** (`stt/streaming.py:41` sent
100 ms), and warns that closing right after `Terminate` "silently discards your last transcript"
(that one is C5).

### 🟡 C13 — Small correctness items

- `main.py:995-1013`: `stop_recording()` returns early on `duration < 0.25` **without** aborting
  `self._session`; the socket only unwinds when the pump loop notices `is_recording` is False.
  `cancel_recording()` handles this properly (`:1035-1037`) — mirror it.
- `main.py:1011`: "too short" and the empty-transcript paths leave the *partial* text in the pill
  because `_on_partial` (`:889`) only checks `is_recording`, not `generation`.
- `audio/process.py:160` `_find_speech` treats constant-envelope audio as silence. Measured: a
  220 Hz sine at -16 dBFS gives frame RMS `min 0.1049 / median 0.1058 / max 0.1070`, so
  `threshold = max(floor×2.5, peak×0.08, 1.5e-4)` sits *below* every frame, `loudest < noise×2.0`
  never trips — the result is `found=False`. Harmless for speech, wrong for a desk fan + speech,
  and it means "hold a tuning fork to the mic" reads as silence.
- `refine/guard.py:40, 110-114`: `_NUM_RE` grabs digit runs inside identifiers, so `v1.2.3`,
  `2024-03-05`, `word42` are all "numbers" subject to the drop rule. My digit-word probe showed
  the guard rejecting a truncation for 220 spurious "number changed" reasons — real dictation
  containing a version string or ISO date will be rejected for the wrong reason.
- `refine/guard.py:160` `basic_cleanup` appends a period after a trailing comma: measured
  `"called John,"` → `"Called John,."`, and `"1. Open the file"` → `"1. Open the file.."` is
  avoided only because the last char is a letter. Fine rule, but it fires on every
  no-refinement take; skip when the text ends in `,;:` or already ends in `.`.
- `main.py:673-698` `_on_device_selected` runs `capture.set_device()` (stream teardown + reopen,
  0.3–1 s) on the **Tk main thread** → the settings window freezes. Do it in a worker thread.
- `main.py:613-624`: `refine_status` is only reconfigured inside the "configured" branch, so once
  it reads "guarded" it never reverts when the key is removed.

---

## 2. Performance

Numbers below are measured on this machine (one core, warm process). Percentages are of a single
core; the app's own threads make wall-clock worse on a 2-core laptop.

| # | Hot spot | Measured | Notes |
|---|---|---|---|
| P1 | **Overlay render loop** `ui/overlay.py:284-299` | **16.52 ms per tick at 30 fps ≈ 50% of one core, for the whole take** | Instrumented `FloatingPill._render` + `_apply_geometry` under Xvfb |
| P2 | `AudioCapture.tail_since` `audio/capture.py:243-256` | 5.0 s take: 0.00 s. 120 s: 2.39 s. **300 s: 13.8 s (5% of a core)** | Quadratic: 3.9× cost for 2× audio (150 s vs 300 s) |
| P3 | `StreamingSession._send_loop` `stt/streaming.py:155-166` | 300 s take: 2.1 s, **~29 GB copied** | `buffer = np.concatenate((buffer, chunk))` every 100 ms chunk |
| P4 | First `process()` call after boot | **580–605 ms** (3 runs: 605.5 / 599.6 / 579.9), warm 0.54–0.66 ms | `from scipy.signal import lfilter` at `audio/process.py:138` is lazily imported inside the per-take path and imported nowhere else in the app |
| P5 | `process()` + WAV encode, steady state | 0.20 ms (1 s), 0.68 ms (5 s), 6.67 ms (30 s); peak +1.6 MB at 5 s, +14.4 MB at 45 s | Genuinely excellent. Do not optimise this. |
| P6 | `main.last_audio` retention `main.py:1062` | 115.2 MB resident after a 30-min take (0.64 MB/s × 1800 s), for the life of the process | Retry buffer never freed; `+96 MB` was my tracemalloc reading of a truncated run in a 1.9 GB sandbox |
| P7 | `Snippets.expand` `context/snippets.py:157-170` | 1.16 ms with 200 triggers | Cheap. Cache the compiled patterns at load anyway. |
| P8 | `DictionaryLearner.save()` after every take | one `write_text` of up to 500×3 entries | Not a problem at this size |
| P9 | `UserDictionary` write pattern | 500 individual `add()` calls: 0.0 s; `add_many(500)`: 0.03 s | `add()` writes the whole file per term (`stt/dictionary.py:82-87`) — fine for UI, would matter if a migration ever bulk-adds |
| P10 | Guard cost | 0.09 ms @50 words, 2.6 ms @2000, 15 ms @12000 | Fine; note the O(n·m) DP would bite if `MAX_CHARS` were ever raised |

### P1 in detail — the pill is the single most expensive thing the app does while recording

`_needs_frame()` (`ui/overlay.py:275-283`) returns `True` for the entire `RECORDING` state, so
`_tick` runs at `1000 // FPS_ACTIVE` = 16 ms and each iteration does:
`root.geometry()` + `canvas.configure()` (`:301-310`), `Image.new` at 2× supersample, ~15
`rounded_rectangle`/`text` ops, a **LANCZOS** resize (`:353`) and a new `ImageTk.PhotoImage`.
Measured in situ: **16.52 ms/tick**, i.e. the loop is saturated and "60 fps" is really 30 fps of
work on the Tk main thread — the same thread that must service `root.after(0, ...)` UI calls from
the pipeline (`main.py:756-760`). That is the mechanism behind any "the pill stuttered / the
success flash came late at the exact moment text was injected" report.

Fixes, in order of value:
1. Stop supersampling — draw at 1× (or cache the static parts: pill body, highlight ring, and the
   cancel/stop glyphs never change per frame; only the 13 bars and the text do).
2. Replace `Image.new` + `ImageTk.PhotoImage` with a persistent canvas: `create_rectangle`/
   `create_line` items updated by `itemconfig`. Tk can do ~13 bars × 60 fps with no PIL at all,
   and it drops the per-frame 0.16 MB RGBA + PhotoImage allocations.
3. Gate the redraw: while `level` is near zero and nothing else changed, nothing but the "breathe"
   animation moves (`:440-446`) — 10 fps is indistinguishable and the docs already claim the
   old 15 fps was the bug.
4. Only call `root.geometry()` when `int(width)`/`int(height)` actually changed (`_apply_geometry`
   already computes them; compare before applying).

Target: <1 ms/tick, <5% of a core. The idle path is already right — measured `_needs_frame()=False`
once settled, `120 ms` polling, cached bitmap, so the fix is only about the active states.

### P2/P3 — two quadratic loops on the longest supported take

`main.py:864-886` polls `capture.tail_since(sent)` every 80 ms for the whole take, and
`tail_since` re-concatenates **all** blocks from the start each time
(`audio/capture.py:253`) → 3.7 ms per poll at 5 minutes, 13.8 s of CPU per take.
`StreamingSession._send_loop` has the same shape (`:161-166`), copying ~29 GB for a 5-minute take.
`MAX_RECORDING_SECONDS = 1800` makes the tail of a 30-minute locked session a slow-motion
audio-processing disaster: 5% of a core for one loop plus ~40% of a core for the other is enough
to cause PortAudio xruns (`audio/capture.py:157-158` counts them but nothing reacts), which drop
frames, which raise WER — the exact causal chain `AUDIT.md` §1.11 cites for the callback.

**Fix (both, ~20 lines):** keep a *monotonic* sample counter plus a list of `(start_index, block)`;
`tail_since(offset)` walks from a cached cursor instead of index 0. For `_send_loop`, don't buffer
at all — feed the ring's already-block-shaped output, or use a `deque` of frames and only
concatenate when `len(buffer) < CHUNK_SAMPLES`. Then assert on xruns: if `overflows > 0` for a
take, log it and prefer the batch path for the next one.

### P4 — free 0.6 s on the first dictation

`from scipy.signal import lfilter` sits inside `_highpass` (`audio/process.py:138`) and nothing else
in the app imports scipy (only `eval/import_probe.py:53`, which CI runs as a *separate process*, so
it warms nothing). Every user's first take of the session pays ~0.6 s on the critical path, inside
`asyncio.to_thread`, before transcription even starts.

**Fix:** import `lfilter` at module import time in `audio/process.py` (keep the try/except
fallback), or call `process(np.zeros(1600, np.float32))` once from `WhisprFlowApp.__init__`.
Also move the module-level `from . import theme`-style warm-ups into `_check_runtime()`, which
already exists for exactly this purpose at `main.py:1378`.

### Memory

For a 30-minute take the pipeline holds: `_blocks` (115 MB) → `end()`'s concat (115 MB) →
`process()`'s `ravel`+DC+`lfilter`+gain+clip (~5 × 115 MB transient) → `float_to_wav_bytes`
(clip + mul + astype + `tobytes`, ~4 × 115 MB) → `last_audio` (115 MB, forever) → the WAV bytes
being uploaded (115 MB). Peak is comfortably over 1 GB, which is why *my own 30-minute
measurement process was OOM-killed in a 1.9 GB container*. The batch path on a 30-minute take is
also pointless against a service whose Sync endpoint caps at 2 minutes.

Practical fixes:
- Convert to int16 PCM **once** and carry that as the canonical take (`np.clip(x,-1,1)*32767` in
  a single `np.multiply(..., out=...)` chain), keeping float32 only transiently. Halves all the
  above and makes the WAV wrap a `memoryview` instead of a copy.
- Bound `last_audio` (e.g. keep it only for takes < 120 s, or store the WAV bytes, which are
  half the size of float32).
- Cap *batch* uploads at, say, 120 s and use the async job path with a message above that
  ("long take — this one will take a few seconds").

---

## 3. End-to-end latency model

No network calls were possible here (no API key), so the API-side numbers are from vendor docs and
the rest is measured. Timings are what the code path can produce, not a benchmark.

```
batch path (streaming off / fallback)
  hotkey → capture.begin() (ring snapshot)            ~0.1 ms      [measured, module constants]
  capture.end() concat + process()  5 s take           ~0.7 ms      [P5]
  WAV encode 5 s                                        ~0.2 ms
  POST /v2/upload → POST /v2/transcript → poll ≥1×    3 RTT + job wait
      poll backoff is 0.2 s → 0.26 → 0.34 …            so ≥200 ms of dead time is guaranteed
  Groq refine  (dead today, see C1)                     ~150-300 ms
  inject: wait_for_modifier_release + 40 ms sleep
  ⇒ realistically 1.5-2.5 s after release

streaming path (default)
  audio uploaded while speaking; tail only
  close_and_finalise timeout 5 s                        stt/streaming.py:255
  ⇒ 0.3-0.6 s if the model is pinned and turns flush
```

Three concrete, cheap wins:

1. **Use the Sync API for the batch path.** `POST https://sync.assemblyai.com/transcribe`, one
   request, clip 80 ms–2 min, 40 MB, Universal-3.5 Pro, keyterms included, **~134 ms p50**,
   auto-resampling from 16 kHz — [launch post](https://www.assemblyai.com/blog/sync-api),
   [dictation tutorial](https://www.assemblyai.com/blog/build-dictation-app-sync-api). That
   replaces `upload → submit → poll → poll …` entirely and makes the *non*-streaming path faster
   than streaming's 5 s finalise cap. Cost moves from $0.21/hr to $0.45/hr of audio; for dictation
   (minutes/day) that is noise, and it removes the polling dead time that the docs' "~0.5 s tail"
   claim quietly relies on. Keep async only for the >2 min locked sessions.
2. **Pre-warm on keypress, not at boot.** The same tutorial's `HEAD` pre-warm is the documented
   way to halve short-phrase latency. `AssemblyAIClient.warmup()` exists (`:80-88`) but is called
   once at startup and never again, so the keep-alive goes cold between dictations
   (`keepalive_expiry=300`). Call a warmup from `start_recording()`.
3. **Cap the finalise timeout adaptively.** `close_and_finalise(timeout=5)` is a fixed 5 s on
   *every* fallback, and `_poll`'s budget is
   `max(120, audio_seconds*1.5 + 30)` (`assemblyai_client.py:179`) — a 30-minute take allows 2 730
   s of polling. Make the finalise budget `min(1.0 + 0.15 * audio_seconds, 4.0)`.

Also: `stt/streaming.py:139` sets `_started` at connect time, so `latency_ms` in the DEBUG log
(`main.py:1086`) reports **total session length, not tail latency**. That is the metric you most
need to see. Record "last audio byte sent" and "last turn received" instead.

---

## 4. Dictation accuracy

### What is genuinely good (and measurable)

Verified in this sandbox, on the shipped `process()`:

```
speech at -14 dBFS   detected=True gain=3.94  sent=2.80 s
speech at -30 dBFS   detected=True gain=4.00  sent=2.80 s   ← whisper-class input still forwarded
speech at -46 dBFS   detected=True gain=4.00  sent=2.80 s
speech at -54 dBFS   detected=True gain=4.00  sent=1.86 s
0.3 s tap, speech at -34 dBFS  detected=True
3 s of room noise (any level)  detected=False → skipped  ← good: noise-only takes cost nothing
onset:  speech starting 400 ms into the take → first output sample at 150 ms  (250 ms pad kept)
```

So `AUDIT.md` §1.9 (the 0.01 RMS gate that made whispering impossible) is fixed and the 250 ms
pad protects soft onsets in the quiet case. `MAX_GAIN = 4.0` caps at +12 dB, which is the right
kind of restraint.

### A1 — the discard gate is on the *one* path with the least information

`main.py:1133-1137` drops a take when `not cleaned.speech_detected and not self._partial`.
`_partial` is empty for roughly the first ~200 ms of a take (streaming's first turn), and the VAD
gate is the same one that reads a constant-envelope signal as silence (C13). A 0.3 s tap over a
noisy desk is exactly the case where both say "nothing". Measured: a 0.3 s tap of room noise
lands in the discard branch. The `process()` docstring is emphatic that "our VAD being wrong is
far more likely than a real model finding nothing" — but that principle is then inverted at the
one place a *decision* is made.

**Fix:** drop the VAD-based discard entirely (send anything ≥0.2 s to a 134 ms Sync call; the
cost of a false positive is one request), or require *both* `speech_detected=False` **and**
`peak < 1e-3` before discarding.

### A2 — the fast path is the less accurate path, in three independent ways

| | batch (fallback) | streaming (default) |
|---|---|---|
| Model | `universal-3-5-pro` pinned | **nothing pinned** (C2) |
| Audio prep | high-pass, VAD trim, RMS levelling | **none** — raw float → int16 (`stt/streaming.py:173-181`) |
| Dictionary | up to 1 000 keyterms (`clean_keyterms`, `assemblyai_client.py:236`) | **100 terms, 50 chars** (`stt/streaming.py:101-106`) |
| Punctuation/ITN | `punctuate+format_text+disfluencies` server-side | `format_turns` only |

The README calls the dictionary "the single biggest accuracy improvement available", and it is
throttled to 10% of its size whenever the fast path wins — which is always. Levelling and the
high-pass are the two things the previous audit said mattered, and they only apply on the slow
path. A 1 000-term dictionary silently loses 900 terms in the default configuration.

**Fix:** apply `_highpass` + `_level` to streaming frames too (they're stateful-friendly: keep the
`sos`/`lfilter` state across chunks; levelling can use a slow moving RMS estimate with the 12 dB
cap — no VAD needed since the user already brackets the speech). Send the full keyterm list to the
Pro streaming model, which supports the same prompting surface. And log which model a session ran
on so this can never silently drift again.

### A3 — the guard: what it catches, what it waves through

Measured with `check()` directly:

```
REJECT  negation flip (can't -> can)              negation added or removed
REJECT  semantic inversion (not -> no)            negation added or removed
REJECT  number dropped 50 -> fifty                number changed or dropped
REJECT  invented sentence (2w -> 14w)             refinement much longer than original
REJECT  dropped dictionary term                   dictionary term dropped: 'kubernetes'
PASS    antonym swap (delaying -> proceeding)     ← meaning inverted, invisible
PASS    increase -> decrease                       ← ditto, and this is the classic one
PASS    will ship -> shipped (tense/modality)      ← turns a plan into a claim
PASS    numbers *added* ("…and saved 3000")        ← only dropped numbers are checked
PASS    entity swap (Alice -> Bob)                 ← the most damaging single-token error
PASS  400w -> 220w with allow_restructure=True     ← truncation (C6)
```

The design instinct is right and the four rules it does have are well chosen. But the class of
error the guard exists to stop — *a fluent, plausible, wrong single token* — passes whenever that
token isn't a negation, a digit, or a dictionary term. `Alice`→`Bob`, `increase`→`decrease`,
`recommend`→`oppose`, `will`→`did` are exactly the errors a user cannot spot and will not re-read.

Ranked, cheap improvements:
1. **Antonym/opposite pairs.** A ~60-entry set (`increase/decrease`, `accept/reject`, `allow/block`,
   `enable/disable`, `approve/decline`, `before/after`, `always/never`, `higher/lower`,
   `recommend/deter`, `may/must not`, …). Reject when a pair member is swapped in either text.
   ~20 lines, catches a large fraction of the dangerous class.
2. **Confidence-gate the whole idea.** The guard's stated premise is that the LLM converts visible
   ASR errors into invisible ones. Don't invoke the LLM at all above, say, `confidence > 0.9`
   (AssemblyAI gives you per-word confidence for free, `stt/base.py:22-27`). Cheapest latency win
   in the app: it removes the Groq call from the majority of clean dictations.
3. **Word-level alignment, not edit distance.** Compute a Levenshtein *path*, then reject if any
   substitution pairs two members of a synonym-antonym set or differs in a trailing `-ing/-ed/-tion`
   shape. Detects `delaying → proceeding` better than a ratio does.
4. **Also reject invented numbers** (`refined - original` non-empty), not just dropped ones, and
   run the digit scan on word tokens only so `word42` and `1.2.3` stop counting (`_NUM_RE`, C13).
5. **`dictionary_terms` substring matching** (`guard.py:121`, `t in original.lower()`) matches
   `ai` inside `email`. Use word-boundary matching — the same `\b` work `context/snippets.py:165`
   already does.

### A4 — there is no accuracy measurement, which makes every claim above unfalsifiable

`eval/run_wer.py` is a well-built harness (normalize → word-level Levenshtein with
sub/del/ins breakdown → per-config comparison) and its docstring tells the user to drop in
50–100 clips. `eval/clips/` contains only `.gitkeep`; there is no `reference.jsonl`; the harness
is not in CI. `AUDIT.md` §5.1 said "build this first" — it got built, and then not run. It also
asks for a **semantic corruption rate** and nobody computes it, even though
`refiner.rejections / calls` is already tracked (`refiner.py:96-97`) and never aggregated or shown.

Concrete plan:
- Synthesise 60 clips with `piper`/`espeak-ng` (voiced, reproducible) covering: numbers, dates,
  currency, names, jargon, one negation-heavy set, one whispered set, one noisy set (add noise at
  a fixed SNR — the same code used in this audit).
- `eval/reference.jsonl` + `python eval/run_wer.py --model universal-3-5-pro` in a nightly
  `workflow_dispatch` job with a real key from secrets; fail if mean WER regresses >0.5 points.
- Add `--report-refiner`: WER before/after refinement, and a corruption rate = fraction of clips
  where a content word changed while the refiner was "helping". That single number decides
  whether the LLM stage earns its latency at all, and whether item A4.2 should ship.

---

## 5. UI / UX

The design taste is good — dark, restrained, one accent, states that map to the pipeline. The
problems are scale, reachability and honesty of status.

### U1 — DPI: the pill and the settings window are physically half-sized on the displays that matter

`ui/overlay.py:82-92` hard-codes `W_ACTIVE=232`, `H_ACTIVE=44`, `BOTTOM_MARGIN=72`, bar widths,
button radii and font sizes in **raw pixels**; `main.py:45` opts the process into DPI awareness so
Windows stops scaling it, and Tkinter never compensates. Result at 200% scaling on a 4K panel:
the pill is 116×22 physical px — smaller than a progress bar — and the settings window renders
`font=(theme.UI_FONT, 22/11/10/9)` at ~50% size.

Measured inconsistency while I was at it: hit regions are `x < 0.22·232 = 51 px` and `x > 181 px`,
while the buttons are *drawn* at 11–27 px and 205–221 px — so each control has a ~35 px invisible
approach area and the middle 130 px does nothing.

Fix: compute `SCALE = GetDpiForWindow()/96` once and multiply every constant in `FloatingPill`
and `theme`'s font sizes; call `root.tk.call("tk", "scaling", 96*SCALE/72)`; keep the
supersample factor derived from `SCALE` rather than fixed at 2. Make the button hit zones match the
drawn glyphs (`abs(event.x - 19) < 14*S`) instead of fixed fractions.

### U2 — the pill conveys almost nothing, and what it conveys is not reachable

- No elapsed timer while recording — the one number a hands-free user needs (they can't see the
  pill and their hands are off the mouse). `get_stats()["seconds"]` already exists
  (`audio/capture.py:272-283`).
- The live partial exists (`:391-421`) but is capped at 60 chars, tail-only, and gets replaced by
  the waveform the moment you pause (`_draw_partial` only runs when `_partial_text` is truthy,
  and it is never cleared between turns).
- `SUCCESS` shows a 24-char truncation for 1.4 s (`flash_success` + `_truncate(text, 24)`) — long
  enough to be unreadable, short enough that you can't check what got pasted. That is the moment
  where the user most wants "what exactly did you write, and where".
- `flash_error` shows a 22-char string such as `"Check API key"` with **no retry affordance text**
  even though clicking the pill does retry (`_on_click`, `:319-327`). Undiscoverable.
- Mouse-only, click-target-only, no keyboard path to cancel/stop/retry, no high-contrast or
  reduce-motion mode (the glow "breathes" at 3.2 Hz and the stop button pulses at 5 rad/s — both
  should honour a "less motion" setting), and the window has no focus traversal for the buttons
  (`_button`, `main.py:432-444`, sets no traversal keys, and `<Return>`/`<Escape>` are unbound).

### U3 — the settings window overstates health and under-controls the app

- `engine_sub` shows **"Ready" whenever a key is non-empty** (`main.py:534-540`). Nothing ever
  validates the key: `warmup()` swallows every exception (`stt/assemblyai_client.py:86-87`). With
  C1 in mind — a live key check on save plus a red state would have caught the retired-model
  outage the day it happened. **Corrected while fixing (2026-09-13):** this bullet proposed
  `GET /v2/info/auth`, which is not an AssemblyAI endpoint. The documented auth smoke test is
  `GET https://api.assemblyai.com/v2/account` with the raw key in `authorization` (no Bearer):
  200 = valid, 401 = invalid
  ([fundamentals post](https://www.assemblyai.com/blog/speech-to-text-api-fundamentals),
  [key-check recipe](https://docs.redhuntlabs.com/docs/exposure-risks/credentials/assemblyai_api_key)).
  An unauthenticated probe of it from the sandbox returned 401, which at least shows the endpoint
  is real; `api.assemblyai.com/v2/sync` returns 404 — Sync lives on `sync.assemblyai.com/transcribe`.
- The Mic card prints `"16 kHz mono · ready · latency low"` as a literal (`main.py:611`) —
  `self._stream.latency` is never read.
- The Context card shows "Reading foreground app" while UIA is disabled-for-the-session (C12).
- `Refiner.articulate_mode` (`refine/refiner.py:91`) is a real, documented product mode with no
  UI at all. Streaming, context, auto-learn and `WHISPRFLOW_READ_FIELD` are `.env`-only
  (`main.py:77-81`); `open_profiles`/`open_snippets`/`open_dictionary` are `os.startfile` of raw
  JSON (`main.py:560-581`) with no in-app editor and no reload watcher, and profiles require an app
  restart.
- **Hotkeys are not configurable.** The single most requested control in this product category,
  and `Ctrl+Win` collides with Windows' own desktop shortcuts (C11 is that collision's bug report).
- No "Pause/Off" in the tray. Only Quit. For a system-wide text-injector, a one-click disable is a
  safety control, not a nicety (bank sites, password fields, terminals where a stray paste runs a
  command).
- Dictionary feedback: rejections collapse into one message `(duplicate or too long)`
  (`main.py:519`) covering four different rules (`stt/dictionary.py:120-127`), and the 1 000-term
  cap — the same cap that A2 shows is already throttled to 100 on the default path — is invisible.

### U4 — smaller UI items

- `main.py:216-224` `canvas.bind_all("<MouseWheel>", …)` steals wheel events app-wide, including
  from `log_area` (handled by early-return) and any future dialog; on Linux/macOS `event.delta` is
  pixels, so `int(-1*(delta/120))` scrolls both directions. Bind to the canvas window instead, and
  branch on platform.
- `main.py:789-803` trims the activity log with `delete("1.0","100.0")` — deletes 100 lines when
  1 is over budget; harmless but sloppy. `_ui`/`_safe_call` (`:1333-1339`) swallow every exception
  from every UI callback with no logging: in production a broken pill is indistinguishable from a
  closed one.
- `main.py:470-479` uses a callable label for the tray "Mic:" item; `pystray`'s `MenuItem` text
  callable support is version-dependent and the menu is never refreshed, so it may render the
  repr of a function and then go stale.
- `_log_ui` timestamps are `%H:%M:%S` only; a 30-minute take spanning two minutes looks
  out-of-order next to the "30 seconds left" warning.
- `setup_stt.py` reads/writes `.env` **relative to CWD** (`:17`, `ENV_PATH = Path(".env")`) while the
  app reads `%APPDATA%/WhisprFlow/.env` (`main.py:63-65`). Following the README exactly —
  `python setup_stt.py` then `python main.py` — writes the key where the app does not look. This is
  the same bug `stt/dictionary.py:26-29` was written to fix.

### U5 — accessibility, briefly

Not a checkbox item for this product class, but worth stating: colour is the *only* channel for
state (violet/blue/red/green at low contrast against a near-black pill — the muted
`MUTED=(138,138,148)` hint text on `BG_PILL=(14,14,16)` is ~5:1, fine; the faint
`FAINT=(92,92,102)` card labels are ~2.6:1, below 4.5:1 for small text); there is no audio cue
for success (the beeps are start/stop/lock only, `main.py:820, 999, 1247`) so a deaf user has no
non-visual confirmation, and the pill is invisible to screen readers by construction.

### ⚪ U6 — the right-hand Ctrl and Win keys cannot start or stop a dictation

Found while re-verifying C4, not in the original pass. `hotkey_combo` is `{Key.ctrl_l, Key.cmd}` and
`command_combo` is `{Key.ctrl_l, Key.shift, Key.cmd}` (`main.py:157-158, 165 (baseline)`). In pynput those are
*not* aliases of the right-hand keys — measured against the real enum:

```
Key.ctrl   -> <Key.ctrl: 65507>      Key.ctrl_l  -> <Key.ctrl: 65507>   (same member)
Key.shift  -> <Key.shift: 65505>     Key.shift_l -> <Key.shift: 65505>  (same member)
Key.ctrl_r -> <Key.ctrl_r: 65508>    Key.cmd_r   -> <Key.cmd_r: 65516>  (distinct members)
```

So the left Ctrl and the left Super work, and a user whose hands sit on the right Ctrl, or who uses
the right-hand Windows key, gets nothing: `ctrl_r + cmd` and `ctrl_l + cmd_r` both start no
recording (measured on the real `on_press`). This is not a misfire risk — the exact-set comparison
does its job — it is 100 % silence for a plausible hand position.

**Fix shipped:** `WhisprFlowApp._canonical_key` folds `ctrl_r→ctrl_l`, `cmd_r→cmd`, `shift_r→shift_l`,
`alt_r→alt_l` on both press and release, before the set comparisons. Folding on release matters as
much as on press: a chord started with `ctrl_r` has to be *finished* by it, or the take hangs. The
exact-set semantics stay, so Ctrl+Win+D is still refused. The alias table is built with `getattr` and
cached, because which members a pynput backend exposes is not this app's business.

---

## 6. Security, privacy, robustness

- **S1 — No privacy disclosure at all.** Per dictation, plaintext leaves the machine to
  AssemblyAI (the audio) and Groq (the transcript, the low-confidence words, up to 60 dictionary
  terms, the foreground **app name and window title**, and in Command Mode up to **12 000 chars**
  of the user's selection). The `WHISPRFLOW_READ_FIELD` toggle for focused-field text is
  thoughtfully off-by-default (`app_context.py:146`), but window-title reading — which can
  carry document names, email subject lines, patient names, `Private browsing - …` — is on by
  default and is not mentioned in the README. There is no privacy section, no data-retention note
  about the vendor, and no way to see what would be sent. Fix: add a README/docs privacy section
  and a "what gets sent" disclosure in the Context card, listing app+title+terms.
- **S2 — Keys in plaintext, and project-local `.env` can override them.** `main.py:64-65` calls
  `load_dotenv(ENV_PATH)` then `load_dotenv()`, so a `.env` in the current working directory (or
  in the PyInstaller temp dir) **overrides the user's real keys** — a write-once-to-a-temp-dir
  path is a credential-substitution vector, and it also means running the app from a repo checkout
  can silently change which account is billed. `python-dotenv` doesn't override existing env by
  default, but `load_dotenv()` here is called *after* the first, so the second file wins for any
  key not yet in `os.environ`. Fix: drop the second `load_dotenv()`, or use
  `override=False` semantics explicitly and document that dev `.env` is only read when
  `WHISPRFLOW_DEBUG=1`. On Windows, prefer `wincred`/DPAPI (`win32crypt.CryptProtectData`) for the
  stored key.
- **S3 — Prompt-injection surface is instructions-only.** `refiner.py:204-221` interpolates raw
  transcript, dictionary terms, profile instruction and history into the prompt with no delimiter
  hardening; `ARTICULATE`/profile instructions are appended to the *system* prompt
  (`:196-201`), so anything that can write to `profiles.json` or the user dictionary gets a
  system-prompt line. The `max_tokens` + guard limits damage to text, but the text is then
  **pasted into the user's document**. The guard's length check is the only real barrier. Fix:
  fence the user payload (`<TEXT>…</TEXT>`, with any `</TEXT>` inside escaped — one line in
  `_build_input`), keep dictionary/profile content out of the system role, and never treat
  `finish_reason`-length output as trustworthy.
- **S4 — `learned.json` is a plaintext vocabulary ledger of everything dictated**, up to 500
  tokens with counts, retained forever, no expiry, no wipe button (`context/learner.py`). The
  module argues at length that keylogging would be a worse privacy trade — fair — but the file it
  writes *is* a soft transcript store (it contains "password1" if you dictate it in a context
  where it looks like a term, since `WORD_RE` accepts digits: `:53`). Fix: cap to tokens seen ≥2
  times, drop anything matching a secret-ish shape, add an age cutoff and a "clear learned words"
  action next to the suggestions.
- **S5 — Injection is best-effort with no confirmation.** `_type_directly` (`injector.py:84-92`)
  types up to 120 chars with no inter-key delay and returns `True` on "no exception" — which on
  Windows is not the same as "arrived". Some apps (games, elevated windows, some Electron fields)
  drop synthetic keystrokes under load. Fix: for `DIRECT_TYPE_LIMIT`, verify by reading back via
  UIA `TextPattern` when available, or reduce the limit and always prefer clipboard for
  anything where a dropped char matters. Also add a tiny `time.sleep(0.001)` per char — measured
  cost is nothing (120 chars → 0.12 s) and it is the standard fix for dropped-synthetics.
- **S6 — Clipboard restore is time-based, not event-based.** `injector.py:118-130` restores the
  user's previous clipboard after a fixed 0.5 s; `selection.py:36,138` uses 0.45 s and a fixed
  `COPY_SETTLE = 0.14` s wait for the copy to land. Two problems: (a) if the user copies something
  in that 0.5 s window, their new clipboard is destroyed; (b) if the target app services Ctrl+C in
  more than 140 ms (Electron and Office do), `capture()` reads the sentinel back and reports
  "No text selected". Fix: poll the clipboard for a change in ~10 ms steps until the settle budget
  (with `AddClipboardFormatListener`/`GetClipboardSequenceNumber` to detect our own writes), and
  before restoring, verify the sequence number still matches what we wrote.
- **S7 — Blocking work on the keyboard hook thread.** `on_press` → `start_command_mode` runs
  `selection.capture()` (`main.py:902`), which blocks on `wait_for_modifier_release()` (up to
  0.4 s, `injector.py:53-59`) plus `COPY_SETTLE` + two `pyperclip` calls, all on the pynput hook
  callback thread. Windows silently uninstalls a low-level hook whose callback overruns its
  timeout — the failure is "dictation stopped working" with no error. Fix: `on_press`/`on_release`
  must only enqueue; do the work on a worker thread (the file-level docstring of `injector.py`
  shows the author knows this territory — the hotkey path doesn't follow it).
- **S8 — Robustness nits.** `_poll` retries the poll GET with no backoff-on-error
  (`assemblyai_client.py:181-198`): a transient `ConnectError` aborts an otherwise healthy job
  and (via `_pipeline`) the user's take; add 2 retries with jitter and reuse the transcript id.
  `StreamingSession.feed` swallows queue errors with a bare `except Exception: pass`
  (`stt/streaming.py:145-152`) — audio can vanish silently while the session still reports `ok`.
  `audio/capture.py:157-158` counts xruns but no caller ever reads `get_stats()["overflows"]`.
  `main.py:1301` `os._exit(0)` (see C8).

---

## 7. Quality, tests, hygiene

**Coverage map (from the collected suite, 191 tests):** `audio/process.py` and
`audio/capture.py` — good, including the exact things the old audit broke (preroll, ring wrap,
bounded recording, gain cap, transient-click levelling). `refine/guard.py` — 10 tests, covering the
four implemented rules and missing the five blind spots in A3. `stt/assemblyai_client.py` — parse,
keyterm gating, error paths, no HTTP-level tests beyond `eval/mock_api_test.py` (which is a
genuinely good integration test with a mock server). `stt/streaming.py` — turn assembly,
format/unformat dedup, confidence; **no test that a `Terminate` frame is sent, no test of the
connect params, no test of `close_and_finalise` timeouts.** `refine/commands.py` — output cleaning,
validate, canonicalisation, size limits. `context/*` — profiles, learner, snippets, context
formatting. `ui/overlay.py` — **only `Spring`**.

**What has no test at all:** `injector.py` (`TextInjector`), `selection.py`
(`SelectionManager`) — both are the "silently destroys the user's text" modules — `FloatingPill`
state machine and hit regions, `setup_stt.py`, `create_shortcut.py`, `WhisprFlow.spec`, and the
**hotkey state machine**: `eval/smoke_test.py:216-281` exercises it but through a stubbed pynput
with string keys (`smoke_test.py:88-90`), which cannot express tap-vs-hold timing, the
`Key.cmd`/`cmd_l`/`cmd_r` identity subtleties, or C4/C11. `test_audio.py:296-320` then re-implements
`on_release`'s decision as a local `_decide()` helper — a test that cannot fail even if `main.py`
deletes the code it mirrors.

Recommendations, in order of value per minute:

1. **Test against real pynput `Key` objects.** `test_audio.py:296-320`'s `_decide` mirror should be
   replaced by importing `WhisprFlowApp` and calling `on_press`/`on_release` with a fake clock
   (`monkeypatch time.monotonic`). That single change turns C4, C11 and the tap/lock contract into
   tested behaviour. Add the three regression tests for the findings above (Esc order, extra key,
   right-hand modifiers).
   *Shipped as `test_hotkey.py` (23 tests, real `on_press`/`on_release`/`_anchor_ok`/`_inject`,
   peripheral stubs only, an `IntEnum` whose aliases mirror pynput's so the folding is tested against
   a faithful key space). The `test_audio.py` mirror is still open.*
2. **Tests for the two untested data-loss modules.** For `TextInjector`: paste-path selection
   (≤120 direct type, >120 clipboard), restore-after-failure, and `inject` returning `False` on a
   clipboard exception. For `SelectionManager`: sentinel logic with a fake clock — slow app
   (copy lands at 200 ms) must be detected, not reported as "No text selected".
3. **Guard tests for A3's blind spots** (antonym pair, entity swap, added numbers, `word42`
   digit noise, `</TEXT>` fence) — each is one `assert not check(...)`, and each documents intent.
4. **A `Terminate`-is-sent test** and a **connect-params test** (`speech_model` present) — both
   cheap, both pinning C2/C5.
5. **Wire `run_wer.py` into CI** (A4) and publish the number in the README.
6. **Add `pytest-cov` with a floor**, e.g. `--cov=refine --cov=audio --cov=stt --cov-fail-under=85`,
   and move `test_*.py` into `tests/`. Add `pyproject.toml` with `[tool.pytest.ini_options]`
   (`asyncio_mode`, markers), `[tool.ruff]` (the CI installs `pyflakes` for undefined names —
   `ruff` gives that plus unused imports and complexity in one tool), `[tool.mypy]` on the
   dataclasses-only modules (`audio/`, `refine/`, `stt/` are already `from __future__ import
   annotations`; `main.py` is not annotated and would need work).
7. **Repo hygiene.** 33 tracked `__pycache__/*.pyc` + a committed `.pytest_cache/` +
   `@AutomationLog.txt`, all of which are in or should be in `.gitignore`
   (`__pycache__/` is at line 2 — they were committed anyway):
   `git rm -r --cached __pycache__ */__pycache__ .pytest_cache @AutomationLog.txt`.
   Drop the dead `*.onnx`/`models/*` rules (line 51-68, leftovers from the SenseVoice era) or keep
   them and say why.
8. **Docs vs reality.**
   - `README.md:1` and `eval/*` point at `mabeldesign008-code/flow`; the download link, the clone
     URL, and the "report this" URLs in `main.py:1411-1421` all point at the **other** repo
     (`flow` has a published release, `whisper-dev` has **zero** — checked via the GitHub API).
     Anyone following the README from this repo installs from a different repo than the one they
     cloned, and pushing a `v*` tag here builds nothing users can get.
   - "133 unit tests" (`README.md`) — it is 191 here.
   - `README.md` claims streaming gives "~0.5 s tail" and "both paths hit the same model": both
     wrong today (C2, §3).
   - No app version anywhere except `version_info.txt` (hard-coded `1.0.0.0`). `main.py` has no
     `__version__`; the log, the tray tooltip and the fatal dialogs cannot tell you what build a
     user is running. Set `__version__` once and put it in the window title, the log header, and the
     `_fatal()` body; derive `version_info.txt` from the tag in `release.yml`.

---

## 8. Prioritised plan

**Week 1 — stop the bleeding (all small, all testable)**

| | Change | LOC |
|---|---|---|
| 1 | Groq model IDs → `openai/gpt-oss-20b` / `openai/gpt-oss-120b`, env-overridable; non-200 → visible error + counter (C1) | ~25 |
| 2 | Pin `speech_model` on the streaming connect params + test (C2) | ~10 |
| 3 | `Terminate` on `abort()` (C5) | ~3 |
| 4 | ~~Esc `pressed_keys` leak (C4)~~ retracted; Esc cancels mid-hold, right-hand modifiers fold (U6), C11 cancel-on-extra-keys + `test_hotkey.py` | ~5 |
| 5 | Check `inject()`'s return value; clear `last_text_injected` on failure (C3.3) | ~5 |
| 6 | Warm `lfilter` at import; drop the `duration<0.25` session leak (P4, C13) | ~8 |
| 7 | `git rm --cached` the pycache/log junk; fix README/issue URLs and the test count (7.7, 7.8) | ~20 |

**Week 2 — don't lose people's text**

8. Focus/caret snapshot + re-verify before `inject()`; selection re-verify before Command Mode
   paste; refuse-with-clipboard-fallback on mismatch (C3.1, C3.2).
9. `finish_reason == "length"` → reject; `allow_restructure` length floor; chunk or skip refinement
   above 600 words (C6).
10. Clear `_history` on cancel/generation and key it per-app (C7).
11. Clipboard: change-detection instead of fixed sleeps, and verify before restoring (S6).
12. `on_press`/`on_release` enqueue-only (S7).

**Week 3 — make it fast**

13. Sync API for the batch path + pre-warm on keypress + adaptive finalise budget (§3 items 1-3),
    plus the live key check U3 asks for (`GET /v2/account`, not `info/auth`) and C14's turn supersede fix.
14. Overlay: drop supersampling to 1×, cache static layers, `itemconfig` instead of PIL, redraw
    gate → target <1 ms/tick (P1).
15. Ring/`tail_since` cursor and `_send_loop` block pass-through (P2, P3).
16. int16 as the canonical take; bound `last_audio`; cap batch length (Memory).

**Week 4 — accuracy you can defend**

17. WER corpus + nightly job + README number + corruption-rate report (A4).
18. Guard: antonym pairs, word-boundary dictionary match, added-number rule, digit-in-identifier
    exclusion (A3.1, A3.4, A3.5).
19. Remove the VAD discard gate (A1); apply high-pass + levelling to streaming (A2).
20. DPI-aware geometry and fonts (U1); elapsed timer + readable success text + pause/mute in the
    tray (U2, S-adjacent); hotkey configuration (U3).

---

## Appendix — quick index

| ID | Finding | Severity | File:line |
|---|---|---|---|
| C1 | Retired Groq models; dictation degrades silently | 🔴 | `refiner.py:84,169`, `commands.py:114` |
| C2 | Streaming pins no model; "same model" claim false | 🔴 | `streaming.py:96-101`, `main.py:1116` |
| C3 | No focus/selection anchor; undo after failed inject | 🔴 | `main.py:1096,1251`, `injector.py` |
| C4 | ~~Esc mid-hold kills the hotkey~~ **retracted**; Esc-mid-hold cancel shipped | ⚪ | `main.py:1207-1212` |
| U6 | Right-hand Ctrl/Win never trigger the hotkey | 🟡 | `main.py:157,165` |
| C5 | No `Terminate` on abort → 3 h billed session | 🟠 | `streaming.py:268-275` |
| C6 | `max_tokens` truncation invisible to guard in restructure mode | 🟠 | `refiner.py:165`, `guard.py:128-136` |
| C7 | Refinement history leaks across takes/apps | 🟠 | `refiner.py:94,219` |
| C8 | `set_api_key` orphans clients; `os._exit(0)` skips cleanup | 🟠 | `assemblyai_client.py:71` +2, `main.py:1301` |
| C9 | `_frame_energies` ignores `sample_rate` | 🟡 | `process.py:146,37-39` |
| C10 | Dead `if`, unconditional `_command_mode` reset | 🟡 | `overlay.py:178-181` |
| C11 | Extra key mid-hold still produces a take | 🟡 | `main.py:1220-1223,1233` |
| C12 | UIA disabled for the session; status lies; thread w/o CoInit | 🟡 | `app_context.py:177,222`, `main.py:376` |
| C13 | Misc (short-take session leak, VAD tone case, `basic_cleanup` comma, device switch on UI thread…) | 🟡 | various |
| C14 | Streaming turns appended, not superseded → duplicate sentences (added while fixing C5) | 🔴 | `streaming.py:41,219-238` |
| P1 | Pill = 50% of a core while recording | perf | `overlay.py:283-299` |
| P2 | `tail_since` quadratic — 13.8 s CPU per 5-min take | perf | `audio/capture.py:243-256` |
| P3 | `_send_loop` concat — 29 GB copied per 5-min take | perf | `streaming.py:157-170` |
| P4 | +0.6 s on the first take (lazy `scipy.signal`) | perf | `process.py:138` |
| P6 | 115 MB held forever by `last_audio` | perf | `main.py:1062` |
| A1 | VAD gate discards takes on the path with least info | accuracy | `main.py:1133-1137` |
| A2 | Streaming = no prep, 100 keyterms, unqualified model | accuracy | `streaming.py:97-181` |
| A3 | Guard blind spots (antonym, entity, tense, added numbers) | accuracy | `guard.py:104-137` |
| A4 | No WER corpus, no corruption rate, harness unused | accuracy | `eval/` |
| U1 | DPI: pill + window half-sized at 200% | UI | `overlay.py:82-92`, `main.py:45` |
| U2 | No elapsed timer, unreadable success, no keyboard access | UI | `overlay.py:319-353` |
| U3 | "Ready"/"latency low"/"Reading foreground app" all hardcoded; no hotkeys, no Articulate toggle, no pause | UI | `main.py:594,611,613,376` |
| S1 | Zero privacy disclosure (titles, 12 KB selections, dict content) | sec | docs + `app_context.py` |
| S2 | Plaintext keys; project `.env` can override user's | sec | `main.py:64-65` |
| S3 | Prompt interpolation without delimiters; config → system role | sec | `refiner.py:196-228` |
| S4 | `learned.json` = permanent plaintext vocabulary ledger | sec | `learner.py:129` |
| S5 | Direct-type returns True on "no exception" | sec/rel | `injector.py:84-92` |
| S6 | Clipboard save/restore on fixed timers | rel | `injector.py:118`, `selection.py:34,36` |
| S7 | Blocking work on the keyboard-hook thread | rel | `main.py:902` → `selection.py:63-66` |
| Q1 | No tests for injector/selection/overlay/pill/hotkey timing; mirrored test | quality | `test_audio.py:296-320` |
| Q2 | README points at a different repo; stale test count; no app version | quality | `README.md`, `version_info.txt` |

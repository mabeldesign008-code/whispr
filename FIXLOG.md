# Fix Log — audit follow-up (branch `fix/audit-2026-09`)

Every entry: finding id → what changed → **external research actually consulted** (fetched
2026-09-13 unless noted) → how it was verified in this Linux sandbox → what still needs
Windows hardware. Key: ⬜ not implemented · ✅ implemented + tested · ⚠️ implemented, needs a
real Windows run to confirm.

---

## Batch 1 — service contracts (model ids, streaming auth, Sync API, key validation)

### C1 · Retired Groq model ids made every refinement fail in silence — ✅
`refine/refiner.py`: default `llama-3.1-8b-instant` → `openai/gpt-oss-20b`, overridable with
`GROQ_REFINE_MODEL`. `refine/commands.py`: `llama-3.3-70b-versatile` → `openai/gpt-oss-120b`,
overridable with `GROQ_COMMAND_MODEL`. Both now keep a **candidate chain**
(`Refiner.models` / `CommandProcessor.models`) instead of one hardcoded id, and the retired ids
stay last in it rather than being deleted.

Research: Groq model deprecation page, "August 16, 2026: llama-3.1-8b-instant and
llama-3.3-70b-versatile … we recommend migrating to `openai/gpt-oss-20b` (for 8B Instant) and
`openai/gpt-oss-120b` or `qwen/qwen3.6-27b` (for 70B Versatile)" —
<https://console.groq.com/docs/deprecations>. Cross-checked against a July-2026 benchmark table
listing both retired ids with `Deprecating 08/16/26` and gpt-oss 20B at ~1,000 tok/s
(<https://markaicode.com/benchmarks/groq-production-benchmark-latency/>), and Groq's own Batch
API page listing `openai/gpt-oss-20b` / `openai/gpt-oss-120b` as current
(<https://console.groq.com/docs/batch>).

Verified here: new `TestModelFallback` (3) + `test_uses_a_reasoning_grade_model_with_fallbacks` +
`test_ships_no_deprecated_model_ids` — a 400 "not a valid model" from the first id falls back to
the second, the dead id is remembered (`health.is_dead`) and pushed to the back of the chain on
the next call. 195 tests pass.

Not checkable here: that the four ids resolve live for *this user's* account (a key is required;
`GET /v1/models` returns `invalid_api_key` from the sandbox).

### C1 (second half) · New `refine/api_health.py` — ✅
`ApiHealth` counters (`calls/failures/http_errors/timeouts/fallbacks`, `last_error`, dead-model
memory, `ordered()` candidate ordering, `summary()`), plus a shared `error_detail(resp)` helper.
Motivation recorded in the module docstring: both Groq callers previously mapped *any* non-200 to
`basic_cleanup()` with no trace anywhere. Verified: used by both callers; `summary()` shown in
Settings.

### U3 · Status text was hardcoded reassurance — ✅
`main.py`: the refinement row now reads
`{model_used} · NOT refining: {last_api_error}` (danger colour) when the last provider call failed,
`{model_used} · guard blocked n/m (raw text kept)` on guard rejections; the mic row replaced the
constant `latency low` with a rolling `last N ms · avg M ms` measured from
`TranscriptionResult.latency_ms` (`main.py` `_latency_ms` deque). Verified: `pytest`,
`eval/smoke_test.py` (builds the whole Tk window, so the new `config(...)` kwargs are exercised).

### C2 · Streaming never named a speech model — ✅
`stt/streaming.py`: `speech_model` is now a connection parameter, chosen by
`streaming_model_for(batch_model)` from the *same* id the batch client uses (`main.py` passes
`batch_model=self.stt.model`), and the app verifies the server's echo.

Research: <https://www.assemblyai.com/docs/streaming/select-the-speech-model> — `speech_model` is
optional and "if omitted, the session defaults to `universal-3-5-pro`"; only three ids exist
(`universal-3-5-pro`, `universal-streaming-english`, `universal-streaming-multilingual`).
<https://www.assemblyai.com/docs/streaming/message-sequence> — "Unrecognized or misspelled query
parameters are ignored rather than rejected, so this echo is the fastest way to catch a typo";
`Begin` carries `configuration.model` and the session `id`.
So the risk was not "silently downgraded today" but "silently *changes* when the account default
changes, and silently diverges from the batch fallback whenever `ASSEMBLYAI_MODEL` is overridden" —
pinning plus the echo check is what closes both.

Verified: `test_connection_params_name_the_model`, `test_batch_only_models_map_to_a_realtime_id`,
`test_begin_echo_mismatch_degrades_the_session`, `test_begin_records_session_id_when_it_matches`.

### C5 · `Terminate` without reading the flush — ✅
`close_and_finalise()` now sends `ForceEndpoint` → drains the pump → reads for 1.5 s → sends
`Terminate` → reads again until `Termination` (or a bounded grace) → only then closes. `abort()`
also sends `Terminate` and reads for 0.35 s instead of killing the socket mid-flush.
Research: <https://www.assemblyai.com/docs/streaming/message-sequence#session-termination> —
"After you send `Terminate`, **keep reading from the WebSocket until you receive the
`Termination` message**. The server first flushes any in-flight messages, which can include: the
final (and formatted) `Turn` for audio you already sent … **Closing the socket as soon as you send
`Terminate` silently discards your last transcript.**"

### C13 · `stop_recording` leaked the streaming session on a short take — ✅
Every early return now aborts the session (report C13, first bullet): a session
left open is billed on connection duration, unread, and capped at 3 h.

### C14 · Streaming turns were appended, not superseded — ✅
_new finding, added to the report as C14 while fixing C5_
Vendor text: "Within a turn, each `Turn` message supersedes the previous one. Render the latest
`transcript`; **do not append**. A turn is complete on the message where both `end_of_turn` and
`turn_is_formatted` are `true`", and with `format_turns=true` "you receive two `end_of_turn: true`
messages for the same `turn_order`: the unformatted final first, then a formatted final right
after … or you'll process every turn twice."
`_handle_turn` keyed turns by `turn_order` (dict, so re-delivery overwrites instead of
duplicating), completed a turn only on `end_of_turn && turn_is_formatted`, and rebuilt text in
turn order. Word lists are taken from the final message only, so confidences no longer double up.
A turn that never got its formatted copy is still used as best-available text rather than
discarded, because "the tail of my sentence disappeared" is worse than unpunctuated text.
Also from the same page: PCM frames should be ~50 ms — `CHUNK_MS` 100 → 50, which halves what a
cancel can lose, and the send loop's buffer became a `bytearray` (see P4).
Verified: `TestStreaming` is now 16 tests covering supersede, dedupe, repeat-final, word
confidences and the terminate ordering.

### §3.1–3.3 · Batch-only transcription: three round trips + a poll floor — ✅
New `stt/sync_transcribe.py` (Sync endpoint + live upload), wired into
`AssemblyAIClient.transcribe`, which now prefers, in order: a finished **live
upload** transcript → **Sync** one-shot → **async** upload/submit/poll. Takes
over the documented 120 s cap go straight to async (`sync_supported`).
Async polling also stopped sleeping a fixed 200 ms before its first look
(`poll_interval` default 0.2 → 0.05, growth 1.35×, ceiling 1 s) and now
handles 429/5xx/transport errors inside the poll loop instead of aborting the take.

Measured here (localhost mock with a modelled **60 ms RTT** per request, median of 7
takes, "wait after the user stops speaking"):

| path | median |
| --- | --- |
| async, old fixed 200 ms first poll | **784 ms** |
| async, adaptive poll | **431 ms** |
| Sync one-shot (1 RTT) | **63 ms** |

The 63 ms is the request *shape* win (3 round trips → 1), not a claim about
real inference time — AssemblyAI's own example response reports
`request_time_ms: 243.7` for a 101 s clip. The live-upload number came out the
same as one-shot here because the harness feeds audio in a tight loop instead of
in real time; its win is upload-bytes-during-speech, which needs a real
microphone (or a paced harness) to measure.

Research:
- endpoint/auth/multipart contract and required header:
  <https://www.assemblyai.com/docs/api-reference/sync-api/transcribe> —
  `POST https://sync.assemblyai.com/transcribe`, `X-AAI-Model` **required**
  (canonical `universal-3-5-pro`), `Authorization` raw key (Bearer optional),
  multipart `audio` part typed `audio/wav` or `audio/pcm`, optional `config`
  JSON part; response `text`, `words[]`, `confidence`, `audio_duration_ms`,
  `session_id`, `request_time_ms`.
- limits: <https://www.assemblyai.com/docs/sync-stt/audio-requirements> —
  80 ms … 120 s, ≤40 MB, 16-bit, mono/stereo, sample rates
  {8000,16000,22050,24000,32000,44100,48000}, and "For raw PCM, pass
  `sample_rate` and `channels` in the config part"; longer than 120 s →
  Pre-recorded (which is what the client now falls back to).
- live upload: <https://www.assemblyai.com/docs/sync-stt/getting-started/transcribe-live-audio>
  + <https://www.assemblyai.com/docs/api-reference/sync-api/transcribe-live> —
  `POST /v1/transcribe/live`, `config` part **required and first, ahead of
  `audio`**, `audio` typed `audio/pcm`, "upload audio as your code produces
  it, so authorization, the upload, and every speech segment but the last are
  done by the time the speaker stops". Because httpx's `files=` is
  buffered/one-shot, the live body is hand-encoded and streamed with
  `content=`. It is **off by default** (`WHISPRFLOW_SYNC_LIVE_UPLOAD=1`),
  documented in `docs/sync-live-upload.md`, and only opened when the streaming
  WebSocket is not already carrying the take.
- errors: <https://www.assemblyai.com/docs/sync-stt/error-handling> —
  `{"error_code","message"}` for audio/capacity/inference vs `{"detail"}` for
  auth/rate-limit; 429/503 transient and honour `Retry-After`; "400, 413, and
  415 indicate a problem with the request itself"; "500 and 504 are safe to
  retry once". Implemented as: one retry for 429/503 (≤ `Retry-After` 1 s,
  capped) and 500/504, no retry for 400/413/415 → fall back to async.
- pre-warming: <https://www.assemblyai.com/docs/sync-stt/connection-pre-warming>
  — `GET /warm` unauthenticated, "the ideal moment … is when you know audio is
  coming but don't have it yet", "httpx drops idle connections after 5 seconds
  by default", and "pre-warming only helps if the /warm and /transcribe
  requests share a connection pool". Implemented as `SyncTranscriber.warm()`
  on the *shared* client, called at hotkey press, with `keepalive_expiry=60`
  and `pool=2.0` so a stall cannot masquerade as a provider timeout.
- keyterms/prompt caps: <https://www.assemblyai.com/docs/sync-stt/prompting-and-keyterms>
  — `keyterms_prompt` max **100 terms / 8000 characters** total, prompt ≤ 6000
  chars, and "including a large number of terms or common terms … could lead to
  overcorrections and hallucinations", plus the note that `language_code` is
  ignored when a custom `prompt` is set. Implemented as `sync_keyterms()`
  (drops obvious filler singletons, caps 100/8000) and `SyncConfig.as_json()`
  omitting `language_code` whenever a prompt is sent.

Verified: `eval/mock_api_test.py` sections 6–9 (37 new checks) drive the mock
through the real `AssemblyAIClient`, asserting the header set (`authorization`
without `Bearer`, `X-AAI-Model`), multipart part order and per-part content
types, raw-PCM byte count (`2 × samples`, no RIFF header), `sample_rate`/
`channels` in the config part, prompt forwarding, keyterm filtering, the
>120 s → async route, 400 → async fallback, 401 → immediate "rejected the key"
with exactly one request, 503 → one retry honouring `Retry-After`, live-upload
config-first ordering, and `GET /warm`. Plus 14 unit tests in
`TestSyncClient`/`TestKeyValidation`. 222 pytest tests, smoke test, import
probe and pyflakes all green.

### U3 (second half) · No key validation — ✅
`AssemblyAIClient.validate_key()` calls `GET /v2/account` and returns
`(True/False/None, message)`; `main.py`'s save button now runs it and reports
"Key verified (credits_amount=…)" / "AssemblyAI rejected that key" *before* any
audio is sent, with `None` reserved for "couldn't reach the API" so an offline
machine is never told its key is wrong. `session_id` is kept on every result
(`stt/base.py`) and shown on the status page, since that is what support asks
for when a transcript is wrong.

Research: <https://www.assemblyai.com/blog/speech-to-text-api-fundamentals> —
"send your API key in the authorization header … **no Bearer prefix** … If your
key is valid, you get a 200 … If it's missing or wrong, you get a 401. That's
your authentication smoke test before you send any audio";
<https://docs.redhuntlabs.com/docs/exposure-risks/credentials/assemblyai_api_key> —
`curl -X GET "https://api.assemblyai.com/v2/account" -H "authorization: [KEY]"`
as the way to verify a key is active. `session_id` requirement from the error
page above.

⚠️ Not verifiable from this sandbox: the endpoint itself. `api.assemblyai.com`
answers 401 to an unauthenticated probe (endpoint alive), but there is no key
here, so `Sync`/`/v2/account` behaviour is implemented **to the documented
contract** and exercised against a mock that mirrors that contract, not against
the live service.

---


---

## C3 — focus anchor, and believing `inject()` (Batch 2)

**Files:** `main.py` (`_focus_signature`, `_foreground_is_ours`, `_anchor_ok`, `_inject`,
`stop_recording`), `injector.py` (`stage`), `test_hotkey.py`

**Research.** The anchor needs the two handles Win32 exposes for free. `GetForegroundWindow`
"retrieves a handle to the foreground window (the window with which the user is currently
working)" and — the part that decides the code's shape — "can be **NULL** in certain circumstances,
such as when a window is losing activation"
([learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getforegroundwindow](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getforegroundwindow)).
`GetGUIThreadInfo(0, …)` with `idThread` 0 "returns information for the foreground thread", giving
`hwndFocus` — the control that would actually receive the paste — and its remarks warn twice: "the
function may not return valid window handles in the `GUITHREADINFO` structure when called to retrieve
information for the foreground thread, such as when a window is losing activation", and "for an edit
control, the returned **rcCaret** rectangle contains the caret plus information on text direction and
padding. Thus, it may not give the correct position of the cursor"
([learn.microsoft.com/windows/win32/api/winuser/nf-winuser-getguithreadinfo](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getguithreadinfo)).

Three design consequences, each traceable to a sentence above:

* A `0` from either call means *unknown*, never *changed*. The comparison refuses only when both
  snapshots are real, or the app would drop valid pastes exactly when Windows is mid-activation.
* `rcCaret` is not used. The caret rectangle is documented as unreliable for edit controls, so a
  caret-position signature would produce false refusals in the very app class that matters most;
  `hwndFocus` is stable, cheap, and catches the "clicked into another field" case.
* UI Automation is not used. `selection.py` already shows what a UIA round trip costs in latency and
  failure modes, and the anchor is sampled twice per take on a timing-critical path.

**Change.** `stop_recording` snapshots `(hwnd, hwndFocus, pid)` on the keyboard-hook thread — the
only moment that can be sampled with no extra machinery, and the moment the user's intent is defined.
Before writing, `_anchor_ok` re-samples and refuses if the top-level window changed or the focused
control changed. If the new foreground window belongs to *this* process (`pid == os.getpid()`) the
refusal is suppressed: the pill or the settings window coming forward is not the user changing apps,
and treating it as such would refuse every paste the moment a tooltip appears. On refusal the text
goes through the new `injector.stage()` — clipboard, no keys — so refusing never means losing. In
Command Mode the captured selection is re-captured and compared, because a click, an arrow key or an
Escape in the target app drops the highlight while the LLM runs and the paste then *inserts* the
rewrite beside the original instead of replacing it.

`_inject` now reads `inject()`'s return value. It has always returned a bool; `main.py` discarded it,
so a failed clipboard write or a failed paste still set `last_text_injected`, and the next
`Ctrl+Alt+Z` undid whatever the user had really last changed in the document — a failure that is
invisible until it destroys something.

## C11 — a chord with an extra key is not a dictation

**Files:** `main.py` (`on_release`), `test_hotkey.py`

`on_press` refuses to start on Ctrl+Win+<anything> because the combo test is an exact set match, but
*stopping* had no such rule: `Ctrl+Win+D` (new virtual desktop), `Ctrl+Win+arrows` (switch desktop)
started a take on the way in and submitted it on the way out — measured: `events=['start','stop']`
with a 1.2 s take handed to the pipeline. `on_release` now computes
`pressed_keys - the combo actually in use` before deciding, and cancels if anything but Esc is left.
`command_combo` is subtracted in command mode so Ctrl+Shift+Win does not look like an intrusion
(`shift` belongs to one combo and not the other — that asymmetry is the whole bug).

## U6 — right-hand Ctrl and Win never fired

**Files:** `main.py` (`_key_aliases`, `_canonical_key`, `on_press`, `on_release`), `test_hotkey.py`

`hotkey_combo` is `{ctrl_l, cmd}`, and in pynput the bare `Key.ctrl`/`Key.shift` *are* the left-hand
members while `ctrl_r`/`cmd_r`/`shift_r`/`alt_r` are distinct ones — verified in this environment:
`Key.ctrl -> <Key.ctrl: 65507>` and `Key.ctrl_l -> <Key.ctrl: 65507>` are the same member, while
`Key.ctrl_r -> <Key.ctrl_r: 65508>` is not. Pressing the right Ctrl with the left Super, or the left
Ctrl with the right Super, started nothing (measured against the real `on_press`).

`_canonical_key` folds the right-hand members onto their left-hand twins before every comparison, on
release as well as press — folding only the press would leave a take started by `ctrl_r` unfinishable.
The alias table is built with `getattr` and cached, since which members exist is a property of the
pynput backend (and of test stubs), not of this app. Exact-set semantics are untouched: Ctrl+Win+D
still does nothing.

## C4 — retracted: the Esc leak is not in the code

**Files:** `AUDIT_PERF_UI_ACCURACY_2026-09-13.md`, `test_hotkey.py`

The finding said a non-locked Esc falls through into `pressed_keys`. It does not: the `return` in the
Esc branch is outside the `locked` guard, and `pressed_keys` measured empty in all three Esc
orderings with the hotkey working afterwards. The original reproduction read `is_recording` right
after the next Ctrl+Win press while the session was in tap-to-lock state — that press legitimately
*finishes* the locked take, so `is_recording=False` was misread as a dead hotkey. Instrumenting which
of `start/stop/cancel_recording` fired is what separated the two.

Two real items came out of the re-verification and are shipped: `Esc` now cancels an ordinary hold as
well as a locked one (it required `locked`, so the documented panic key did nothing mid-hold), and
`test_hotkey.py` pins `pressed_keys` across the orderings so the claim cannot come back. **No fix was
made to the add-gate, because there was nothing to fix** — and the guard the finding proposed adding
(`if key in combo … or self.is_recording`) would have been dead code that *creates* the leak it
fears, by moving the `add` above the `return`.

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

### C2 · New `refine/api_health.py` — ✅
`ApiHealth` counters (`calls/failures/http_errors/timeouts/fallbacks`, `last_error`, dead-model
memory, `ordered()` candidate ordering, `summary()`), plus a shared `error_detail(resp)` helper.
Motivation recorded in the module docstring: both Groq callers previously mapped *any* non-200 to
`basic_cleanup()` with no trace anywhere. Verified: used by both callers; `summary()` shown in
Settings.

### C3 · Status text was hardcoded reassurance — ✅
`main.py`: the refinement row now reads
`{model_used} · NOT refining: {last_api_error}` (danger colour) when the last provider call failed,
`{model_used} · guard blocked n/m (raw text kept)` on guard rejections; the mic row replaced the
constant `latency low` with a rolling `last N ms · avg M ms` measured from
`TranscriptionResult.latency_ms` (`main.py` `_latency_ms` deque). Verified: `pytest`,
`eval/smoke_test.py` (builds the whole Tk window, so the new `config(...)` kwargs are exercised).

### C10 · Streaming never named a speech model — ✅
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

### C11 · `Terminate` without reading the flush — ✅
`close_and_finalise()` now sends `ForceEndpoint` → drains the pump → reads for 1.5 s → sends
`Terminate` → reads again until `Termination` (or a bounded grace) → only then closes. `abort()`
also sends `Terminate` and reads for 0.35 s instead of killing the socket mid-flush.
Research: <https://www.assemblyai.com/docs/streaming/message-sequence#session-termination> —
"After you send `Terminate`, **keep reading from the WebSocket until you receive the
`Termination` message**. The server first flushes any in-flight messages, which can include: the
final (and formatted) `Turn` for audio you already sent … **Closing the socket as soon as you send
`Terminate` silently discards your last transcript.**"

### NEW · C14 (found while fixing C11) · streaming turns were appended, not superseded — ✅
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

### C12/C13 · Sync endpoint, adaptive polling, key validation — ⬜ next commit
Research already done (contracts in the entries below this line and in the audit report); code has
not been written yet.

---

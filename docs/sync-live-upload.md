# Sync live upload (`WHISPRFLOW_SYNC_LIVE_UPLOAD=1`)

**Status: implemented, off by default.** This page exists because the option
changes what leaves your machine while you are still speaking, and because it
interacts with the streaming socket — you should read this before turning it on.

## The three paths

| path | request shape | when it wins |
|---|---|---|
| **Sync one-shot** (default) | 1 POST after you release the key | short takes, no second connection, nothing on the wire while you talk |
| **Sync live upload** | 1 chunked POST that starts at key-press | you want the tail of the wait gone and you are not using the socket |
| **Real-time socket** | WebSocket for the whole take | you want partial text in the pill *while* you speak |

AssemblyAI's own framing for the live path
([docs](https://www.assemblyai.com/docs/sync-stt/getting-started/transcribe-live-audio)):

> `transcribe()` needs the whole clip before it can send anything. Live upload
> starts the request immediately and uploads audio as your code produces it, so
> authorization, the upload, and every speech segment but the last are done by
> the time the speaker stops. What is left to wait for is the final segment.

That is exactly the delay this app exists to remove, so it is worth
understanding why it is not simply on.

## Why it is off by default

1. **It duplicates the socket, and the socket is better.** With
   `WHISPRFLOW_STREAMING=1` (default) the audio is *already* uploaded live and
   you additionally get partial text in the pill. Running both means sending
   the same speech twice, to two endpoints, and being billed for two sessions —
   AssemblyAI's real-time API is billed on connection duration. So the app opens
   the live Sync upload **only when the socket is not carrying the take**
   (streaming disabled, `websockets` missing, or command mode).

2. **A take you cancel is still uploaded.** Cancellation is handled (the
   request is terminated and the transcript discarded), but "no text leaves
   your machine" is not the property of a live upload, and the app's default
   should be the least-surprising one.

3. **The 120 s cap becomes load-bearing.** Sync rejects anything longer than
   120 s (`413 audio_too_large`). The live session therefore stops feeding at
   the cap and marks itself failed, and the pipeline falls back to
   upload+submit+poll. That is correct, but it means a long dictation makes two
   requests instead of one.

## Contract notes (as implemented)

- `POST https://sync.assemblyai.com/v1/transcribe/live`, `Authorization: <key>`
  (raw key, no Bearer), `X-AAI-Model: universal-3-5-pro` on every request.
- The body is multipart, and for this endpoint the **`config` part is required
  and must be flushed before the `audio` part** — an empty `{}` is fine.
  Because httpx's `files=` encoder buffers the whole body, the live path
  hand-encodes the multipart frame (`LiveSyncSession._iter_body`) and streams
  it via `content=`.
- Raw S16LE PCM needs `sample_rate` and `channels` in `config`; a WAV header
  would have to describe a length that is not known yet, so the live path never
  wraps audio in WAV.
- `Retry-After` is honoured for 429/503; 400/413/415 are treated as "this
  request is wrong" and fall back rather than retry.
- The `session_id` in every response is kept on the result and shown in
  Settings, because that is what support asks for.

## If you turn it on

```
WHISPRFLOW_SYNC_LIVE_UPLOAD=1
WHISPRFLOW_STREAMING=0        # otherwise the socket already covers it
```

Check the Settings window after a few dictations: the Transcription card shows
which path served the takes ("Sync ≤120 s · live upload · N fast, M fell back")
and any provider error, so a silent fallback is visible rather than inferred.

## Not yet verified on hardware

The request shape, part order, byte counts and error handling are exercised
against the mock in `eval/mock_api_test.py` (section 8). What has *not* been
measured is the real-world tail latency, because that needs a microphone, a
network and a key. To measure it properly, pace the feed at 1× real time (a
tight loop just measures the request shape, not the overlap) and compare
`request_time_ms` with the wall-clock time between releasing the hotkey and the
text appearing in the target field.

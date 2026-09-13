"""
Integration check: run AssemblyAIClient against a fake AssemblyAI server.

Verifies the real upload -> submit -> poll HTTP flow, including that
keyterms are sent, WAV bytes arrive intact, and errors surface correctly.
Run: python eval/mock_api_test.py
"""

import asyncio
import io
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stt.assemblyai_client import AssemblyAIClient  # noqa: E402

STATE = {"uploaded": b"", "submitted": {}, "polls": 0, "mode": "ok"}


class Handler(BaseHTTPRequestHandler):
    def _read_chunked(self):
        """HTTP chunked-transfer decode. httpx streams a body without a
        Content-Length, so the live-upload test needs this."""
        out = b""
        while True:
            line = self.rfile.readline().strip()
            if not line:
                continue
            try:
                size = int(line.split(b";")[0], 16)
            except ValueError:
                break
            if size == 0:
                trailer = self.rfile.readline()
                while trailer.strip():
                    trailer = self.rfile.readline()
                break
            out += self.rfile.read(size)
            self.rfile.read(2)
        return out

    def _sync_like(self, live=False, raw=None):
        """Mirror POST /transcribe (and /v1/transcribe/live): split the real
        multipart body on the boundary the client actually used and record
        what arrived, so the checks below assert on the vendor contract (part
        order, per-part content types, raw-PCM byte count) rather than on our
        own encoder agreeing with itself."""
        ctype = self.headers.get("content-type", "")
        body = raw or b""
        parts, order, headers = _parse_multipart(ctype, body)
        STATE.setdefault("sync", []).append({
            "auth": self.headers.get("authorization"),
            "model": self.headers.get("x-aai-model"),
            "ctype": ctype,
            "body": body,
        })
        STATE["sync_parts"] = parts
        STATE["sync_order"] = order
        STATE["sync_headers"] = headers
        STATE["sync_config"] = json.loads(parts["config"]) if parts.get("config") else {}
        STATE["sync_audio_len"] = len(parts.get("audio", b""))

        if STATE["mode"] == "sync_error":
            return self._json(400, {"error_code": "bad_audio",
                                    "message": "misaligned PCM"})
        if STATE["mode"] == "sync_capacity":
            payload = json.dumps({"error_code": "capacity_exceeded",
                                  "message": "at cap"}).encode()
            self.send_response(503)
            self.send_header("retry-after", "1")
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if STATE["mode"] == "sync_badkey":
            return self._json(401, {"detail": "unauthorized"})
        if STATE["mode"] == "sync_too_long":
            return self._json(413, {"error_code": "audio_too_large",
                                    "message": "duration exceeds limit"})
        return self._json(200, {
            "text": "Deploy the Kubernetes cluster to production.",
            "words": [
                {"text": "Deploy", "confidence": 0.99, "start": 0, "end": 300},
                {"text": "the", "confidence": 0.98, "start": 310, "end": 400},
                {"text": "Kubernetes", "confidence": 0.42, "start": 410, "end": 900},
                {"text": "cluster", "confidence": 0.97, "start": 910, "end": 1200},
            ],
            "confidence": 0.94,
            "audio_duration_ms": 2000,
            "session_id": "sync-sess-1",
            "request_time_ms": 134.2,
        })


    def log_message(self, *a):
        pass

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.headers.get("transfer-encoding", "").lower() == "chunked":
            raw = self._read_chunked()
        else:
            raw = self.rfile.read(int(self.headers.get("content-length", 0) or 0))
        if self.path == "/transcribe":
            return self._sync_like(raw=raw)
        if self.path == "/v1/transcribe/live":
            return self._sync_like(live=True, raw=raw)
        if self.path == "/v2/upload":
            STATE["uploaded"] = raw
            self._json(200, {"upload_url": "https://cdn.test/audio.wav"})
        elif self.path == "/v2/transcript":
            STATE["submitted"] = json.loads(raw)
            if STATE["mode"] == "submit_error":
                self._json(400, {"error": "invalid keyterms"})
            else:
                self._json(200, {"id": "t_123", "status": "queued"})
        else:
            self._json(404, {"error": "nope"})

    def do_GET(self):
        if self.path == "/warm":
            return self._json(200, {"warm": "toasty"})
        if self.path == "/v2/account":
            if STATE["mode"] == "account_denied":
                return self._json(401, {"detail": "invalid key"})
            return self._json(200, {"name": "Test Account", "credits_amount": 42})
        if not self.path.startswith("/v2/transcript/"):
            self._json(200, {"transcripts": []})
            return
        STATE["polls"] += 1
        if STATE["mode"] == "job_error":
            self._json(200, {"status": "error", "error": "audio too quiet"})
            return
        if STATE["polls"] < 3:
            self._json(200, {"status": "processing"})
            return
        self._json(200, {
            "status": "completed",
            "text": "Deploy the Kubernetes cluster to production.",
            "confidence": 0.96,
            "language_code": "en",
            "words": [
                {"text": "Deploy", "confidence": 0.99, "start": 0, "end": 300},
                {"text": "the", "confidence": 0.98, "start": 310, "end": 400},
                {"text": "Kubernetes", "confidence": 0.42, "start": 410, "end": 900},
                {"text": "cluster", "confidence": 0.97, "start": 910, "end": 1200},
            ],
        })


def _parse_multipart(content_type: str, body: bytes):
    """(name -> payload, [names in order], name -> raw part headers)."""
    marker = "boundary="
    i = content_type.find(marker)
    if i < 0:
        return {}, [], {}
    boundary = content_type[i + len(marker):].strip().strip('"')
    parts, order, headers = {}, [], {}
    for chunk in body.split(("--" + boundary).encode()):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        head, sep, payload = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        text = head.decode("latin-1")
        name = ""
        for line in text.splitlines():
            if "name=" in line:
                name = line.split('name="', 1)[-1].split('"', 1)[0]
        parts[name] = payload.rstrip(b"\r\n")
        order.append(name)
        headers[name] = text
    return parts, order, headers


def start_server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def make_client(base, sync=False, **kw):
    """Sections 1-5 exercise the *async* upload->submit->poll flow, so they
    pin sync off; the sync sections pass sync=True."""
    import stt.assemblyai_client as mod
    mod.BASE_URL = base
    mod.SYNC_BASE_URL = base
    return AssemblyAIClient(api_key="test_key", poll_interval=0.01,
                            sync_enabled=sync, **kw)


def speech_like(seconds=2.0, sr=16000):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 140 * t) + 0.1 * np.sin(2 * np.pi * 900 * t)
    return (sig * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32), sr


async def main():
    srv, base = start_server()
    audio, sr = speech_like()
    failures = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
        if not cond:
            failures.append(name)

    print("\n=== 1. Happy path ===")
    STATE.update(polls=0, mode="ok")
    client = make_client(base)
    r = await client.transcribe(audio, sr, keyterms=["Kubernetes", "WhisprFlow"])

    check("ok", r.ok, r.error or "")
    check("text returned", r.text.startswith("Deploy the Kubernetes"), r.text)
    check("confidence parsed", abs(r.confidence - 0.96) < 1e-6, str(r.confidence))
    check("words parsed", len(r.words) == 4, str(len(r.words)))
    check("low-confidence detected",
          [w.text for w in r.low_confidence_words()] == ["Kubernetes"],
          str([w.text for w in r.low_confidence_words()]))
    check("latency recorded", r.latency_ms > 0)
    check("duration recorded", abs(r.audio_duration_s - 2.0) < 0.01)
    check("polled until ready", STATE["polls"] >= 3, str(STATE["polls"]))

    print("\n=== 2. Request payload ===")
    sub = STATE["submitted"]
    check("model sent", sub.get("speech_models") == ["universal-3-5-pro"], str(sub.get("speech_models")))
    check("keyterms sent", sub.get("keyterms_prompt") == ["Kubernetes", "WhisprFlow"], str(sub.get("keyterms_prompt")))
    check("format_text on", sub.get("format_text") is True)
    check("disfluencies off", sub.get("disfluencies") is False)
    check("language en", sub.get("language_code") == "en")

    print("\n=== 3. Uploaded audio integrity ===")
    with wave.open(io.BytesIO(STATE["uploaded"]), "rb") as wf:
        ch, sw, fr, nf = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
    check("mono", ch == 1, str(ch))
    check("16-bit", sw == 2, str(sw))
    check("16 kHz", fr == 16000, str(fr))
    check("full length", nf == len(audio), f"{nf} vs {len(audio)}")

    print("\n=== 4. keyterms suppressed on universal-2 ===")
    STATE.update(polls=0, submitted={})
    c2 = make_client(base, model="universal-2")
    await c2.transcribe(audio, sr, keyterms=["Kubernetes"])
    check("no keyterms for universal-2", "keyterms_prompt" not in STATE["submitted"])
    await c2.close()

    print("\n=== 5. Error handling ===")
    STATE.update(polls=0, mode="job_error")
    r = await client.transcribe(audio, sr)
    check("job error -> ok=False", not r.ok)
    check("job error message", "too quiet" in (r.error or ""), r.error or "")

    STATE.update(polls=0, mode="submit_error")
    r = await client.transcribe(audio, sr)
    check("HTTP 400 -> ok=False", not r.ok)
    check("HTTP 400 message", "400" in (r.error or ""), r.error or "")

    STATE["mode"] = "ok"
    bad = make_client(base)
    bad.set_api_key("")
    r = await bad.transcribe(audio, sr)
    check("no key -> ok=False", not r.ok and "key" in (r.error or "").lower())
    await bad.close()

    await client.close()

    # ── 6. Sync endpoint (audit C12) ────────────────────────────────
    print("\n=== 6. Sync endpoint ===")
    STATE.update(mode="ok", sync=[], polls=0)
    client = make_client(base, sync=True)
    r = await client.transcribe(audio, sr, keyterms=["Kubernetes", "WhisprFlow"],
                               prompt="Dictation in Code.exe")
    sent = STATE["sync"][-1]
    cfg = STATE["sync_config"]
    check("sync used (async not polled)", STATE["polls"] == 0, str(STATE["polls"]))
    check("auth header has no Bearer", sent["auth"] == "test_key", sent["auth"])
    check("X-AAI-Model header required", sent["model"] == "universal-3-5-pro", str(sent["model"]))
    check("multipart with config + audio parts",
          set(STATE["sync_parts"]) >= {"config", "audio"}, str(list(STATE["sync_parts"])))
    ph = STATE["sync_headers"]
    check("audio part typed audio/pcm (raw S16LE)", "audio/pcm" in ph.get("audio", ""),
          ph.get("audio", ""))
    check("config part typed application/json",
          "application/json" in ph.get("config", ""), ph.get("config", ""))
    check("one-shot body is config then audio",
          [n for n in STATE["sync_order"] if n] == ["config", "audio"],
          str(STATE["sync_order"]))
    check("config part is a JSON field, not a file",
          "filename=" not in ph.get("config", ""), ph.get("config", ""))
    check("raw PCM sent (no WAV header)",
          STATE["sync_audio_len"] == len(audio) * 2, str(STATE["sync_audio_len"]))
    check("config carries sample_rate for raw PCM", cfg.get("sample_rate") == sr, str(cfg))
    check("config carries channels", cfg.get("channels") == 1, str(cfg))
    check("prompt forwarded", cfg.get("prompt") == "Dictation in Code.exe", str(cfg))
    check("keyterms capped/filtered for sync",
          cfg.get("keyterms_prompt") == ["Kubernetes", "WhisprFlow"], str(cfg))
    check("text parsed", r.text.startswith("Deploy the Kubernetes"), r.text)
    check("session_id captured", r.session_id == "sync-sess-1", str(r.session_id))
    check("server request_time used as latency", r.latency_ms == 134, str(r.latency_ms))
    check("duration from server", abs(r.audio_duration_s - 2.0) < 1e-6, str(r.audio_duration_s))
    check("words + confidences", len(r.words) == 4 and r.confidence > 0.9, str(len(r.words)))
    check("model label says sync", "sync" in r.model, r.model)
    await client.close()

    # ── 7. Sync failure modes ───────────────────────────────────────
    print("\n=== 7. Sync failure modes ===")
    STATE.update(mode="sync_error", sync=[])
    client = make_client(base, sync=True)
    r = await client.transcribe(audio, sr)
    check("400 bad_audio falls back to async", STATE["polls"] > 0, str(STATE["polls"]))
    check("fallback still returns text", r.ok and r.text.startswith("Deploy"), r.error or r.text)

    STATE.update(mode="sync_badkey", sync=[], polls=0)
    client = make_client(base, sync=True)
    r = await client.transcribe(audio, sr)
    check("401 reported as a key problem", "401" in (r.error or ""), r.error or "")
    check("401 does not hang on retries", STATE["sync"] and len(STATE["sync"]) == 1,
          str(len(STATE["sync"])))

    STATE.update(mode="sync_capacity", sync=[], polls=0)
    client = make_client(base, sync=True)
    r = await client.transcribe(audio, sr)
    retried = len(STATE["sync"])
    check("503 retried once (Retry-After honoured)", retried >= 2, str(retried))
    check("503 eventually reaches async", STATE["polls"] > 0, str(STATE["polls"]))

    STATE.update(mode="ok", sync=[], polls=0)
    client = make_client(base, sync=True)
    long_audio, _ = speech_like(seconds=140.0)
    r = await client.transcribe(long_audio, sr)
    check("over-120s goes straight to async", not STATE["sync"], str(len(STATE["sync"])))
    check("long take still transcribed", r.ok, r.error or "")
    await client.close()

    # ── 8. Live upload (chunked body, config first) ────────────────
    print("\n=== 8. Live upload ===")
    STATE.update(mode="ok", sync=[], polls=0)
    client = make_client(base, sync=True, live_upload=True)
    sess = await client.open_live(sr, keyterms=["Kubernetes"], prompt="in Code.exe")
    check("live session opened", sess is not None and sess.active, str(sess and sess.error))
    if sess is not None:
        step = sr // 10
        for i in range(0, len(audio), step):
            client.feed_live(audio[i:i + step])
        live = await client.finish_live()
        check("live body flushed config ahead of audio",
              [n for n in STATE["sync_order"] if n][:2] == ["config", "audio"],
              str(STATE["sync_order"]))
        check("live audio arrived as raw PCM", STATE["sync_audio_len"] == len(audio) * 2,
              f"{STATE['sync_audio_len']} vs {len(audio) * 2}")
        check("live audio had no WAV header",
              not STATE["sync_parts"].get("audio", b"")[:4].startswith(b"RIFF"),
              str(STATE["sync_parts"].get("audio", b"")[:8]))
        check("live result used, no extra request", live is not None and live.ok, str(live))
        check("live result carries session_id", live is not None
              and live.session_id == "sync-sess-1", str(live and live.session_id))
        r = await client.transcribe(audio, sr, live=live)
        check("already-finished live transcript short-circuits", STATE["polls"] == 0
              and len(STATE["sync"]) == 1, f"{STATE['polls']}/{len(STATE['sync'])}")

    STATE.update(mode="sync_error", sync=[])
    client = make_client(base, sync=True, live_upload=True)
    sess = await client.open_live(sr)
    if sess is not None:
        client.feed_live(audio)
        live = await client.finish_live()
        check("failed live upload is not used", live is None, str(live))

    # ── 9. Key validation + warm (audit C13) ───────────────────────
    print("\n=== 9. Key validation ===")
    STATE.update(mode="ok")
    client = make_client(base, sync=True)
    ok, msg = await client.validate_key()
    check("validate_key accepts a good key", ok is True, msg)
    check("account details surfaced", "42" in msg or "Test" in msg, msg)
    STATE.update(mode="account_denied")
    ok, msg = await client.validate_key()
    check("validate_key rejects a 401", ok is False and "401" in msg, msg)
    STATE.update(mode="ok")
    check("warm() hits GET /warm", await client.warm_sync() is True, "")
    await client.close()

    await client.close()
    srv.shutdown()

    print("\n" + "=" * 56)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("ALL INTEGRATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

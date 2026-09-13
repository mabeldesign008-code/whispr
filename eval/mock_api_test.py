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
        n = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(n)
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


def start_server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def make_client(base, **kw):
    import stt.assemblyai_client as mod
    mod.BASE_URL = base
    return AssemblyAIClient(api_key="test_key", poll_interval=0.01, **kw)


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
    srv.shutdown()

    print("\n" + "=" * 56)
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("ALL INTEGRATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

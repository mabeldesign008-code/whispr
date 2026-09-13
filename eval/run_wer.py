"""
WER harness -- measure accuracy on YOUR audio, not vendor benchmarks.

Setup
-----
1. Record 50-100 clips covering your real usage: quiet, noisy, fast speech,
   whispered, technical jargon, names, numbers.
2. Put the WAVs in eval/clips/
3. Write eval/reference.jsonl, one JSON object per line:
       {"file": "clip001.wav", "text": "the exact words you said"}
4. Run:
       python eval/run_wer.py
       python eval/run_wer.py --dictionary
       python eval/run_wer.py --model universal-2

Compare the WER columns. That number is the only honest way to decide
whether a model change actually helped.
"""

import argparse
import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stt import AssemblyAIClient, UserDictionary  # noqa: E402

ROOT = Path(__file__).resolve().parent
CLIPS = ROOT / "clips"
REFERENCE = ROOT / "reference.jsonl"

_PUNCT = re.compile(r"[^\w\s']")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Standard WER normalisation: lowercase, strip punctuation, collapse space.
    Without this you penalise the model for formatting choices, not errors."""
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def levenshtein_words(ref: list, hyp: list) -> tuple:
    """Return (distance, substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, 0, m

    # dp[i][j] = (cost, sub, del, ins)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub_c, sub_s, sub_d, sub_i = prev[j - 1]
            del_c, del_s, del_d, del_i = prev[j]
            ins_c, ins_s, ins_d, ins_i = cur[j - 1]
            best = min(
                (sub_c + 1, sub_s + 1, sub_d, sub_i),
                (del_c + 1, del_s, del_d + 1, del_i),
                (ins_c + 1, ins_s, ins_d, ins_i + 1),
                key=lambda t: t[0],
            )
            cur.append(best)
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str) -> dict:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    dist, subs, dels, ins = levenshtein_words(ref, hyp)
    return {
        "wer": dist / len(ref) if ref else (1.0 if hyp else 0.0),
        "errors": dist, "sub": subs, "del": dels, "ins": ins,
        "ref_words": len(ref),
    }


def load_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        sr, n, ch, sw = wf.getframerate(), wf.getnframes(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw)
    if dtype is None:
        raise ValueError(f"Unsupported sample width {sw} in {path.name}")
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio /= float(np.iinfo(dtype).max)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


async def main():
    ap = argparse.ArgumentParser(description="Measure WER on your own clips")
    ap.add_argument("--model", default="universal-3-5-pro", help="AssemblyAI model")
    ap.add_argument("--dictionary", action="store_true", help="Send user dictionary as keyterms")
    ap.add_argument("--reference", default=str(REFERENCE))
    ap.add_argument("--out", default="", help="Write per-clip JSON results here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"No reference file at {ref_path}\n")
        print("Create it with one JSON object per line:")
        print('  {"file": "clip001.wav", "text": "what you actually said"}')
        return 1

    entries = [json.loads(l) for l in ref_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not entries:
        print("Reference file is empty.")
        return 1

    engine = AssemblyAIClient(model=args.model)
    keyterms = UserDictionary().as_keyterms() if args.dictionary else None

    print(f"\nModel:  {args.model}")
    print(f"Clips:  {len(entries)}   Dictionary: {len(keyterms) if keyterms else 0} terms")
    print("=" * 74)

    rows, total_err, total_words, total_latency, failures = [], 0, 0, 0, 0

    for e in entries:
        path = CLIPS / e["file"]
        if not path.exists():
            print(f"  MISSING  {e['file']}")
            failures += 1
            continue

        audio, sr = load_wav(path)
        t0 = time.perf_counter()
        result = await engine.transcribe(audio, sr, keyterms=keyterms)
        elapsed = int((time.perf_counter() - t0) * 1000)

        if not result.ok:
            print(f"  ERROR    {e['file']}: {result.error}")
            failures += 1
            continue

        m = wer(e["text"], result.text)
        total_err += m["errors"]
        total_words += m["ref_words"]
        total_latency += elapsed

        rows.append({
            "file": e["file"], "reference": e["text"], "hypothesis": result.text,
            "latency_ms": elapsed, "confidence": result.confidence, **m,
        })

        flag = "  <<<" if m["wer"] > 0.15 else ""
        print(f"  {m['wer']*100:5.1f}%  {elapsed:5d}ms  {e['file']}{flag}")
        if args.verbose and m["wer"] > 0:
            print(f"           ref: {e['text']}")
            print(f"           hyp: {result.text}")

    print("=" * 74)
    if total_words:
        n = len(rows)
        overall = total_err / total_words * 100
        median = sorted(r["wer"] for r in rows)[n // 2] * 100
        perfect = sum(1 for r in rows if r["wer"] == 0)
        print(f"  Overall WER : {overall:.2f}%  ({total_err} errors / {total_words} words)")
        print(f"  Median WER  : {median:.2f}%")
        print(f"  Perfect     : {perfect}/{n} clips ({perfect/n*100:.0f}%)")
        print(f"  Avg latency : {total_latency//n} ms")
        print(f"  Substitutions {sum(r['sub'] for r in rows)}  "
              f"Deletions {sum(r['del'] for r in rows)}  "
              f"Insertions {sum(r['ins'] for r in rows)}")
    if failures:
        print(f"  Failed      : {failures} clips")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n  Wrote {args.out}")

    await engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

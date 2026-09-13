"""
Setup helper: configure the AssemblyAI key and verify it works.

    python setup_stt.py            # interactive setup + live API test
    python setup_stt.py --check    # just verify the current config
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stt import AssemblyAIClient, UserDictionary, default_config_dir  # noqa: E402

ENV_PATH = Path(".env")
SIGNUP = "https://www.assemblyai.com/dashboard/signup"


def read_env_key(name: str) -> str:
    if not ENV_PATH.exists():
        return ""
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return ""


def write_env_key(name: str, value: str) -> None:
    lines, found = [], False
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                lines.append(f"{name}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{name}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def speech_like(seconds=1.5, sr=16000):
    """A short synthetic clip -- enough to prove the round trip works."""
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 140 * t) + 0.1 * np.sin(2 * np.pi * 900 * t)
    return (sig * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32), sr


async def verify(key: str, model: str) -> bool:
    client = AssemblyAIClient(api_key=key, model=model)
    audio, sr = speech_like()
    print(f"\n  Testing {model} ... ", end="", flush=True)
    result = await client.transcribe(audio, sr, keyterms=["WhisprFlow"])
    await client.close()

    if result.ok:
        print(f"OK  ({result.latency_ms} ms)")
        print(f"  Transcript: {result.text!r}")
        print("  (Synthetic tone, so empty or nonsense text is expected --")
        print("   what matters is that the API accepted and processed it.)")
        return True

    print("FAILED")
    print(f"  {result.error}")
    if "401" in (result.error or "") or "unauthor" in (result.error or "").lower():
        print("  -> The key was rejected. Check for typos or trailing spaces.")
    elif "insufficient" in (result.error or "").lower():
        print("  -> Out of credit. Free tier gives $50 (~185 hours).")
    return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, don't prompt")
    ap.add_argument("--model", default="universal-3-5-pro")
    args = ap.parse_args()

    print("=" * 66)
    print("  WhisprFlow — AssemblyAI setup")
    print("=" * 66)

    key = os.getenv("ASSEMBLYAI_API_KEY") or read_env_key("ASSEMBLYAI_API_KEY")

    if not key and not args.check:
        print("\n  No AssemblyAI key found.\n")
        print(f"  1. Sign up (no credit card, $50 free credit ~185 hours):")
        print(f"     {SIGNUP}")
        print("  2. Copy your API key from the dashboard home page.")
        print("  3. Paste it below.\n")
        try:
            key = input("  API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Cancelled.")
            return 1
        if not key:
            print("  No key entered.")
            return 1

    if not key:
        print("\n  No key configured. Run without --check to set one up.")
        return 1

    ok = await verify(key, args.model)
    if not ok:
        return 1

    if not args.check:
        write_env_key("ASSEMBLYAI_API_KEY", key)
        print(f"\n  Saved to {ENV_PATH.resolve()}")

    d = UserDictionary()
    print(f"\n  Config dir : {default_config_dir()}")
    print(f"  Dictionary : {d.path.name} ({len(d)} terms)")
    if len(d) == 0:
        print("               Add your names/jargon there — biggest accuracy win.")

    print("\n  Ready. Run:  python main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stt import DictationClient  # noqa: E402

ENV_PATH = Path(".env")
SIGNUP = "https://www.assemblyai.com/dashboard/signup"


def read_env_key(name: str) -> str:
    if not ENV_PATH.exists():
        return ""
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def write_env_key(name: str, value: str) -> None:
    lines = []
    if ENV_PATH.exists():
        lines = [l for l in ENV_PATH.read_text(encoding="utf-8").splitlines()
                 if not l.strip().startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(ENV_PATH, 0o600)


async def check(key: str) -> bool:
    client = DictationClient(api_key=key)
    try:
        ok, msg = await client.validate_key()
        if ok:
            print(f"Key verified: {msg}")
            print("Dictation API ready (transcription + cleanup in one call).")
            return True
        if ok is None:
            print(f"Could not verify offline: {msg}")
            return False
        print(f"Key rejected: {msg}")
        return False
    finally:
        await client.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify current config only")
    args = ap.parse_args()

    key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip() or read_env_key("ASSEMBLYAI_API_KEY")

    if args.check:
        if not key:
            print("No ASSEMBLYAI_API_KEY found in environment or .env")
            return 1
        return 0 if asyncio.run(check(key)) else 1

    if key:
        print("Found an existing key. Verifying…")
        if asyncio.run(check(key)):
            print("Setup already complete.")
            return 0
        print("The saved key did not pass verification.")

    print(f"\nGet a free API key at {SIGNUP}")
    print("(New accounts include $50 of free credits; the Dictation API")
    print(" then costs $0.62 per hour of audio. No card needed to start.)\n")
    key = input("Paste your AssemblyAI API key: ").strip()
    if not key:
        print("No key entered.")
        return 1

    print("Verifying…")
    if not asyncio.run(check(key)):
        print("Verification failed — the key was not saved.")
        return 1

    write_env_key("ASSEMBLYAI_API_KEY", key)
    print(f"Saved to {ENV_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

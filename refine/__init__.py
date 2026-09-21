"""Text cleanup for WhisprFlow.

Cleanup is performed by the AssemblyAI Dictation API inside the same
request that transcribes (see stt/dictation.py). This package contains
what stays local: the per-take instruction builder (refiner.py), the
deterministic fallback cleanup, and Command Mode (commands.py), the one
feature that still calls an LLM directly.
"""

from .refiner import (
    BASE_INSTRUCTION,
    LEGACY_TONE_ALIASES,
    TONE_NAMES,
    TONES,
    basic_cleanup,
    build_instruction,
    default_tone,
)

__all__ = [
    "BASE_INSTRUCTION",
    "LEGACY_TONE_ALIASES",
    "TONE_NAMES",
    "TONES",
    "basic_cleanup",
    "build_instruction",
    "default_tone",
]

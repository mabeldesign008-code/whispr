"""Speech-to-text for WhisprFlow: the AssemblyAI Dictation API.

A single POST per take returns both the verbatim transcript and the
cleaned text (filler removal, self-correction resolution, punctuation),
so no separate streaming/batch/Groq stages exist anymore.
"""

from .base import TranscriptionResult, WordInfo, float_to_wav_bytes
from .dictation import DictationClient, DictationResult
from .dictionary import UserDictionary, default_config_dir

__all__ = [
    "TranscriptionResult",
    "WordInfo",
    "float_to_wav_bytes",
    "DictationClient",
    "DictationResult",
    "UserDictionary",
    "default_config_dir",
]

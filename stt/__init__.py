"""Speech-to-text for WhisprFlow. AssemblyAI only -- no fallback engines."""

from .base import TranscriptionResult, WordInfo, float_to_wav_bytes
from .assemblyai_client import AssemblyAIClient, clean_keyterms
from .dictionary import UserDictionary, default_config_dir
from .streaming import StreamingSession, websockets_available

__all__ = [
    "TranscriptionResult",
    "WordInfo",
    "float_to_wav_bytes",
    "AssemblyAIClient",
    "clean_keyterms",
    "UserDictionary",
    "default_config_dir",
    "StreamingSession",
    "websockets_available",
]

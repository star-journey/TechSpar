"""STT (speech-to-text) provider abstraction.

Each provider implements `STTProvider.transcribe(audio_bytes, suffix)` and declares
`native_formats` — formats it accepts directly without re-encoding. The base class's
`_prepare()` hook transparently routes unsupported formats through ffmpeg → wav 16k mono.

Use `backend.stt.get_provider(name)` to instantiate the configured provider.
"""

from backend.stt.base import STTProvider
from backend.stt.factory import get_provider, list_providers

__all__ = ["STTProvider", "get_provider", "list_providers"]

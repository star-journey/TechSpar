"""ElevenLabs Speech-to-Text provider (sync multipart).

API: POST https://api.elevenlabs.io/v1/speech-to-text
Body: multipart `file` + `model_id`（默认 scribe_v2）；同步返回 `text`。
本地直传，原生支持 m4a。
"""
from __future__ import annotations

import logging

import requests

from backend.config import settings
from backend.stt.base import STTProvider

logger = logging.getLogger("uvicorn")

_ENDPOINT = "https://api.elevenlabs.io/v1/speech-to-text"
_NATIVE = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".webm", ".flac", ".mp4", ".mpeg", ".mpga"}


class ElevenLabsProvider(STTProvider):
    name = "elevenlabs"
    needs_public_url = False
    native_formats = _NATIVE

    def _api_key(self) -> str:
        key = settings.elevenlabs_api_key
        if not key:
            raise RuntimeError("ELEVENLABS_API_KEY not configured")
        return key

    def _do_transcribe(self, audio_bytes: bytes, suffix: str) -> str:
        model_id = settings.elevenlabs_model or "scribe_v2"
        files = {"file": (f"audio{suffix}", audio_bytes)}
        data = {"model_id": model_id}
        headers = {"xi-api-key": self._api_key()}

        resp = requests.post(_ENDPOINT, headers=headers, files=files, data=data, timeout=600)
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs STT failed [{resp.status_code}]: {resp.text}")

        body = resp.json()
        text = body.get("text") or ""
        if not text:
            # 多通道场景：transcripts[].text 拼接兜底
            transcripts = body.get("transcripts") or []
            text = "\n".join(t.get("text", "") for t in transcripts if t.get("text"))
        logger.info("ElevenLabs transcription done: %d chars", len(text))
        return text

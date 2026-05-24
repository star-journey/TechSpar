"""Azure Fast Transcription provider (sync multipart, no public URL needed).

API: POST https://{region-or-resource}.cognitiveservices.azure.com/speechtotext/transcriptions:transcribe
Body: multipart/form-data 包含 `audio` 文件 + `definition` JSON。
本地 m4a → ffmpeg 转 wav 后再上传（m4a 不在 Azure 官方支持列表内）。
"""
from __future__ import annotations

import json
import logging

import requests

from backend.config import settings
from backend.stt.base import STTProvider

logger = logging.getLogger("uvicorn")

_API_VERSION = "2025-10-15"

# Azure 官方列出的可直传格式：m4a / mp4 不在列，会被 base._prepare 转 wav。
_NATIVE = {".wav", ".mp3", ".ogg", ".opus", ".flac", ".aac", ".amr", ".webm", ".wma"}

_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".amr": "audio/amr",
    ".webm": "audio/webm",
    ".wma": "audio/x-ms-wma",
}


class AzureFastTranscriptionProvider(STTProvider):
    name = "azure"
    needs_public_url = False
    native_formats = _NATIVE

    def _endpoint(self) -> str:
        key = settings.azure_speech_key
        if not key:
            raise RuntimeError("AZURE_SPEECH_KEY not configured")
        region = (settings.azure_speech_region or "").strip().lower()
        if not region:
            raise RuntimeError("AZURE_SPEECH_REGION not configured")
        # 允许填全资源域名（自带 cognitiveservices.azure.com）或仅区域代号。
        host = region if "." in region else f"{region}.cognitiveservices.azure.com"
        return f"https://{host}/speechtotext/transcriptions:transcribe?api-version={_API_VERSION}"

    def _locales(self) -> list[str]:
        raw = settings.azure_speech_locales or ""
        items = [s.strip() for s in raw.split(",") if s.strip()]
        return items or ["zh-CN", "en-US"]

    def _do_transcribe(self, audio_bytes: bytes, suffix: str) -> str:
        url = self._endpoint()
        mime = _MIME_BY_SUFFIX.get(suffix, "application/octet-stream")
        definition = {
            "locales": self._locales(),
            "profanityFilterMode": "None",
        }
        files = {
            "audio": (f"audio{suffix}", audio_bytes, mime),
            "definition": (None, json.dumps(definition), "application/json"),
        }
        headers = {"Ocp-Apim-Subscription-Key": settings.azure_speech_key}

        resp = requests.post(url, headers=headers, files=files, timeout=600)
        if resp.status_code != 200:
            raise RuntimeError(f"Azure fast transcription failed [{resp.status_code}]: {resp.text}")

        data = resp.json()
        combined = data.get("combinedPhrases") or []
        if combined:
            text = "\n".join(p.get("text", "") for p in combined if p.get("text"))
            if text.strip():
                logger.info("Azure transcription done: %d chars", len(text))
                return text
        # fallback：拼 phrases[]
        phrases = data.get("phrases") or []
        text = " ".join(p.get("text", "") for p in phrases if p.get("text"))
        logger.info("Azure transcription (phrases fallback) done: %d chars", len(text))
        return text

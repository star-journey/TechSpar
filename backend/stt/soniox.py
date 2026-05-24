"""Soniox async transcription provider.

流程：
1) POST /v1/files (multipart 上传) → 拿 file_id；
2) POST /v1/transcriptions { model, file_id } → 拿 transcription_id；
3) 轮询 GET /v1/transcriptions/{id} 直到 status == "completed"；
4) GET /v1/transcriptions/{id}/transcript → tokens[].text 拼接。
本地直传，原生支持 m4a。
"""
from __future__ import annotations

import logging
import time

import requests

from backend.config import settings
from backend.stt.base import STTProvider

logger = logging.getLogger("uvicorn")

_BASE = "https://api.soniox.com/v1"
_NATIVE = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".webm", ".flac", ".mp4", ".amr", ".aiff", ".asf"}
_POLL_INTERVAL = 2.0
_POLL_MAX_SECONDS = 900  # 15min


class SonioxProvider(STTProvider):
    name = "soniox"
    needs_public_url = False
    native_formats = _NATIVE

    def _api_key(self) -> str:
        key = settings.soniox_api_key
        if not key:
            raise RuntimeError("SONIOX_API_KEY not configured")
        return key

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key()}"}

    def _do_transcribe(self, audio_bytes: bytes, suffix: str) -> str:
        session = requests.Session()
        session.headers.update(self._headers())

        # 1) 上传文件
        files = {"file": (f"audio{suffix}", audio_bytes)}
        resp = session.post(f"{_BASE}/files", files=files, timeout=300)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Soniox upload failed [{resp.status_code}]: {resp.text}")
        file_id = resp.json().get("id")
        if not file_id:
            raise RuntimeError(f"Soniox upload missing id: {resp.text}")
        logger.info("Soniox file uploaded: %s", file_id)

        # 2) 创建转写任务
        model = settings.soniox_model or "stt-async-v4"
        resp = session.post(
            f"{_BASE}/transcriptions",
            json={"model": model, "file_id": file_id},
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Soniox transcription create failed [{resp.status_code}]: {resp.text}")
        transcription_id = resp.json().get("id")
        if not transcription_id:
            raise RuntimeError(f"Soniox transcription missing id: {resp.text}")
        logger.info("Soniox transcription created: %s", transcription_id)

        # 3) 轮询
        deadline = time.time() + _POLL_MAX_SECONDS
        while time.time() < deadline:
            time.sleep(_POLL_INTERVAL)
            poll = session.get(f"{_BASE}/transcriptions/{transcription_id}", timeout=30)
            if poll.status_code != 200:
                raise RuntimeError(f"Soniox poll failed [{poll.status_code}]: {poll.text}")
            status = (poll.json().get("status") or "").lower()
            if status == "completed":
                break
            if status in ("error", "failed"):
                raise RuntimeError(f"Soniox transcription {status}: {poll.text}")
        else:
            raise RuntimeError("Soniox transcription timed out")

        # 4) 取 transcript
        resp = session.get(f"{_BASE}/transcriptions/{transcription_id}/transcript", timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Soniox fetch transcript failed [{resp.status_code}]: {resp.text}")
        tokens = resp.json().get("tokens") or []
        text = "".join(t.get("text", "") for t in tokens)
        logger.info("Soniox transcription done: %d chars", len(text))
        return text

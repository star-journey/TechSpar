"""DashScope (qwen3-asr-flash) provider.

两条链路保持兼容老实现：
- 短音频 (<=7MB)：base64 data URI → qwen3-asr-flash 同步 chat/completions；
- 长音频：落盘/OSS 拿公网 URL → qwen3-asr-flash-filetrans 异步轮询。
"""
from __future__ import annotations

import base64
import logging
import time

import requests

from backend.config import settings
from backend.stt.base import STTProvider

logger = logging.getLogger("uvicorn")

_SYNC_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
_SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
_QUERY_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/"
# 同步端点限制：输入 ≤10MB / 时长 ≤5min。base64 膨胀 4/3，留 7MB 安全线。
_SYNC_MAX_RAW_BYTES = 7 * 1024 * 1024

_MIME_BY_SUFFIX = {
    ".webm": "audio/webm",
    ".mp3": "audio/mp3",
    ".wav": "audio/wav",
    ".m4a": "audio/m4a",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


class DashScopeProvider(STTProvider):
    name = "dashscope"
    # 长音频需要公网 URL（filetrans 限制）；短音频走 base64 不需要，但保留长路径。
    needs_public_url = True
    native_formats = set(_MIME_BY_SUFFIX.keys())

    _submit_url = _SUBMIT_URL
    _query_url = _QUERY_URL

    def _api_key(self) -> str:
        key = settings.effective_dashscope_api_key
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY not configured")
        return key

    def _do_transcribe(self, audio_bytes: bytes, suffix: str) -> str:
        if len(audio_bytes) <= _SYNC_MAX_RAW_BYTES:
            return self._transcribe_sync(audio_bytes, suffix)
        return self._transcribe_async(audio_bytes, suffix)

    def _transcribe_sync(self, audio_bytes: bytes, suffix: str) -> str:
        mime = _MIME_BY_SUFFIX.get(suffix, "audio/webm")
        data_uri = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode()}"
        payload = {
            "model": "qwen3-asr-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": data_uri}},
                    ],
                }
            ],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        resp = requests.post(_SYNC_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope sync ASR failed [{resp.status_code}]: {resp.text}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"DashScope response missing transcript: {exc}; body={data}")
        logger.info("DashScope sync transcription done: %d chars", len(text))
        return text

    def _transcribe_async(self, audio_bytes: bytes, suffix: str) -> str:
        file_url = self._publish(audio_bytes, suffix)
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        payload = {
            "model": "qwen3-asr-flash-filetrans",
            "input": {"file_url": file_url},
            "parameters": {"channel_id": [0]},
        }
        resp = requests.post(self._submit_url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Transcription submit failed: {resp.text}")
        task_id = resp.json()["output"]["task_id"]
        logger.info("DashScope async task: %s", task_id)

        query_headers = {"Authorization": f"Bearer {self._api_key()}"}
        for _ in range(300):
            time.sleep(3)
            qr = requests.get(self._query_url + task_id, headers=query_headers)
            output = qr.json().get("output", {})
            status = output.get("task_status", "").upper()
            if status == "SUCCEEDED":
                text = _extract_text(output)
                logger.info("DashScope async transcription done: %d chars", len(text))
                return text
            if status in ("FAILED", "UNKNOWN"):
                raise RuntimeError(f"Transcription {status}: {output.get('message', '')}")
        raise RuntimeError("Transcription timed out")

    def _publish(self, audio_bytes: bytes, suffix: str) -> str:
        """获取一个公网可读的 URL：优先本机签名链路，否则回退到 OSS。"""
        if settings.public_base_url:
            from backend.public_url import build_signed_url, save_audio_blob

            key = save_audio_blob(audio_bytes, suffix)
            return build_signed_url(key)
        return _upload_to_oss(audio_bytes, suffix)


def _upload_to_oss(audio_bytes: bytes, suffix: str) -> str:
    """上传到阿里云 OSS，返回 1h 签名 URL。"""
    import uuid

    import oss2

    missing = [
        name for name, val in (
            ("ALIYUN_OSS_ACCESS_KEY_ID", settings.aliyun_oss_access_key_id),
            ("ALIYUN_OSS_ACCESS_KEY_SECRET", settings.aliyun_oss_access_key_secret),
            ("ALIYUN_OSS_BUCKET", settings.aliyun_oss_bucket),
            ("ALIYUN_OSS_ENDPOINT", settings.aliyun_oss_endpoint),
        ) if not val
    ]
    if missing:
        raise RuntimeError(f"Alibaba OSS not configured: missing {', '.join(missing)}")
    auth = oss2.Auth(settings.aliyun_oss_access_key_id, settings.aliyun_oss_access_key_secret)
    bucket = oss2.Bucket(auth, settings.aliyun_oss_endpoint, settings.aliyun_oss_bucket)
    key = f"audio/{uuid.uuid4().hex}{suffix}"
    bucket.put_object(key, audio_bytes)
    url = bucket.sign_url("GET", key, 3600, slash_safe=True)
    logger.info("Uploaded to OSS: %s", key)
    return url


def _extract_text(output: dict) -> str:
    result = output.get("result", {})
    url = result.get("transcription_url")
    if not url:
        for item in output.get("results", []):
            url = item.get("transcription_url")
            if url:
                break
    if not url:
        return ""
    resp = requests.get(url)
    if resp.status_code != 200:
        return ""
    data = resp.json()
    texts = [t.get("text", "") for t in data.get("transcripts", []) if t.get("text")]
    return "\n".join(texts)

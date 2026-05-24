"""QwenCloud (DashScope International) provider.

接口与 DashScope 完全同形，仅 host 换成 dashscope-intl.aliyuncs.com，
鉴权用独立 qwencloud_api_key（未配置时回落到 dashscope_api_key）。
仍只接受公网 URL → 复用 backend/public_url.py + OSS 链路。
"""
from __future__ import annotations

from backend.config import settings
from backend.stt.dashscope import DashScopeProvider

_SUBMIT_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/asr/transcription"
_QUERY_URL = "https://dashscope-intl.aliyuncs.com/api/v1/tasks/"


class QwenCloudProvider(DashScopeProvider):
    name = "qwencloud"
    _submit_url = _SUBMIT_URL
    _query_url = _QUERY_URL

    def _api_key(self) -> str:
        key = settings.qwencloud_api_key or settings.effective_dashscope_api_key
        if not key:
            raise RuntimeError("QWENCLOUD_API_KEY (or DASHSCOPE_API_KEY fallback) not configured")
        return key

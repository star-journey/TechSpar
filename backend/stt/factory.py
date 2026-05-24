"""STT provider 工厂：按名称返回单例适配器。

provider 名称小写、稳定，前后端共享同一套常量。
"""
from __future__ import annotations

from functools import lru_cache

from backend.stt.azure import AzureFastTranscriptionProvider
from backend.stt.base import STTProvider
from backend.stt.dashscope import DashScopeProvider
from backend.stt.elevenlabs import ElevenLabsProvider
from backend.stt.qwencloud import QwenCloudProvider
from backend.stt.soniox import SonioxProvider

_REGISTRY: dict[str, type[STTProvider]] = {
    "dashscope": DashScopeProvider,
    "azure": AzureFastTranscriptionProvider,
    "soniox": SonioxProvider,
    "elevenlabs": ElevenLabsProvider,
    "qwencloud": QwenCloudProvider,
}


def list_providers() -> list[str]:
    return list(_REGISTRY.keys())


@lru_cache(maxsize=None)
def _instance(name: str) -> STTProvider:
    return _REGISTRY[name]()


def get_provider(name: str | None) -> STTProvider:
    """Return STTProvider singleton by name. Defaults to DashScope."""
    key = (name or "").strip().lower() or "dashscope"
    if key not in _REGISTRY:
        raise RuntimeError(f"Unknown STT provider: {name!r}. Available: {list_providers()}")
    return _instance(key)

"""STTProvider 抽象基类。

子类只需声明 `name` / `native_formats` 并实现 `_do_transcribe`；基类负责：
- 调用 `_prepare()` 在格式不被原生支持时自动转码为 wav 16k mono；
- 对外暴露统一的 `transcribe(audio_bytes, suffix) -> str` 入口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.stt.codec import to_wav_16k_mono


class STTProvider(ABC):
    """所有 STT 厂商适配器的统一接口。"""

    name: str = ""
    # 该厂商无需公网 URL（可本地直传或走 base64）即可工作。
    # DashScope 长音频、QwenCloud 仍需 public URL → True。
    needs_public_url: bool = False
    # 厂商原生支持的扩展名集合（小写，含点）。不在集合内的格式会经 ffmpeg 转 wav。
    native_formats: set[str] = set()

    @abstractmethod
    def _do_transcribe(self, audio_bytes: bytes, suffix: str) -> str:
        """子类实现：在此处实际调用厂商 API。`suffix` 已是 native_formats 内的格式。"""

    def transcribe(self, audio_bytes: bytes, suffix: str) -> str:
        if not audio_bytes:
            raise RuntimeError("empty audio payload")
        prepared_bytes, prepared_suffix = self._prepare(audio_bytes, suffix)
        return self._do_transcribe(prepared_bytes, prepared_suffix)

    def _prepare(self, audio_bytes: bytes, suffix: str) -> tuple[bytes, str]:
        """格式预处理：命中 native_formats 原样返回；否则 ffmpeg → wav 16k mono。"""
        normalized = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        if normalized in self.native_formats:
            return audio_bytes, normalized
        return to_wav_16k_mono(audio_bytes, normalized), ".wav"

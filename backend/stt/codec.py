"""音频格式转码：仅在厂商不原生支持源格式时调用。

设计要求：
- 走 ffmpeg 子进程，stdin/stdout 流式，不落临时文件；
- 统一输出 16kHz / 单声道 / 16-bit PCM wav（ASR 友好且体积可控）；
- ffmpeg 未安装 → RuntimeError，由路由层 4xx/5xx 抛回前端。
"""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger("uvicorn")

_FFMPEG_BIN = "ffmpeg"


def _ensure_ffmpeg() -> None:
    if shutil.which(_FFMPEG_BIN) is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; required for transcoding non-native audio formats. "
            "Install ffmpeg on the server (e.g., `apt-get install -y ffmpeg`)."
        )


def to_wav_16k_mono(audio_bytes: bytes, src_suffix: str) -> bytes:
    """把任意输入音频转成 16kHz mono 16-bit PCM wav。

    src_suffix 仅用于日志/错误提示，ffmpeg 自动嗅探格式。
    """
    _ensure_ffmpeg()

    cmd = [
        _FFMPEG_BIN,
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=audio_bytes,
            capture_output=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg transcode timeout (src={src_suffix})") from exc

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg transcode failed (src={src_suffix}): {err}")

    out = proc.stdout
    logger.info(
        "ffmpeg transcoded %s (%d B) -> wav 16k mono (%d B)",
        src_suffix, len(audio_bytes), len(out),
    )
    return out

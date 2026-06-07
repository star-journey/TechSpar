"""音频格式转码：仅在厂商不原生支持源格式时调用。

设计要求：
- 统一输出 16kHz / 单声道 / 16-bit PCM wav（ASR 友好且体积可控）；
- ffmpeg 未安装 → RuntimeError，由路由层 4xx/5xx 抛回前端。
"""
from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger("uvicorn")

_FFMPEG_BIN = "ffmpeg"
_SEEKABLE_INPUT_SUFFIXES = {".m4a", ".mp4", ".mov"}
_MIN_AUDIO_FRAMES = 1600


def _ensure_ffmpeg() -> None:
    if shutil.which(_FFMPEG_BIN) is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; required for transcoding non-native audio formats. "
            "Install ffmpeg on the server (e.g., `apt-get install -y ffmpeg`)."
        )


def _validate_wav_audio(wav_bytes: bytes, src_suffix: str) -> None:
    try:
        with wave.open(io.BytesIO(wav_bytes)) as wav_file:
            frames = wav_file.getnframes()
    except Exception as exc:
        raise RuntimeError(f"ffmpeg transcode produced invalid wav (src={src_suffix})") from exc

    if frames < _MIN_AUDIO_FRAMES:
        raise RuntimeError(
            f"ffmpeg transcode produced too little audio (src={src_suffix}, frames={frames})"
        )


def _run_ffmpeg(cmd: list[str], audio_bytes: bytes | None, src_suffix: str) -> bytes:
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

    return proc.stdout


def _transcode_from_seekable_file(audio_bytes: bytes, src_suffix: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=src_suffix, delete=False) as input_file:
        input_file.write(audio_bytes)
        input_path = input_file.name

    try:
        cmd = [
            _FFMPEG_BIN,
            "-hide_banner",
            "-loglevel", "error",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            "pipe:1",
        ]
        return _run_ffmpeg(cmd, None, src_suffix)
    finally:
        Path(input_path).unlink(missing_ok=True)


def _transcode_from_stdin(audio_bytes: bytes, src_suffix: str) -> bytes:
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
    return _run_ffmpeg(cmd, audio_bytes, src_suffix)


def to_wav_16k_mono(audio_bytes: bytes, src_suffix: str) -> bytes:
    """把任意输入音频转成 16kHz mono 16-bit PCM wav。

    src_suffix 仅用于日志/错误提示，ffmpeg 自动嗅探格式。
    """
    _ensure_ffmpeg()

    normalized = src_suffix.lower()
    if normalized in _SEEKABLE_INPUT_SUFFIXES:
        out = _transcode_from_seekable_file(audio_bytes, normalized)
    else:
        out = _transcode_from_stdin(audio_bytes, normalized)
    _validate_wav_audio(out, normalized)

    logger.info(
        "ffmpeg transcoded %s (%d B) -> wav 16k mono (%d B)",
        normalized, len(audio_bytes), len(out),
    )
    return out

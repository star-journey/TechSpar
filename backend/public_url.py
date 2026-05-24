"""自有域名 + HMAC 签名链路：替代 OSS 为 DashScope filetrans 提供临时 URL。

仅当 settings.public_base_url 配置后才会启用；落盘文件由后台任务按 TTL 清理。
"""
from __future__ import annotations

import hmac
import logging
import time
import uuid
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote

from backend.config import settings

logger = logging.getLogger("uvicorn")


def _sign(key: str, expires: int) -> str:
    secret = settings.effective_public_url_secret().encode()
    msg = f"{key}|{expires}".encode()
    return hmac.new(secret, msg, sha256).hexdigest()


def save_audio_blob(audio_bytes: bytes, suffix: str) -> str:
    """落盘到 public_audio_dir，返回 key（仅文件名，不含目录）。"""
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    key = f"{uuid.uuid4().hex}{safe_suffix}"
    path = settings.public_audio_dir() / key
    path.write_bytes(audio_bytes)
    logger.info(f"Saved public audio: {key} ({len(audio_bytes)} bytes)")
    return key


def build_signed_url(key: str) -> str:
    if not settings.public_base_url:
        raise RuntimeError("PUBLIC_BASE_URL not configured")
    expires = int(time.time()) + settings.public_audio_ttl_seconds
    sig = _sign(key, expires)
    base = settings.public_base_url.rstrip("/")
    return f"{base}/public/audio/{quote(key)}?expires={expires}&sig={sig}"


def verify_signature(key: str, expires: int, sig: str) -> bool:
    if expires < int(time.time()):
        return False
    expected = _sign(key, expires)
    return hmac.compare_digest(expected, sig)


def cleanup_expired(retain_seconds: int) -> int:
    """删除 mtime 超过 retain_seconds 的文件，返回删除数量。"""
    directory = settings.public_audio_dir()
    cutoff = time.time() - retain_seconds
    removed = 0
    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as exc:
            logger.warning(f"Failed to remove {path}: {exc}")
    if removed:
        logger.info(f"Cleaned up {removed} expired public audio files")
    return removed


def resolve_safe_path(key: str) -> Path | None:
    """把 key 解析为安全的绝对路径；返回 None 表示非法或不存在。"""
    safe_name = Path(key).name  # 防 ../ 穿越
    if not safe_name or safe_name != key:
        return None
    path = settings.public_audio_dir() / safe_name
    if not path.is_file():
        return None
    return path

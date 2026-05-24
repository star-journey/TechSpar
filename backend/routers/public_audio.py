"""匿名签名 URL 端点：DashScope filetrans 通过此路径拉取音频。

不挂 auth 依赖；只校验 HMAC 签名 + 过期时间。
"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.public_url import resolve_safe_path, verify_signature
from backend.transcribe import _AUDIO_MIME

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/public")


@router.get("/audio/{key}")
async def get_public_audio(key: str, expires: int, sig: str):
    if not verify_signature(key, expires, sig):
        raise HTTPException(403, "invalid or expired signature")

    path = resolve_safe_path(key)
    if path is None:
        raise HTTPException(404, "not found")

    suffix = Path(key).suffix.lower()
    media_type = _AUDIO_MIME.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media_type)

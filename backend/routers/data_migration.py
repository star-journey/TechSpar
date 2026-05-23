"""数据迁移端点：导出/导入当前登录用户的数据。

与 CLI (scripts/export_data.py、scripts/import_data.py) 共用
backend.storage.data_migration 中的核心实现。

HTTP 入口始终把数据归到当前登录用户（rebind_user_id=user_id），防止
跨用户写入或泄露。
"""
from __future__ import annotations

import logging
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.auth import get_current_user
from backend.storage.data_migration import (
    SCHEMA_VERSION,
    export_archive,
    import_archive,
)

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/api/data")

# 单次上传的硬上限——防御性，避免临时盘被占满
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def _cleanup(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("迁移临时文件清理失败 %s: %s", path, e)


@router.get("/export")
def export_data(
    background: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    """以 tar.gz 形式返回当前用户的全部数据。"""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tmp_dir = Path(tempfile.mkdtemp(prefix="techspar-export-"))
    archive_path = tmp_dir / f"techspar-backup-{ts}.tar.gz"

    try:
        export_archive(archive_path, user_id=user_id)
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))

    background.add_task(_cleanup, archive_path)
    background.add_task(lambda: tmp_dir.rmdir() if tmp_dir.exists() else None)

    return FileResponse(
        archive_path,
        media_type="application/gzip",
        filename=archive_path.name,
    )


@router.post("/import")
async def import_data(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    db_strategy: str = Form("skip"),
    overwrite_files: bool = Form(False),
    user_id: str = Depends(get_current_user),
):
    """导入上传的备份归档。所有数据归到当前登录用户。"""
    if db_strategy not in {"skip", "overwrite"}:
        raise HTTPException(400, "db_strategy 必须是 'skip' 或 'overwrite'")

    filename = file.filename or "upload"
    if not (filename.endswith(".tar.gz") or filename.endswith(".tgz")):
        raise HTTPException(400, "仅支持 .tar.gz / .tgz 归档")

    tmp_dir = Path(tempfile.mkdtemp(prefix="techspar-import-"))
    archive_path = tmp_dir / "upload.tar.gz"

    total = 0
    try:
        with archive_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"归档过大（上限 {MAX_UPLOAD_BYTES // 1024 // 1024} MB）")
                out.write(chunk)

        try:
            result = import_archive(
                archive_path,
                db_strategy=db_strategy,
                overwrite_files=overwrite_files,
                rebind_user_id=user_id,
            )
        except (RuntimeError, ValueError) as e:
            raise HTTPException(400, f"归档解析失败: {e}")
    finally:
        background.add_task(_cleanup, archive_path)
        background.add_task(lambda: tmp_dir.rmdir() if tmp_dir.exists() else None)

    return {
        "ok": True,
        "schema_version": result.schema_version,
        "current_schema_version": SCHEMA_VERSION,
        "db_inserted": result.db_inserted,
        "db_skipped": result.db_skipped,
        "files_copied": result.files_copied,
        "files_skipped": result.files_skipped,
    }

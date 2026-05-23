"""跨机器数据迁移：导出/导入用户数据为 tar.gz。

CLI (scripts/export_data.py、scripts/import_data.py) 与 HTTP 端点
(routers/data_migration.py) 共享这里的实现。

HTTP 侧通过 `rebind_user_id` 将归档中的数据全部归入当前登录用户，避免跨用户
泄露 / 错配；CLI 默认保留原 user_id 以支持管理员级整库迁移。
"""
from __future__ import annotations

import io
import json
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backend.config import settings

SCHEMA_VERSION = 1
EXCLUDE_DIR_NAMES = {".index_cache", "__pycache__"}

# 与 storage/sessions.py 保持一致；目标库不存在时用它建表
_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    topic TEXT,
    meta TEXT DEFAULT '{}',
    questions TEXT DEFAULT '[]',
    transcript TEXT DEFAULT '[]',
    scores TEXT DEFAULT '[]',
    weak_points TEXT DEFAULT '[]',
    overall TEXT DEFAULT '{}',
    reference_answers TEXT DEFAULT '{}',
    review TEXT,
    status TEXT DEFAULT 'ongoing',
    review_error TEXT,
    user_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


def _data_dir() -> Path:
    return settings.base_dir / "data"


def _db_path() -> Path:
    return settings.db_path


def _users_dir() -> Path:
    return _data_dir() / "users"


@dataclass
class ImportResult:
    db_inserted: int = 0
    db_skipped: int = 0
    files_copied: int = 0
    files_skipped: int = 0
    schema_version: int | None = None


def _filter_tar_member(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(tarinfo.name).parts
    if any(name in EXCLUDE_DIR_NAMES for name in parts):
        return None
    return tarinfo


def _export_filtered_db(user_id: str, dst: Path) -> None:
    """生成只含指定 user_id 行的 DB 副本。"""
    src_path = _db_path()
    with sqlite3.connect(str(src_path)) as src, sqlite3.connect(str(dst)) as dst_conn:
        src.backup(dst_conn)
        dst_conn.execute("DELETE FROM sessions WHERE user_id != ?", (user_id,))
        dst_conn.commit()
    with sqlite3.connect(str(dst)) as dst_conn:
        dst_conn.execute("VACUUM")


def export_archive(
    output_path: Path,
    *,
    user_id: str | None = None,
) -> Path:
    """打包 data/ 为 tar.gz。

    user_id=None 表示导出全部用户（仅 CLI 用）；指定 user_id 时只导出该用户。
    返回 output_path（已确认写入完成）。
    """
    data_dir = _data_dir()
    if not data_dir.exists():
        raise FileNotFoundError(f"data 目录不存在: {data_dir}")

    output_path = Path(output_path).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "source": str(data_dir),
    }

    tmp_db: Path | None = None
    db_source = _db_path()
    if user_id and db_source.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        tmp_db = output_path.parent / f".techspar-export-{ts}.db"
        _export_filtered_db(user_id, tmp_db)
        db_source = tmp_db

    try:
        with tarfile.open(output_path, "w:gz") as tar:
            manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = int(datetime.now().timestamp())
            tar.addfile(info, io.BytesIO(manifest_bytes))

            if db_source.exists():
                tar.add(db_source, arcname="data/interviews.db")

            users_dir = _users_dir()
            if users_dir.exists():
                if user_id:
                    udir = users_dir / user_id
                    if udir.exists():
                        tar.add(udir, arcname=f"data/users/{user_id}", filter=_filter_tar_member)
                else:
                    tar.add(users_dir, arcname="data/users", filter=_filter_tar_member)
    finally:
        if tmp_db and tmp_db.exists():
            tmp_db.unlink()

    return output_path


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest_resolved)):
            raise RuntimeError(f"archive 包含越界路径: {member.name}")
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def _merge_db(
    src_db: Path,
    dst_db: Path,
    *,
    strategy: str,
    rebind_user_id: str | None = None,
) -> tuple[int, int]:
    """合并 sessions 表，返回 (写入行数, 跳过行数)。

    rebind_user_id 非空时，归档中所有行的 user_id 改写为该值——HTTP 导入用，
    防止跨用户写入；同时支持跨机迁移（user_id 在新机器上不同）。
    """
    if not dst_db.exists() and rebind_user_id is None:
        dst_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_db, dst_db)
        with sqlite3.connect(str(dst_db)) as c:
            total = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return total, 0

    dst_db.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(src_db))
    dst = sqlite3.connect(str(dst_db))
    try:
        dst.execute(_SESSIONS_DDL)

        src_cols = [r[1] for r in src.execute("PRAGMA table_info(sessions)")]
        dst_cols = {r[1] for r in dst.execute("PRAGMA table_info(sessions)")}
        common = [c for c in src_cols if c in dst_cols]
        if "session_id" not in common:
            raise RuntimeError("session_id 列缺失，无法合并")

        existing = {r[0] for r in dst.execute("SELECT session_id FROM sessions")}
        rows = src.execute(f"SELECT {', '.join(common)} FROM sessions").fetchall()

        sid_idx = common.index("session_id")
        uid_idx = common.index("user_id") if "user_id" in common else -1

        inserted = 0
        skipped = 0
        for row in rows:
            row = list(row)
            if rebind_user_id is not None and uid_idx >= 0:
                row[uid_idx] = rebind_user_id
            sid = row[sid_idx]
            if sid in existing:
                if strategy == "overwrite":
                    set_cols = [c for c in common if c != "session_id"]
                    assigns = ", ".join(f"{c} = ?" for c in set_cols)
                    vals = [row[common.index(c)] for c in set_cols]
                    dst.execute(f"UPDATE sessions SET {assigns} WHERE session_id = ?", vals + [sid])
                    inserted += 1
                else:
                    skipped += 1
            else:
                placeholders = ", ".join(["?"] * len(common))
                dst.execute(
                    f"INSERT INTO sessions ({', '.join(common)}) VALUES ({placeholders})",
                    row,
                )
                inserted += 1
        dst.commit()
        return inserted, skipped
    finally:
        src.close()
        dst.close()


def _merge_users(
    src_users: Path,
    dst_users: Path,
    *,
    overwrite: bool,
    rebind_user_id: str | None = None,
) -> tuple[int, int]:
    """复制 data/users/ 下的文件。

    rebind_user_id 非空时，归档内任意 <some_id>/ 目录的内容都被写到
    <rebind_user_id>/ 下；用于 HTTP 导入将数据归到当前登录用户。
    """
    copied = 0
    skipped = 0
    for src_file in src_users.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_users)
        if rebind_user_id is not None:
            parts = list(rel.parts)
            if not parts:
                continue
            parts[0] = rebind_user_id
            rel = Path(*parts)
        dst_file = dst_users / rel
        if dst_file.exists() and not overwrite:
            skipped += 1
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
    return copied, skipped


def import_archive(
    archive_path: Path,
    *,
    db_strategy: str = "skip",
    overwrite_files: bool = False,
    rebind_user_id: str | None = None,
) -> ImportResult:
    """导入 export_archive 生成的 tar.gz。

    db_strategy: session_id 冲突时 'skip' 保留本地，'overwrite' 用归档覆盖。
    overwrite_files: 文件冲突时是否覆盖本地。
    rebind_user_id: HTTP 入口必传——把归档数据归到该 user_id。
    """
    if db_strategy not in {"skip", "overwrite"}:
        raise ValueError("db_strategy 必须是 'skip' 或 'overwrite'")

    archive_path = Path(archive_path).resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"归档不存在: {archive_path}")

    result = ImportResult()

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, td)

        manifest_path = td / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                result.schema_version = manifest.get("schema_version")
            except json.JSONDecodeError:
                pass

        data_dir = _data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)

        src_db = td / "data" / "interviews.db"
        if src_db.exists():
            ins, skip = _merge_db(
                src_db,
                _db_path(),
                strategy=db_strategy,
                rebind_user_id=rebind_user_id,
            )
            result.db_inserted = ins
            result.db_skipped = skip

        src_users = td / "data" / "users"
        if src_users.exists():
            copied, skipped = _merge_users(
                src_users,
                _users_dir(),
                overwrite=overwrite_files,
                rebind_user_id=rebind_user_id,
            )
            result.files_copied = copied
            result.files_skipped = skipped

    return result

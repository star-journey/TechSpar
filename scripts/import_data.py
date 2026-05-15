#!/usr/bin/env python3
"""把 export_data.py 生成的备份导入到本机 data/ 目录。

DB 合并策略：按 session_id 主键，默认遇重复跳过；--db-strategy=overwrite 则覆盖现有行。
用户文件策略：默认保留已有文件，--overwrite-files 则覆盖。
"""
import argparse
import json
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "interviews.db"
USERS_DIR = DATA_DIR / "users"

SCHEMA_VERSION = 1

# 与 backend/storage/sessions.py 保持一致；目标库不存在时用它建表
SESSIONS_DDL = """
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
    answers_draft TEXT DEFAULT '[]',
    current_index INTEGER DEFAULT 0,
    status TEXT DEFAULT 'ongoing',
    review_error TEXT,
    user_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
)
"""


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    # 防御 path traversal：拒绝绝对路径和 .. 跳出
    dest_resolved = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        try:
            target.relative_to(dest_resolved)
        except ValueError as exc:
            raise RuntimeError(f"archive 包含越界路径: {member.name}") from exc
    # Python 3.12+ / 3.11.4+ 提供 'data' filter；不可用则降级
    try:
        tar.extractall(dest, filter="data")
    except TypeError:
        tar.extractall(dest)


def _merge_db(src_db: Path, dst_db: Path, strategy: str) -> tuple[int, int]:
    """返回 (写入行数, 跳过行数)。"""
    if not dst_db.exists():
        dst_db.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_db, dst_db)
        with sqlite3.connect(str(dst_db)) as c:
            total = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return total, 0

    src = sqlite3.connect(str(src_db))
    dst = sqlite3.connect(str(dst_db))
    try:
        dst.execute(SESSIONS_DDL)

        src_cols = [r[1] for r in src.execute("PRAGMA table_info(sessions)")]
        dst_cols = {r[1] for r in dst.execute("PRAGMA table_info(sessions)")}
        common = [c for c in src_cols if c in dst_cols]
        if "session_id" not in common:
            raise RuntimeError("session_id 列缺失，无法合并")

        existing = {r[0] for r in dst.execute("SELECT session_id FROM sessions")}
        rows = src.execute(f"SELECT {', '.join(common)} FROM sessions").fetchall()

        inserted = 0
        skipped = 0
        sid_idx = common.index("session_id")
        for row in rows:
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
                    list(row),
                )
                inserted += 1
        dst.commit()
        return inserted, skipped
    finally:
        src.close()
        dst.close()


def _merge_users(src_users: Path, dst_users: Path, overwrite: bool) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for src_file in src_users.rglob("*"):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(src_users)
        dst_file = dst_users / rel
        if dst_file.exists() and not overwrite:
            skipped += 1
            continue
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        copied += 1
    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path, help="备份归档路径 (.tar.gz)")
    parser.add_argument(
        "--db-strategy", choices=("skip", "overwrite"), default="skip",
        help="session_id 冲突时：skip（默认，保留本地）或 overwrite（用归档覆盖）",
    )
    parser.add_argument(
        "--overwrite-files", action="store_true",
        help="覆盖 data/users/ 下已存在的文件（默认保留本地）",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="跳过确认提示")
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.exists():
        print(f"error: 找不到归档: {archive}", file=sys.stderr)
        return 1

    print(f"目标 data 目录: {DATA_DIR}")
    print(f"归档:           {archive}")
    print(f"DB 策略:        {args.db_strategy}")
    print(f"文件覆盖:       {args.overwrite_files}")

    if not args.yes:
        ans = input("继续? [y/N] ").strip().lower()
        if ans != "y":
            print("已取消。")
            return 0

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract(tar, td)

        manifest_path = td / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sv = manifest.get("schema_version")
            if sv != SCHEMA_VERSION:
                print(f"warning: 归档 schema_version={sv}, 当前={SCHEMA_VERSION}，可能不兼容")

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        src_db = td / "data" / "interviews.db"
        if src_db.exists():
            ins, skip = _merge_db(src_db, DB_PATH, args.db_strategy)
            print(f"DB: 写入/更新 {ins} 行，跳过 {skip} 行")
        else:
            print("DB: 归档内无 interviews.db，跳过")

        src_users = td / "data" / "users"
        if src_users.exists():
            copied, skipped = _merge_users(src_users, USERS_DIR, args.overwrite_files)
            print(f"users/: 复制 {copied} 个文件，跳过 {skipped} 个")
        else:
            print("users/: 归档内无 users 目录，跳过")

    print("完成。建议重启后端以刷新索引缓存。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

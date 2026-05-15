#!/usr/bin/env python3
"""导出 TechSpar 个人数据为 tar.gz，便于跨机器迁移。

打包内容：
- data/interviews.db （面试 session、复盘、参考答案等）
- data/users/<user_id>/ （画像、简历、知识库、题库、训练偏好）

跳过：.index_cache/（可重建）、langgraph_checkpoints*（运行时图状态）、.env（含密钥）。
"""
import argparse
import io
import json
import sqlite3
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "interviews.db"
USERS_DIR = DATA_DIR / "users"

SCHEMA_VERSION = 1
EXCLUDE_DIR_NAMES = {".index_cache", "__pycache__"}


def _filter_member(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(tarinfo.name).parts
    if any(name in EXCLUDE_DIR_NAMES for name in parts):
        return None
    return tarinfo


def _export_filtered_db(user_id: str, dst: Path) -> None:
    # 用 backup API 拿到完整副本（含全部 schema/索引），再删掉其他用户的行
    with sqlite3.connect(str(DB_PATH)) as src, sqlite3.connect(str(dst)) as dst_conn:
        src.backup(dst_conn)
        dst_conn.execute("DELETE FROM sessions WHERE user_id IS NULL OR user_id != ?", (user_id,))
        dst_conn.commit()
    # VACUUM 必须在事务外执行
    with sqlite3.connect(str(dst)) as dst_conn:
        dst_conn.execute("VACUUM")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-o", "--output", type=Path,
        help="输出文件路径（默认 ./techspar-backup-<timestamp>.tar.gz）",
    )
    parser.add_argument(
        "--user-id",
        help="仅导出指定 user_id 的数据（默认导出所有用户）",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"error: 找不到 data 目录: {DATA_DIR}", file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = (args.output or Path.cwd() / f"techspar-backup-{ts}.tar.gz").resolve()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "user_id": args.user_id,
        "source": str(DATA_DIR),
    }

    print(f"导出到: {out_path}")
    if args.user_id:
        print(f"过滤 user_id: {args.user_id}")

    tmp_db: Path | None = None
    db_source = DB_PATH
    if args.user_id and DB_PATH.exists():
        tmp_db = out_path.parent / f".techspar-export-{ts}.db"
        _export_filtered_db(args.user_id, tmp_db)
        db_source = tmp_db

    try:
        with tarfile.open(out_path, "w:gz") as tar:
            manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            info.mtime = int(datetime.now().timestamp())
            tar.addfile(info, io.BytesIO(manifest_bytes))

            if db_source.exists():
                tar.add(db_source, arcname="data/interviews.db")
            else:
                print("warning: interviews.db 不存在，跳过", file=sys.stderr)

            if USERS_DIR.exists():
                if args.user_id:
                    udir = USERS_DIR / args.user_id
                    if udir.exists():
                        tar.add(udir, arcname=f"data/users/{args.user_id}", filter=_filter_member)
                    else:
                        print(f"warning: data/users/{args.user_id} 不存在", file=sys.stderr)
                else:
                    tar.add(USERS_DIR, arcname="data/users", filter=_filter_member)
    finally:
        if tmp_db and tmp_db.exists():
            tmp_db.unlink()

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"完成。大小: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""把 export_data.py 生成的备份导入到本机 data/ 目录。

DB 合并策略：按 session_id 主键，默认遇重复跳过；--db-strategy=overwrite 则覆盖现有行。
用户文件策略：默认保留已有文件，--overwrite-files 则覆盖。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.storage.data_migration import (  # noqa: E402
    SCHEMA_VERSION,
    import_archive,
)
from backend.config import settings  # noqa: E402


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

    data_dir = settings.base_dir / "data"
    print(f"目标 data 目录: {data_dir}")
    print(f"归档:           {archive}")
    print(f"DB 策略:        {args.db_strategy}")
    print(f"文件覆盖:       {args.overwrite_files}")

    if not args.yes:
        ans = input("继续? [y/N] ").strip().lower()
        if ans != "y":
            print("已取消。")
            return 0

    try:
        result = import_archive(
            archive,
            db_strategy=args.db_strategy,
            overwrite_files=args.overwrite_files,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if result.schema_version is not None and result.schema_version != SCHEMA_VERSION:
        print(f"warning: 归档 schema_version={result.schema_version}, 当前={SCHEMA_VERSION}，可能不兼容")

    print(f"DB: 写入/更新 {result.db_inserted} 行，跳过 {result.db_skipped} 行")
    print(f"users/: 复制 {result.files_copied} 个文件，跳过 {result.files_skipped} 个")
    print("完成。建议重启后端以刷新索引缓存。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""导出 TechSpar 个人数据为 tar.gz，便于跨机器迁移。

打包内容：
- data/interviews.db （面试 session、复盘、参考答案等）
- data/users/<user_id>/ （画像、简历、知识库、题库、训练偏好）

跳过：.index_cache/（可重建）、__pycache__/、.env（含密钥）。
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.storage.data_migration import export_archive  # noqa: E402


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

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = (args.output or Path.cwd() / f"techspar-backup-{ts}.tar.gz").resolve()

    print(f"导出到: {out_path}")
    if args.user_id:
        print(f"过滤 user_id: {args.user_id}")

    try:
        export_archive(out_path, user_id=args.user_id)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"完成。大小: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

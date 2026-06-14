#!/usr/bin/env bash
#
# TechSpar 一键更新 / 部署脚本。
#   1) 拉取当前分支最新提交（仅 fast-forward）
#   2) 重新构建镜像
#   3) 重建并启动容器
#   4) 清理本次构建产生的悬空镜像
#
# 合并 基础 docker-compose.yml + 本机 docker-compose.prod.yml（端口/绑定/healthcheck，
# 仅存在于服务器、被 .gitignore 忽略，不随仓库分发）。
# 用法:  bash scripts/update-techspar.sh
set -euo pipefail

# ── 定位仓库根目录（脚本位于 <repo>/scripts/）─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

log() { printf '\033[1;32m[update]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[update]\033[0m %s\n' "$*" >&2; }

# ── 选择 docker compose 命令（v2 插件优先，回退 legacy）──────────────────
if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  err "未找到 docker compose，请先安装 Docker Compose。"
  exit 1
fi
# 基础 + 本机覆盖一起传给每个 compose 子命令
COMPOSE+=(-f docker-compose.yml -f docker-compose.prod.yml)

# ── 预检：.env 与 本机覆盖文件 必须存在 ───────────────────────────────────
if [[ ! -f .env ]]; then
  err "仓库根目录缺少 .env，请先 cp .env.example .env 并填写后再运行。"
  exit 1
fi
if [[ ! -f docker-compose.prod.yml ]]; then
  err "缺少 docker-compose.prod.yml（仅服务器的端口/绑定/healthcheck 覆盖，不随仓库分发）。"
  err "请在本机部署目录创建后再运行（参考部署说明）。"
  exit 1
fi

# ── 拉取当前分支最新提交 ──────────────────────────────────────────────────
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
log "从 origin 更新分支 '$BRANCH' ..."
git fetch --prune origin
BEFORE="$(git rev-parse HEAD)"
if ! git merge --ff-only "origin/$BRANCH"; then
  err "无法 fast-forward 分支 '$BRANCH'（本地与 origin 已分叉）。"
  err "请手动处理后重试，例如：git status  /  git log HEAD..origin/$BRANCH"
  exit 1
fi
AFTER="$(git rev-parse HEAD)"

if [[ "$BEFORE" == "$AFTER" ]]; then
  log "已是最新（$AFTER）。仍重新构建以应用可能的 .env / 覆盖改动。"
else
  log "已更新 $BEFORE -> $AFTER"
  git --no-pager log --oneline "$BEFORE..$AFTER" | sed 's/^/    /'
fi

# ── 构建镜像并重建容器 ───────────────────────────────────────────────────
log "构建镜像 ..."
"${COMPOSE[@]}" build

log "启动容器 ..."
"${COMPOSE[@]}" up -d

# ── 清理本次构建产生的悬空镜像（仅 dangling，安全）──────────────────────
log "清理悬空镜像 ..."
docker image prune -f >/dev/null

log "完成。当前状态："
"${COMPOSE[@]}" ps

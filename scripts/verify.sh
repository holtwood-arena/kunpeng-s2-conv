#!/usr/bin/env bash
# 小尺寸正确性冒烟 + 官方四案对拍
# 用法: ./scripts/verify.sh [binary]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build/conv2d_test}"
[ -x "$BIN" ] || { echo "缺少可执行文件: $BIN"; exit 1; }

echo "=== 冒烟（缩小尺寸，快速失败）==="
for k in 7 15 39; do
  printf '  %4dx%4d k=%2d ... ' 512 512 "$k"
  if "$BIN" 512 512 "$k" "$k" 1 2>&1 | grep -q PASS; then echo PASS; else echo "FAIL"; exit 1; fi
done

echo "=== 官方四案（全量，耗时较长）==="
for spec in "4096 6144 39 39" "6144 4096 41 41" "4256 6390 55 55" "6390 4256 81 81"; do
  set -- $spec
  printf '  %sx%s k=%sx%s ... ' "$1" "$2" "$3" "$4"
  "$BIN" "$1" "$2" "$3" "$4" 1 2>&1 | tail -1
done

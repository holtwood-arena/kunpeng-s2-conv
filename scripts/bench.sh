#!/usr/bin/env bash
# 采集性能并落盘（标注环境，避免把本地数据当平台数据）
# 用法: ./scripts/bench.sh [binary] [tag]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build/conv2d_test}"
TAG="${2:-local}"
STAMP="$(date +%Y%m%dT%H%M%S)"
OUT="bench/results/${TAG}_${STAMP}.log"
mkdir -p bench/results

{
  echo "# tag=$TAG  time=$STAMP"
  echo "# uname=$(uname -srm)"
  echo "# cpu=$(lscpu 2>/dev/null | awk -F': +' '/Model name/{print $2; exit}')"
  echo "# omp_threads=${OMP_NUM_THREADS:-unset}"
  echo "# binary=$BIN"
} | tee "$OUT"

for spec in "4096 6144 39 39" "6144 4096 41 41" "4256 6390 55 55" "6390 4256 81 81"; do
  set -- $spec
  for r in 1 2 3; do
    echo "--- case $1x$2 k=$3x$4 run=$r" | tee -a "$OUT"
    "$BIN" "$1" "$2" "$3" "$4" 1 2>&1 | tee -a "$OUT"
  done
done

echo "写入 $OUT"

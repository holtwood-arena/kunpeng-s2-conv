#!/usr/bin/env bash
# 编译指定版本的 conv2d 与官方 benchmark
# 用法: ./scripts/build.sh [src_file] [out_bin]
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-src/conv2d_v0.c}"
OUT="${2:-build/conv2d_test}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O3 -fopenmp -march=native}"

mkdir -p build
echo "[build] $SRC -> $OUT"
echo "[build] CC=$CC CFLAGS=$CFLAGS"
"$CC" $CFLAGS bench/bench_conv.c "$SRC" -lm -o "$OUT"
echo "[build] done"

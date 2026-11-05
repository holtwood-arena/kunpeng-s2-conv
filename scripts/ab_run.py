#!/usr/bin/env python3
"""交错 A/B 采集：两个二进制轮流跑同一组用例，抵消频率漂移。

用法:
    python3 scripts/ab_run.py A=build/v1 B=build/v6 --tag E02 --runs 3 \
        --threads 6 --cpus 0-5 --out bench/results/E02-20261105-x86.csv
"""
import argparse
import csv
import datetime
import os
import re
import subprocess

CASES = [
    (4096, 6144, 39, 39),
    (6144, 4096, 41, 41),
    (4256, 6390, 55, 55),
    (6390, 4256, 81, 81),
]

ROW_RE = re.compile(
    r"^\s*(\d+)\s*x\s*(\d+)\s+(\d+)\s*x\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.eE+-]+|nan)\s+(PASS|FAIL)"
)


def run_one(binary, case, threads, cpus, extra_env=None):
    rows, cols, kr, kc = case
    cmd = [binary, str(rows), str(cols), str(kr), str(kc), "1"]
    if cpus:
        cmd = ["taskset", "-c", cpus] + cmd
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    if extra_env:
        env.update(extra_env)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    for line in out.stdout.splitlines():
        m = ROW_RE.match(line)
        if m:
            return {
                "ms": float(m.group(5)),
                "gflops": float(m.group(6)),
                "err": m.group(7),
                "status": m.group(8),
            }
    raise RuntimeError(f"无法解析输出:\n{out.stdout}\n{out.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+",
                    help="NAME=PATH 或 NAME=PATH@KEY=VAL;KEY2=VAL2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cpus", default="0-5")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    variants = []
    for spec in args.variants:
        name, path = spec.split("=", 1)
        # 支持 NAME=PATH@KEY=VAL;KEY2=VAL2 的形式，给每个变体独立的 OpenMP 环境
        env = {}
        if "@" in path:
            path, envspec = path.split("@", 1)
            for kv in envspec.split(";"):
                k, v = kv.split("=", 1)
                env[k] = v
        variants.append((name, path, env))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    records = []

    print(f"[ab] tag={args.tag} threads={args.threads} cpus={args.cpus or 'unbound'}", flush=True)
    for case in CASES:
        for r in range(1, args.runs + 1):
            for name, path, venv in variants:
                res = run_one(path, case, args.threads, args.cpus, venv)
                records.append({
                    "variant": name,
                    "case": CASES.index(case) + 1,
                    "rows": case[0], "cols": case[1],
                    "kr": case[2], "kc": case[3],
                    "run": r, **res,
                })
                print(f"  case{CASES.index(case)+1} run{r} {name:>10s} "
                      f"{res['gflops']:8.4f} GFLOPS  err={res['err']} {res['status']}", flush=True)

    with open(args.out, "w", newline="") as f:
        f.write(f"# {args.tag} 原始数据\n")
        f.write(f"# platform=LOCAL_REFERENCE  threads={args.threads}  taskset={args.cpus}\n")
        f.write(f"# collected={datetime.datetime.now().isoformat(timespec='seconds')}\n")
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    print(f"[ab] 写入 {args.out}（{len(records)} 行）")


if __name__ == "__main__":
    main()

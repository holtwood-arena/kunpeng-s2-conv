#!/usr/bin/env python3
"""对 ab_run.py 产出的 CSV 做统计：按 (variant, case) 取中位数、抖动、变体间比值。

用法:
    python3 scripts/summarize.py bench/results/E02-20261105-x86.csv
    python3 scripts/summarize.py <csv> --base v1 --cand v6
"""
import argparse
import csv
import statistics as st
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for r in csv.DictReader(lines):
        r["gflops"] = float(r["gflops"])
        r["case"] = int(r["case"])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--base", help="基准变体名")
    ap.add_argument("--cand", help="对比变体名")
    args = ap.parse_args()

    rows = load(args.csv)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["case"])].append(r["gflops"])

    variants = sorted({r["variant"] for r in rows})
    cases = sorted({r["case"] for r in rows})

    print(f"{'case':>4} {'variant':>16} {'median':>10} {'min':>10} {'max':>10} {'jitter%':>8}")
    med = {}
    for c in cases:
        for v in variants:
            v_vals = groups.get((v, c), [])
            if not v_vals:
                continue
            m = st.median(v_vals)
            med[(v, c)] = m
            jitter = (max(v_vals) - min(v_vals)) / m * 100 if m else 0.0
            print(f"{c:>4} {v:>16} {m:>10.4f} {min(v_vals):>10.4f} {max(v_vals):>10.4f} {jitter:>8.2f}")

    if args.base and args.cand:
        print(f"\n相对变化（{args.cand} vs {args.base}）")
        print(f"{'case':>4} {'base':>10} {'cand':>10} {'delta%':>9} {'ratio':>8}")
        for c in cases:
            b = med.get((args.base, c))
            d = med.get((args.cand, c))
            if b is None or d is None:
                continue
            print(f"{c:>4} {b:>10.4f} {d:>10.4f} {(d-b)/b*100:>+9.2f} {d/b:>8.3f}")

        deltas = [((med[(args.cand, c)] - med[(args.base, c)]) / med[(args.base, c)] * 100)
                  for c in cases if (args.base, c) in med and (args.cand, c) in med]
        if deltas:
            print(f"\n中位数delta: {st.median(deltas):+.2f}%   "
                  f"范围: {min(deltas):+.2f}% ~ {max(deltas):+.2f}%")


if __name__ == "__main__":
    main()

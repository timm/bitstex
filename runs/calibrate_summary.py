#!/usr/bin/env python3
"""
Summarize calibrate.log produced by calibrate_oracle.py.

Reports:
  - Number of tasks
  - Quartiles of median rho across tasks
  - Percent of tasks with rho > {0.5, 0.7, 0.8, 0.9}
  - Per-subdir aggregate

Run: python3 data/calibrate_summary.py data/calibrate.log
"""
import re, sys
from collections import defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "data/calibrate.log"

KEY = re.compile(r":(\w+)\s+(\S+)")

def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = dict(KEY.findall(line))
            if "rho_med" in d:
                rows.append({
                    "file": d["file"],
                    "rho": float(d["rho_med"]),
                    "subdir": d["file"].split("/")[0],
                })
    return rows

def quart(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 4], xs[n // 2], xs[(3 * n) // 4]

if __name__ == "__main__":
    rows = parse(LOG)
    n = len(rows)
    rhos = [r["rho"] for r in rows]
    q1, med, q3 = quart(rhos)
    print(f"Tasks scored:        {n}")
    print(f"rho_med distribution (across tasks):")
    print(f"  Q1                 {q1:.3f}")
    print(f"  Median             {med:.3f}")
    print(f"  Q3                 {q3:.3f}")
    print(f"  Min / Max          {min(rhos):.3f} / {max(rhos):.3f}")
    print()
    for thresh in (0.5, 0.7, 0.8, 0.9):
        pct = 100 * sum(1 for r in rhos if r > thresh) / n
        print(f"  rho > {thresh:.1f}        {pct:5.1f}% ({sum(1 for r in rhos if r > thresh)}/{n})")
    print()
    by_dir = defaultdict(list)
    for r in rows: by_dir[r["subdir"]].append(r["rho"])
    print(f"{'subdir':<25}{'n':>5}{'med':>8}{'Q1':>8}{'Q3':>8}")
    print("-" * 54)
    for d in sorted(by_dir):
        v = by_dir[d]
        q1, m, q3 = quart(v)
        print(f"{d:<25}{len(v):>5}{m:>8.3f}{q1:>8.3f}{q3:>8.3f}")

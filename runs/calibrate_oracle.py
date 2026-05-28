#!/usr/bin/env python3
"""
Calibrate the 50-row nearest-neighbor surrogate oracle against ground truth.

For each MOOT optimization task:
  1. Shuffle rows.
  2. Take 50 rows as the 'known' surrogate-training set.
  3. For each remaining row r:
       true_y(r)      = disty(r)                       # using r's actual Y
       surrogate_y(r) = disty(nearest_neighbor(r))     # via known set
  4. Compute Spearman rank correlation between true_y and surrogate_y.
  5. Repeat REPEATS times (different shuffles), report median + IQR per file.

Output: one log line per file, written to stdout.
Run:    python3 data/calibrate_oracle.py > data/calibrate.log
"""
import os, sys, glob
from random import shuffle, seed

sys.path.insert(0, os.path.expanduser("~/gits/timm/ezr"))
from ezr import Data, csv, disty, nearest

MOOT_ROOT = os.path.expanduser("~/gits/moot/optimize")
KNOWN_N   = 50      # surrogate-training set per the paper
UNSEEN_N  = 500     # cap unseen sample for tractable runtime on large files
REPEATS   = 20
SEED      = 1

def ranks(xs):
    """Average ranks (handles ties)."""
    pairs = sorted(enumerate(xs), key=lambda p: p[1])
    r = [0.0] * len(xs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][1] == pairs[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[pairs[k][0]] = avg
        i = j + 1
    return r

def spearman(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = (sum((a - mx) ** 2 for a in rx)) ** 0.5
    dy = (sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / (dx * dy + 1e-32)

def surrogate_disty(known_data, row):
    """disty of row's nearest neighbor in known."""
    near = nearest(known_data, row, known_data.rows)
    proxy = row[:]
    for col in known_data.cols.ys:
        proxy[col.at] = near[col.at]
    return disty(known_data, proxy)

def calibrate(file):
    """Return list of Spearman rhos across REPEATS shuffles."""
    d = Data(csv(file))
    if len(d.rows) < KNOWN_N + 10:
        return None
    rows = d.rows
    names = d.cols.names
    rhos = []
    for _ in range(REPEATS):
        shuffle(rows)
        known = Data([names] + rows[:KNOWN_N])
        unseen = rows[KNOWN_N:KNOWN_N + UNSEEN_N]
        true_y = [disty(d, r) for r in unseen]
        surr_y = [surrogate_disty(known, r) for r in unseen]
        rhos.append(spearman(true_y, surr_y))
    return rhos

def summarize(rhos):
    rhos = sorted(rhos)
    n = len(rhos)
    med = rhos[n // 2]
    q1  = rhos[n // 4]
    q3  = rhos[(3 * n) // 4]
    return med, q1, q3

def work(f):
    rel = os.path.relpath(f, MOOT_ROOT)
    try:
        seed(SEED + hash(f) % 10000)  # per-process determinism
        rhos = calibrate(f)
        if rhos is None:
            return None, f":file {rel} :skip too_few_rows"
        med, q1, q3 = summarize(rhos)
        return (f":file {rel} :rho_med {med:.3f} "
                f":rho_q1 {q1:.3f} :rho_q3 {q3:.3f} :n {len(rhos)}"), None
    except Exception as e:
        return None, f":file {rel} :error {type(e).__name__}:{e}"

if __name__ == "__main__":
    from multiprocessing import Pool, cpu_count
    files = sorted(glob.glob(os.path.join(MOOT_ROOT, "**", "*.csv"),
                             recursive=True))
    nproc = max(1, cpu_count() - 1)
    print(f"# {len(files)} MOOT optimize files; KNOWN={KNOWN_N} "
          f"UNSEEN={UNSEEN_N} REPEATS={REPEATS} SEED={SEED} "
          f"PROCS={nproc}", file=sys.stderr)
    with Pool(nproc) as pool:
        for out, err in pool.imap_unordered(work, files):
            if out:
                print(out); sys.stdout.flush()
            if err:
                print(err, file=sys.stderr); sys.stderr.flush()

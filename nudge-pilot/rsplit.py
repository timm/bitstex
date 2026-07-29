#!/usr/bin/env python3 -B
"""
rsplit: for the NSF proposal's second [TIMM] note (July
2026). BINGO compression r = occupied buckets / rows, one
fixed grid rule everywhere (greedy diverse top-4 split-cost
dims, 8 bins per numeric), then medians for measured config
data versus model-generated feature models.

Result (2026-07-29, b=8): median r = 0.025 measured (36
files), 0.004 generated (11 files) -- both far under the
r < 0.05 threshold, so BINGO is not an artifact of
model-generated data, though generated models compress
harder. NOTE: the 2026-07-21 pilot quoted r=0.017 measured;
that run silently used b=4 bins (binid bound its default at
def time, so setting nudge.B=8 had no effect). If the
proposal sentence quotes 0.017, it should say b=4 -- or use
0.025 with b=8. Same conclusion either way.
"""
import glob, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/gits/timm/src/ezr-py"))
from xai import *
import nudge
from nudge import ranked, buckets

D2, CAP = 4, 4096
nudge.B = 8

def grid_dims(tbl):
  "Top-4 informative dims, skipping redundant cols"
  ats = []
  for at in ranked(tbl):
    if len(ats) >= D2: break
    b4 = len(buckets(tbl, ats)) if ats else 1
    ats += [at]
    if len(buckets(tbl, ats)) == b4: ats.pop()
  return ats

def r_of(file):
  random.seed(the.seed)
  t0  = Tbl(csv(file))
  tbl = clone(t0, some(t0.rows, CAP))
  bs  = buckets(tbl, grid_dims(tbl))
  return len(bs) / len(tbl.rows), len(tbl.rows), len(bs)

MEASURED  = (glob.glob(path("$MOOT/optimize/config/*.csv"))
             + [path("$MOOT/optimize/binary_config/billing10k.csv")])
GENERATED = [f for f in
             glob.glob(path("$MOOT/optimize/binary_config/*.csv"))
             if os.path.basename(f)[:2] in ("Sc", "FM", "FF")]

if __name__ == "__main__":
  out = {}
  for tag, files in (("measured", MEASURED),
                     ("generated", GENERATED)):
    rs = []
    for f in sorted(files):
      r, n, nb = r_of(f)
      rs += [r]
      print("%-9s %-28s n=%5d buckets=%4d r=%.3f" %
            (tag, os.path.basename(f), n, nb, r))
    out[tag] = statistics.median(rs)
    print("%s: %d datasets, median r = %.3f" %
          (tag, len(rs), out[tag]))
  print("\nSENTENCE: Across our pilot runs, median "
        "compression was r=%.3f on the measured "
        "configuration data and r=%.3f on the "
        "model-generated feature models." %
        (out["measured"], out["generated"]))

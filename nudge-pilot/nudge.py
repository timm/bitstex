#!/usr/bin/env python3 -B
"""
nudge: pilot of the NUDGE loop (flag, contrast, nudge,
verify) for the NSF proposal's [TIMM] note, p6 (July 2026).
Layered on ezr-py's xai/xaiplus; nothing here edits them.

Loop, per config task:
  1. BINGO-bucket all rows on greedy-picked informative,
     non-redundant x-dims, b bins per numeric.
  2. Trust a bucket if it holds >= MIN rows and its disty
     spread is at most the median spread of the big
     buckets; else flag (pilot proxy for the proposal's
     repeated-runs fluctuation test).
  3. For each flagged row: target = the most desirable
     (lowest median d2h) trusted bucket reachable by <=3
     flips; else the nearest trusted bucket.
  4. Nudge: set just the differing dims to the trusted
     bucket's mid values (protected cols untouched);
     verify = the nudged row re-buckets into a trusted
     bucket. Delta disty scored by snapping the nudged row
     to its nearest real row in its landing bucket
     (surrogate labels).
  5. Control: same verify on a 3-random-column mutation
     (xaiplus.picks), per RQ2's random-perturbation control.

Output: the [TIMM] table in LaTeX. Needs ~/gits/moot.
"""
import os, sys, statistics
sys.path.insert(0, os.path.expanduser("~/gits/timm/src/ezr-py"))
from xai import *
import xai
from xaiplus import picks, nearest

B    = 4             # bins per numeric dim
MIN  = 8             # min rows for a trusted bucket
MAXD = 8             # most dims the grid may use
CAP  = 1024          # max rows used per task

def ranked(tbl):
  "All x-cols, best (lowest) split cost first"
  Y, best = lambda r: disty(tbl, r), {}
  for at in tbl.x:
    for cost, a, _ in bins(tbl, tbl.rows, at, Y):
      best[a] = min(best.get(a, BIG), cost)
  return sorted(best, key=best.get) or tbl.x[:]

def binid(col, v, b=None):
  if v == "?": return "?"
  if is_sym(col): return v
  return min((b or B) - 1, int(norm(col, v) * (b or B)))

def key(tbl, ats, row):
  return tuple(binid(tbl.cols[at], row[at]) for at in ats)

def buckets(tbl, ats):
  out = {}
  for r in tbl.rows: out.setdefault(key(tbl, ats, r), []).append(r)
  return out

def spread(tbl, rows):
  return sd(adds(disty(tbl, r) for r in rows))

def cut_of(tbl, bs):
  "Fluctuation cut: median y-spread of the big buckets"
  sps = sorted(spread(tbl, rs) for rs in bs.values()
               if len(rs) >= MIN)
  return sps[len(sps) // 2] if sps else 0

def flagged_bucket(tbl, rows, cut):
  "Thin evidence: too few rows, or y-values fluctuate"
  return len(rows) < MIN or spread(tbl, rows) > cut

def flagfrac(tbl, ats):
  "Fraction of rows living in flagged buckets"
  bs  = buckets(tbl, ats)
  cut = cut_of(tbl, bs)
  return sum(len(rs) for rs in bs.values()
             if flagged_bucket(tbl, rs, cut)) / len(tbl.rows)

def dims(tbl):
  """Greedy grid growth: informative, non-redundant dims
  (skip a col that opens no new corners), then keep the
  prefix whose flagged fraction is nonzero, at most half,
  and nearest 20%."""
  ats = []
  for at in ranked(tbl):
    if len(ats) >= MAXD: break
    b4 = len(buckets(tbl, ats)) if ats else 1
    ats += [at]
    if len(buckets(tbl, ats)) == b4: ats.pop()
  best, gap = 3, BIG
  for d in range(2, len(ats) + 1):
    f = flagfrac(tbl, ats[:d])
    if 0 < f <= .5 and abs(f - .2) < gap:
      best, gap = d, abs(f - .2)
  return ats[:best]

def nudge(tbl, ats, row, k, trust):
  """Target = the most desirable (lowest median d2h)
  trusted bucket reachable by <=3 flips; else the nearest.
  Flip <=3 differing dims to that bucket's mids."""
  reach = [k2 for k2 in trust
           if sum(a != b for a, b in zip(k, k2)) <= 3]
  kt  = (min(reach, key=lambda k2: trust[k2][2]) if reach
         else min(trust,
                  key=lambda k2: distx(tbl, row, trust[k2][1])))
  mid = trust[kt][1]
  out, flips = row[:], 3
  for at, a, b in zip(ats, k, kt):
    if a != b and at not in tbl.protect and flips > 0:
      out[at] = mid[at]; flips -= 1
  return out

def verify(tbl, ats, row, new, bs, trustkeys):
  """(repaired?, delta disty), delta>0 = nudged row better.
  The nudged row's y comes from its landing bucket: snap to
  the nearest real row there (surrogate labels)."""
  k2   = key(tbl, ats, new)
  pool = ([r for r in bs.get(k2, []) if r is not row]
          or [r for r in tbl.rows if r is not row])
  d = disty(tbl, row) - disty(tbl, nearest(tbl, new, pool))
  return k2 in trustkeys, d

def pilot(file, name):
  random.seed(the.seed)
  t0  = Tbl(csv(file))
  tbl = clone(t0, some(t0.rows, CAP))
  ats = dims(tbl)
  bs  = buckets(tbl, ats)
  cut = cut_of(tbl, bs)
  trust = {k: (rows, mids(clone(tbl, rows)),
               statistics.median(disty(tbl, r) for r in rows))
           for k, rows in bs.items()
           if not flagged_bucket(tbl, rows, cut)}
  flagged = [(k, r) for k, rows in bs.items()
             if k not in trust for r in rows]
  if not trust or not flagged:
    return dict(name=name, rows=len(tbl.rows), r=0,
                flagged=len(flagged), fail=True)
  okn, dn, okr, dr = 0, [], 0, []
  for k, row in flagged:
    new    = nudge(tbl, ats, row, k, trust)
    ok, d  = verify(tbl, ats, row, new, bs, trust)
    okn   += ok; dn += [d]
    rnd    = picks(tbl, row, 3)
    ok, d  = verify(tbl, ats, row, rnd, bs, trust)
    okr   += ok; dr += [d]
  return dict(name=name, rows=len(tbl.rows),
              r=len(bs) / len(tbl.rows), nb=len(bs),
              nt=len(trust), flagged=len(flagged),
              pn=100 * okn / len(flagged),
              pr=100 * okr / len(flagged),
              dn=statistics.median(dn),
              dr=statistics.median(dr),
              diff=not same(dn, dr), fail=False)

TASKS = [("SS-A", "$MOOT/optimize/config/SS-A.csv"),
         ("SS-M", "$MOOT/optimize/config/SS-M.csv"),
         ("X264", "$MOOT/optimize/config/X264_AllMeasurements.csv"),
         ("SQL", "$MOOT/optimize/config/SQL_AllMeasurements.csv"),
         ("Apache", "$MOOT/optimize/config/Apache_AllMeasurements.csv")]

if __name__ == "__main__":
  rs = [pilot(f, n) for n, f in TASKS]
  print(r"\begin{tabular}{lrrrrrrrrl}")
  print(r"\toprule")
  print(r"task & rows & $r$ & flagged & \multicolumn{2}{c}"
        r"{repaired (\%)} & \multicolumn{2}{c}"
        r"{median $\Delta$d2h} & distinct? \\")
  print(r" & & & & nudge & random & nudge & random & \\")
  print(r"\midrule")
  for x in rs:
    if x["fail"]:
      print(r"%s & %d & -- & %d & \multicolumn{5}{c}{no "
            r"trusted buckets} \\" %
            (x["name"], x["rows"], x["flagged"]))
    else:
      print(r"%s & %d & %.3f & %d & %.0f & %.0f & "
            r"%+.3f & %+.3f & %s \\" %
            (x["name"], x["rows"], x["r"], x["flagged"],
             x["pn"], x["pr"], x["dn"], x["dr"],
             "yes" if x["diff"] else "no"))
  print(r"\bottomrule")
  print(r"\end{tabular}")

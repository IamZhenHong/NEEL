"""Summarise the Phase 3 dev sweep (exploratory). For each donor x position set x layer, mean over 12 dev
templates of the patched-minus-unpatched grade for each author role, plus:
  G change = mean of the neutral and newcomer changes (positive = general drop reduced)
  T change = insulter change - neutral change
Reference: unpatched drop (assistant_insult - donor) for neutral+newcomer, i.e. what full removal would undo.
Usage: uv run python scripts/analyze_phase3_dev.py [results_dir]
"""
import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
B = {(r["condition"], r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"]): r["expected"]
     for r in map(json.loads, open(d / "phase3_dev_base.jsonl"))}
items = sorted({k[1] for k in B})
for donor in ("flat_v2", "third_party_v2"):
    ref = [mean(B[("assistant_insult",) + k[1:]] - B[(donor,) + k[1:]] for k in B if k[0] == donor and k[1] == it and k[4] != k[2])
           for it in items]
    print(f"reference general drop vs {donor}: {mean(ref):+.3f} (negative in {sum(x < 0 for x in ref)}/12)")
acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
mass = defaultdict(list)
for r in map(json.loads, open(d / "phase3_dev_patch.jsonl")):
    e0 = B[("assistant_insult", r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"])]
    acc[(r["donor"], r["posset"], r["layer"])][r["item_id"]][r["author_role"]].append(r["expected"] - e0)
    mass[(r["donor"], r["posset"], r["layer"])].append(r["mass"])
print(f"\n{'donor':15} {'posset':16} {'L':>3}  {'insulter':>9} {'neutral':>8} {'newcomer':>9} {'G chg':>7} {'G+':>5} {'T chg':>7} {'mass':>6}")
for c in sorted(acc, key=lambda c: (c[0], c[1], c[2])):
    per = acc[c]
    pv = [mean(per[it]["provoker"]) for it in per]; nv = [mean(per[it]["neutral"]) for it in per]
    cv = [mean(per[it]["newcomer"]) for it in per]; g = [(a + b) / 2 for a, b in zip(nv, cv)]
    print(f"{c[0]:15} {c[1]:16} {c[2]:>3}  {mean(pv):+9.3f} {mean(nv):+8.3f} {mean(cv):+9.3f} {mean(g):+7.3f} "
          f"{sum(x > 0 for x in g):>2}/{len(g):<2} {mean(pv) - mean(nv):+7.3f} {min(mass[c]):6.3f}")

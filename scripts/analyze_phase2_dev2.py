"""Summarise the Phase 2 dev additions (exploratory). Per position set and layer, mean over 12 dev
templates of the change vs the unpatched run for each author role, the targeted-gap change
(insulter change minus neutral change), templates with a positive gap change, and tokens patched.
Forward rows: recipient assistant_insult (positive gap change = penalty reduced).
Reverse rows: recipient third_party_v2 (negative gap change = penalty induced).

Usage: uv run python scripts/analyze_phase2_dev2.py [results_dir]
"""
import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
base = {(r["condition"], r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"]): r["expected"]
        for r in map(json.loads, open(d / "phase2_dev2_base.jsonl"))}
rows = [json.loads(l) for l in open(d / "phase2_dev2_patch.jsonl")]
items = sorted({r["item_id"] for r in rows})
ref = []
for it in items:
    def gap(c):
        ks = [k for k in base if k[0] == c and k[1] == it]
        pv = {k[2:4] + (k[5],): base[k] for k in ks if k[4] == k[2]}
        nv = {k[2:4] + (k[5],): base[k] for k in ks if k[4] not in (k[2], "User-3")}
        return mean(pv[x] - nv[x] for x in pv)
    ref.append(gap("assistant_insult") - gap("third_party_v2"))
print(f"DEV: {len(items)} templates. Unpatched targeted difference (assistant_insult - third_party_v2): {mean(ref):+.3f}\n")
acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
ntok = defaultdict(list); mass = defaultdict(list)
for r in rows:
    c = (r["direction"], r["posset"], r["layer"])
    e0 = base[(r["condition"], r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"])]
    acc[c][r["item_id"]][r["author_role"]].append(r["expected"] - e0)
    ntok[c].append(r["n_patched"]); mass[c].append(r["mass"])
print(f"{'dir':7} {'posset':15} {'L':>3} {'tokens':>7}  {'insulter':>9} {'neutral':>8} {'newcomer':>9} {'gap chg':>8} {'+ in':>6} {'min mass':>8}")
for c in sorted(acc, key=lambda c: (c[0], c[2], c[1])):
    per = acc[c]
    pv = [mean(per[it]["provoker"]) for it in per]; nv = [mean(per[it]["neutral"]) for it in per]
    cv = [mean(per[it]["newcomer"]) for it in per]; g = [a - b for a, b in zip(pv, nv)]
    print(f"{c[0]:7} {c[1]:15} {c[2]:>3} {mean(ntok[c]):7.1f}  {mean(pv):+9.3f} {mean(nv):+8.3f} {mean(cv):+9.3f} {mean(g):+8.3f} "
          f"{sum(x > 0 for x in g):>3}/{len(g):<2} {min(mass[c]):8.3f}")

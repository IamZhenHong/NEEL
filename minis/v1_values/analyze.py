"""V1 analysis (as frozen in DESIGN.md). Usage: python3 minis/v1_values/analyze.py -> results/analysis.json + stdout."""
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
CONDS, BON, SETS = ["base", "equal", "total"], [0, 1, 2, 4, 8, 16], ["people", "teams"]
ch = [json.loads(l) for l in open(R / "choices.jsonl")]
st = [json.loads(l) for l in open(R / "stated.jsonl")]
unparsed = sum(r["equal"] is None for r in ch)
ok = [r for r in ch if r["equal"] is not None]

cell = defaultdict(list)   # (cond, setting, bonus, equal_first) -> [0/1]
for r in ok:
    cell[(r["cond"], r["setting"], r["bonus"], r["equal_first"])].append(int(r["equal"]))
p = {k: sum(v) / len(v) for k, v in cell.items()}


def peq(cond, setting=None, bonus=None, ef=None):
    vs = [x for (c, s, b, e), v in cell.items() if c == cond and (setting is None or s == setting)
          and (bonus is None or b == bonus) and (ef is None or e == ef) for x in v]
    return sum(vs) / len(vs) if vs else None


def switch(cond, setting):
    ps = [peq(cond, setting, b) for b in BON]
    if ps[0] < 0.5:
        return "<0"
    for b, q in zip(BON, ps):
        if q < 0.5:
            return b
    return ">16"


def contrast(c1, c2, keys):
    """mean over matched cells of p(c1) - p(c2); bootstrap over cells (10k, seed 0)."""
    d = [p[(c1, *k)] - p[(c2, *k)] for k in keys if (c1, *k) in p and (c2, *k) in p]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"diff": statistics.mean(d), "ci": [bs[249], bs[9749]], "n_cells": len(d)}


allk = [(s, b, e) for s in SETS for b in BON for e in (True, False)]
out = {"n_choices": len(ch), "unparsed": unparsed, "flag_unparsed_gt10pct": unparsed > 0.1 * len(ch),
       "spend_usd": round(sum(json.loads(l)["cost"] for l in open(R / "api_ledger.jsonl")), 3),
       "p_equal": {c: {s: {b: peq(c, s, b) for b in BON} for s in SETS} for c in CONDS},
       "p_equal_by_order": {c: {("EQfirst" if e else "EQsecond"): peq(c, ef=e) for e in (True, False)} for c in CONDS},
       "switch_point": {c: {s: switch(c, s) for s in SETS} for c in CONDS},
       "contrasts": {f"{a}-{b}": contrast(a, b, allk) for a, b in [("equal", "total"), ("base", "total"),
                                                                    ("equal", "base")]},
       "contrast_by_setting": {s: contrast("equal", "total", [k for k in allk if k[0] == s]) for s in SETS},
       "contrast_by_order": {("EQfirst" if e else "EQsecond"): contrast("equal", "total", [k for k in allk if k[2] == e])
                             for e in (True, False)},
       "stated": {}}
for c in CONDS:
    pr = Counter(r.get("priority") for r in st if r["cond"] == c and r["q"] == "Q1")
    th = [r.get("threshold") for r in st if r["cond"] == c and r["q"] == "Q2"]
    nums = [t for t in th if isinstance(t, int)]
    out["stated"][c] = {"priority": dict(pr), "thresholds": th, "n_never": th.count("never"),
                        "median_numeric": statistics.median(nums) if nums else None}
json.dump(out, open(R / "analysis.json", "w"), indent=1)

print(f"V1: {len(ch)} choices, {unparsed} unparsed; spend ${out['spend_usd']}")
for c in CONDS:
    for s in SETS:
        print(f"  {c:6s} {s:6s} P(equal) by bonus {BON}: " + " ".join(f"{peq(c, s, b):.2f}" for b in BON)
              + f"   switch: {switch(c, s)}")
for k, v in out["contrasts"].items():
    print(f"  {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
for k, v in {**out["contrast_by_setting"], **out["contrast_by_order"]}.items():
    print(f"  equal-total within {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
print("  order:", out["p_equal_by_order"])
for c, v in out["stated"].items():
    print(f"  stated {c}: {v['priority']} | thresholds never={v['n_never']} median={v['median_numeric']}")

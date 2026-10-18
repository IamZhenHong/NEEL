"""V1b analysis (as frozen in DESIGN.md). Usage: python3 minis/v1b_values_offceiling/analyze.py -> results/analysis.json."""
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
CONDS = ["base", "equal", "total"]
WIDE, OFF, V1B = [16, 32, 64, 128, 256], [0, 1, 2, 4, 8, 16], [0, 1, 2, 4, 8]


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


wide, off = load(R / "wide.jsonl"), load(R / "off.jsonl")
v1 = [r for r in load(HERE.parent / "v1_values" / "results" / "choices.jsonl") if r["setting"] == "people"]


def cells(rows):
    c = defaultdict(list)
    for r in rows:
        if r.get("equal") is not None:
            c[(r["cond"], r["bonus"], r["equal_first"])].append(int(r["equal"]))
    return c


def peq(c, cond, bonus=None, ef=None):
    vs = [x for (cc, b, e), v in c.items() if cc == cond and (bonus is None or b == bonus) and (ef is None or e == ef)
          for x in v]
    return sum(vs) / len(vs) if vs else None


def switch(c, cond, grid, top):
    ps = [peq(c, cond, b) for b in grid]
    if ps[0] is not None and ps[0] < 0.5 and grid[0] == 0:
        return "<0"
    for b, q in zip(grid, ps):
        if q is not None and q < 0.5:
            return b
    return f">{top}"


def contrast(c, c1, c2, keys):
    p = {k: sum(v) / len(v) for k, v in c.items()}
    d = [p[(c1, *k)] - p[(c2, *k)] for k in keys if (c1, *k) in p and (c2, *k) in p]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"diff": statistics.mean(d), "ci": [bs[249], bs[9749]], "n_cells": len(d)}


cw, co = cells(wide), cells(off)
cv = cells([r for r in v1 if r["bonus"] in V1B])
call = defaultdict(list)
for src in (cv, cw):
    for k, v in src.items():
        call[k] += v
okeys = [(b, e) for b in OFF for e in (True, False)]
wkeys = [(b, e) for b in WIDE for e in (True, False)]
out = {
    "spend_usd": round(sum(json.loads(l)["cost"] for l in open(R / "api_ledger.jsonl")), 3),
    "spend_calibration_usd": round(sum(r["cost"] for r in load(R / "calibration.jsonl")), 3),
    "wide": {"n": len(wide), "unparsed": sum(r.get("equal") is None for r in wide),
             "p_equal": {c: {b: peq(cw, c, b) for b in WIDE} for c in CONDS},
             "switch_point_v1_plus_wide": {c: switch(call, c, V1B + WIDE, 256) for c in CONDS},
             "contrasts": {f"{a}-{b}": contrast(cw, a, b, wkeys) for a, b in
                           [("equal", "base"), ("base", "total"), ("equal", "total")]}},
    "off": {"n": len(off), "unparsed": sum(r.get("equal") is None for r in off),
            "p_equal": {c: {b: peq(co, c, b) for b in OFF} for c in CONDS},
            "p_equal_pooled": {c: peq(co, c) for c in CONDS},
            "p_equal_by_order": {c: {("EQfirst" if e else "EQsecond"): peq(co, c, ef=e) for e in (True, False)}
                                 for c in CONDS},
            "switch_point": {c: switch(co, c, OFF, 16) for c in CONDS},
            "contrasts": {f"{a}-{b}": contrast(co, a, b, okeys) for a, b in
                          [("equal", "base"), ("base", "total"), ("equal", "total")]},
            "equal_minus_base_by_order": {("EQfirst" if e else "EQsecond"):
                                          contrast(co, "equal", "base", [k for k in okeys if k[1] == e])
                                          for e in (True, False)}},
}
for br in ("wide", "off"):
    n = out[br]["n"]
    out[br]["flag_unparsed_gt10pct"] = n > 0 and out[br]["unparsed"] > 0.1 * n
json.dump(out, open(R / "analysis.json", "w"), indent=1)

print(f"V1b spend ${out['spend_usd']} (calibration ${out['spend_calibration_usd']})")
for br, grid in (("wide", WIDE), ("off", OFF)):
    o = out[br]
    print(f"[{br}] n={o['n']} unparsed={o['unparsed']}")
    for c in CONDS:
        print(f"  {c:6s} P(equal) {grid}: " + " ".join("  -  " if o['p_equal'][c][b] is None else
                                                      f"{o['p_equal'][c][b]:.2f}" for b in grid))
    for k, v in o["contrasts"].items():
        print(f"  {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] ({v['n_cells']} cells)")
print("  switch (V1 people 0-8 + wide):", out["wide"]["switch_point_v1_plus_wide"])
print("  switch (off):", out["off"]["switch_point"], " pooled:", out["off"]["p_equal_pooled"])
print("  by order:", out["off"]["p_equal_by_order"])
for k, v in out["off"]["equal_minus_base_by_order"].items():
    print(f"  equal-base within {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")

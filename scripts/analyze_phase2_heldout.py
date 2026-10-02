"""Phase 2 held-out frozen analysis (see prereg/phase2_heldout.md). Do not change after the
phase2-heldout-freeze tag.

Usage: uv run python scripts/analyze_phase2_heldout.py [results_dir]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev

from scipy import stats

ALPHA = 0.025
NEUT_TOL = {"I1_author_label": 0.05, "I2_request_nograde": 0.10}


def main(d):
    d = Path(d)
    base = {(r["condition"], r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"]): r
            for r in map(json.loads, open(d / "phase2_heldout_base.jsonl"))}
    rows = [json.loads(l) for l in open(d / "phase2_heldout_patch.jsonl")]
    items = sorted({r["item_id"] for r in rows})

    def gap(cond, it):
        ks = [k for k in base if k[0] == cond and k[1] == it]
        pv = {(k[2], k[3], k[5]): base[k]["expected"] for k in ks if base[k]["author_role"] == "provoker"}
        nv = {(k[2], k[3], k[5]): base[k]["expected"] for k in ks if base[k]["author_role"] == "neutral"}
        return mean(pv[x] - nv[x] for x in pv)

    ref = [gap("assistant_insult", it) - gap("third_party_v2", it) for it in items]
    print(f"Held-out: {len(items)} templates, git {rows[0]['git']}. Unpatched targeted difference "
          f"(assistant_insult - third_party_v2): {mean(ref):+.3f} (sd {stdev(ref):.3f})\n")

    by = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    lg = defaultdict(lambda: defaultdict(list))
    massmin = defaultdict(lambda: 1.0)
    spread = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = (r["direction"], r["intervention"])
        b = base[(r["condition"], r["item_id"], r["speaker"], r["order"], r["author"], r["paragraph_id"])]
        by[c][r["item_id"]][r["author_role"]].append(r["expected"] - b["expected"])
        k = max(range(9), key=lambda i: b["label_probs"][i])
        if k >= 1:
            g = lambda lp: math.log(max(lp[k], 1e-12)) - math.log(max(lp[k - 1], 1e-12))  # noqa: E731
            lg[c][r["author_role"]].append(g(r["label_probs"]) - g(b["label_probs"]))
        massmin[c] = min(massmin[c], r["mass"])
        spread[c][(r["item_id"], r["speaker"], r["order"], r["author"])].append((r["expected"], b["expected"]))

    def summ(c):
        per = by[c]
        pv = [mean(per[it]["provoker"]) for it in items]; nv = [mean(per[it]["neutral"]) for it in items]
        cv = [mean(per[it]["newcomer"]) for it in items]
        return pv, nv, cv, [a - b for a, b in zip(pv, nv)]

    def disc(c):
        sp_p = [pstdev([x for x, _ in v]) for v in spread[c].values() if len(v) > 1]
        sp_u = [pstdev([y for _, y in v]) for v in spread[c].values() if len(v) > 1]
        return mean(sp_p) / mean(sp_u) if sp_u else float("nan")

    print("=== PRIMARY (forward: contractor-condition activations into the assistant-insult run, layer 17) ===")
    for name in ("I1_author_label", "I2_request_nograde"):
        c = ("forward", name)
        pv, nv, cv, g = summ(c)
        n, m = len(g), mean(g)
        lo, hi = stats.t.interval(1 - ALPHA, n - 1, loc=m, scale=stdev(g) / n ** 0.5)
        eff = lo > 0
        k1 = mean(summ(("forward", "K1_nearby_request"))[3])
        pres = {"neutral |change| <= tol": abs(mean(nv)) <= NEUT_TOL[name], "valid mass >= 0.95": massmin[c] >= 0.95,
                "discrimination >= 0.8x": disc(c) >= 0.8, "K1 |gap change| < 0.03": abs(k1) < 0.03}
        print(f"  {name}: gap change {m:+.3f}  97.5% CI [{lo:+.3f}, {hi:+.3f}]  positive {sum(x > 0 for x in g)}/{n}  -> {'EFFECT' if eff else 'no effect'}")
        print(f"    insulter {mean(pv):+.3f}  neutral {mean(nv):+.3f}  newcomer {mean(cv):+.3f}  "
              f"log-prob gap change insulter {mean(lg[c]['provoker']):+.3f} neutral {mean(lg[c]['neutral']):+.3f}")
        print(f"    preservation: " + "  ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in pres.items())
              + f"  (min mass {massmin[c]:.3f}, discrimination {disc(c):.2f}x)")
        print(f"    VERDICT: {'selective attenuation' if eff and all(pres.values()) else ('effect, preservation failed' if eff else 'not shown')}")

    print("\n=== COMPARISON, CONTROLS, REVERSE ===")
    for c in [("forward", "C1_tail_before_request"), ("forward", "K1_nearby_request"), ("forward", "K2_self"),
              ("reverse", "I1_author_label"), ("reverse", "I2_request_nograde")]:
        pv, nv, cv, g = summ(c)
        m = mean(g)
        lo, hi = stats.t.interval(0.95, len(g) - 1, loc=m, scale=stdev(g) / len(g) ** 0.5) if stdev(g) > 0 else (m, m)
        print(f"  {c[0]:7} {c[1]:24} gap change {m:+.3f} [95% {lo:+.3f}, {hi:+.3f}]  insulter {mean(pv):+.3f} neutral {mean(nv):+.3f} "
              f"newcomer {mean(cv):+.3f}  positive {sum(x > 0 for x in g)}/{len(g)}")
    k2 = max(abs(x) for it in by[("forward", "K2_self")].values() for v in it.values() for x in v)
    print(f"\n  K2 self-patch max |change|: {k2:.2e} ({'ok' if k2 < 1e-4 else 'FAIL'})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")

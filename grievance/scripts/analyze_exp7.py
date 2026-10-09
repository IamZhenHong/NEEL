"""Experiment 7 frozen analysis (see prereg/exp7.md). Do not change after the exp7-freeze tag.

Usage: uv run python scripts/analyze_exp7.py [results/exp7_scripted.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp6 import decide  # noqa: E402  (same decision rule, alpha 0.025, SESOI 0.10)
from scipy import stats  # noqa: E402

CELLS = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]
CONDS = ("assistant_insult", "third_party_insult", "negative_context")


def per_template(rows):
    out = {}
    for it in sorted({r["item_id"] for r in rows}):
        idx = {(r["condition"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r["expected_score"]
               for r in rows if r["item_id"] == it}
        ps = sorted({k[4] for k in idx})
        y = lambda c, s, o, role, p: idx[(c, s, o, role, p)]  # noqa: E731
        avg = lambda f: mean(f(s, o, p) for s, o in CELLS for p in ps)  # noqa: E731
        gap = lambda c: avg(lambda s, o, p: y(c, s, o, "provoker", p) - y(c, s, o, "neutral", p))  # noqa: E731
        d = {}
        for c in CONDS:
            d[f"T {c}"] = gap(c) - gap("flat")
            d[f"G {c}"] = avg(lambda s, o, p, c=c: mean(y(c, s, o, r, p) - y("flat", s, o, r, p) for r in ("neutral", "newcomer")))
            d[f"prov {c}"] = avg(lambda s, o, p, c=c: y(c, s, o, "provoker", p) - y("flat", s, o, "provoker", p))
        d["A"] = d["T assistant_insult"] - d["T third_party_insult"]
        d["B"] = d["G assistant_insult"] - d["G negative_context"]
        out[it] = d
    return out


def ci95(vals):
    n, m = len(vals), mean(vals)
    se = stats.tstd(vals) / n ** 0.5
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=se)
    return m, lo, hi, sum(v < 0 for v in vals), n


def main(path):
    rows = [json.loads(l) for l in open(path)]
    pt = per_template(rows)
    vals = lambda k: [pt[it][k] for it in pt]  # noqa: E731
    print(f"Experiment 7: {len(pt)} templates, {len(rows)} rows, git {rows[0]['git']}, dtype {rows[0]['dtype']}, "
          f"valid mass min {min(r['valid_mass'] for r in rows):.3f}\n")
    print("=== PRIMARY ===")
    decide(vals("A"), "F1 A = T_assistant_insult - T_third_party_insult (assistant-specificity of the targeted part)")
    decide(vals("B"), "F2 B = G_assistant_insult - G_negative_context (hostility-specificity of the general part)")

    print("\n=== SECONDARY (95% t-intervals over templates; not decisive) ===")
    print(f"  {'quantity':34}{'mean':>8}   {'95% CI':>18}   negative")
    for k in [f"{q} {c}" for q in ("T", "G", "prov") for c in CONDS]:
        m, lo, hi, neg, n = ci95(vals(k))
        print(f"  {k:34}{m:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {neg}/{n}")
    m = mean(vals("prov assistant_insult"))
    print(f"\n  replication: insulter assistant_insult - flat = {m:+.3f} (Experiment 6: -0.348; diff {m+0.348:+.3f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/exp7_scripted.jsonl")

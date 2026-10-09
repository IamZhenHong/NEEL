"""Experiment 10 frozen analysis (see prereg/exp10.md). Do not change after the exp10-freeze tag.

Unit = template; average over 4 speaker x order cells and selected paragraphs.
y = expected grade. "drop" = y(assistant_insult) - y(third_party_v2) for the same prompt type.

PRIMARY (Bonferroni, alpha = 0.05/3 each, two-sided, 98.33% t-intervals):
  M1 anonymisation:  drop(anonymous) - drop(named insulter)        > 0 = anonymising removes the extra penalty
  M2 B2_request:     T(B2) - T(named),  T = drop(insulter) - drop(neutral)   > 0 = blocking reduces the targeted penalty
  M3 B1_label:       T(B1) - T(named)
Usage: uv run python scripts/analyze_exp10.py [results/exp10.jsonl]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from statistics import mean, pstdev, stdev

from scipy import stats

ALPHA = 0.05 / 3
CELLS = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]


def decide(vals, name):
    n, m = len(vals), mean(vals)
    se = stdev(vals) / n ** 0.5
    lo, hi = stats.t.interval(1 - ALPHA, n - 1, loc=m, scale=se)
    print(f"  {name}\n    n {n}  mean {m:+.3f}  98.33% CI [{lo:+.3f}, {hi:+.3f}]  t p {2*stats.t.sf(abs(m/se), n-1):.4f}  "
          f"positive {sum(v > 0 for v in vals)}/{n}  VERDICT: {'EFFECT' if lo > 0 or hi < 0 else 'INCONCLUSIVE'}")


def ci95(vals):
    m = mean(vals)
    lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=m, scale=stdev(vals) / len(vals) ** 0.5)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main(path):
    rows = [json.loads(l) for l in open(path)]
    items = sorted({r["item_id"] for r in rows})
    Y = {(r["kind"], r["condition"], r["item_id"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r for r in rows}
    print(f"Experiment 10: {len(items)} templates, {len(rows)} rows, git {rows[0]['git']}; "
          f"valid mass min {min(r['mass'] for r in rows):.3f}\n")

    def per(fn):
        out = []
        for it in items:
            ps = sorted({k[6] for k in Y if k[2] == it and k[0] == "named"})
            out.append(mean(fn(it, s, o, p) for s, o in CELLS for p in ps))
        return out

    y = lambda kind, c, it, s, o, role, p: Y[(kind, c, it, s, o, role, p)]["expected"]  # noqa: E731
    drop = lambda kind, it, s, o, role, p: y(kind, "assistant_insult", it, s, o, role, p) - y(kind, "third_party_v2", it, s, o, role, p)  # noqa: E731
    T = lambda kind: (lambda it, s, o, p: drop(kind, it, s, o, "provoker", p) - drop(kind, it, s, o, "neutral", p))  # noqa: E731

    print("=== PRIMARY ===")
    decide(per(lambda it, s, o, p: drop("anon", it, s, o, "anonymous", p) - drop("named", it, s, o, "provoker", p)),
           "M1 anonymisation: drop(anonymous) - drop(named insulter)")
    for kind, label in (("B2_request", "M2 block request -> treatment turns"), ("B1_label", "M3 block author label -> treatment turns")):
        decide(per(lambda it, s, o, p, kind=kind: T(kind)(it, s, o, p) - T("named")(it, s, o, p)), f"{label}: T(blocked) - T(unblocked)")

    print("\n=== SECONDARY (95% CIs) ===")
    for role in ("provoker", "neutral", "newcomer"):
        print(f"  named drop, {role:9}: {ci95(per(lambda it, s, o, p, role=role: drop('named', it, s, o, role, p)))}")
    print(f"  anonymous drop        : {ci95(per(lambda it, s, o, p: drop('anon', it, s, o, 'anonymous', p)))}")
    print(f"  anonymous - named neutral drop: {ci95(per(lambda it, s, o, p: drop('anon', it, s, o, 'anonymous', p) - drop('named', it, s, o, 'neutral', p)))}")
    print(f"  T named (targeted penalty): {ci95(per(T('named')))}")
    for kind in ("B1_label", "B2_request", "B3_control"):
        print(f"  {kind:11} T change {ci95(per(lambda it, s, o, p, kind=kind: T(kind)(it, s, o, p) - T('named')(it, s, o, p)))}  | drop change: "
              + "  ".join(f"{role} {mean(per(lambda it, s, o, p, kind=kind, role=role: drop(kind, it, s, o, role, p) - drop('named', it, s, o, role, p))):+.3f}"
                          for role in ("provoker", "neutral", "newcomer")))
        # absolute effect of the block within each condition
        for c in ("assistant_insult", "third_party_v2"):
            d = mean(per(lambda it, s, o, p, kind=kind, c=c: mean(y(kind, c, it, s, o, r, p) - y("named", c, it, s, o, r, p) for r in ("provoker", "neutral", "newcomer"))))
            print(f"      mean grade change in {c:17}: {d:+.3f}")
    disc = defaultdict(list)
    for kind in ("named", "B1_label", "B2_request", "B3_control"):
        g = defaultdict(list)
        for k, r in Y.items():
            if k[0] == kind:
                g[k[1:6]].append(r["expected"])
        disc[kind] = mean(pstdev(v) for v in g.values() if len(v) > 1)
    print("  paragraph discrimination (mean within-cell spread): " + "  ".join(f"{k} {v:.2f}" for k, v in disc.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/exp10.jsonl")

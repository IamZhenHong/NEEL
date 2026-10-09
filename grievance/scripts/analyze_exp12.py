"""Experiment 12 frozen analysis (see prereg/exp12.md). Do not change after the exp12-freeze tag.

Per template (average over 4 speaker x order cells and selected paragraphs), with y = expected grade:
  G(base, block) = mean over neutral and newcomer of [y(assistant_insult) - y(base)] under the block
  T(base, block) = [y(prov) - y(neut)]_assistant_insult - [y(prov) - y(neut)]_base   under the block
Change due to a block = value(block) - value(none); positive = the provocation effect is reduced.

PRIMARY (Bonferroni alpha 0.05/3, 98.33% t-intervals over 24 templates):
  Q1  G(flat_v2) change under B_tail
  Q2  G(flat_v2) change under B_req
  Q3  T(flat_v2) change under B_tail
Usage: uv run python scripts/analyze_exp12.py [results/exp12.jsonl]
"""
import json
import sys
from statistics import mean, pstdev, stdev
from collections import defaultdict

from scipy import stats

ALPHA = 0.05 / 3
CELLS = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results/exp12.jsonl")]
Y = {(r["condition"], r["block"], r["item_id"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r["expected"] for r in rows}
items = sorted({r["item_id"] for r in rows})
print(f"Experiment 12: {len(items)} templates, {len(rows)} rows, git {rows[0]['git']}, valid mass min {min(r['mass'] for r in rows):.3f}\n")


def per(fn):
    out = []
    for it in items:
        ps = sorted({k[6] for k in Y if k[2] == it})
        out.append(mean(fn(it, s, o, p) for s, o in CELLS for p in ps))
    return out


y = lambda c, b, it, s, o, role, p: Y[(c, b, it, s, o, role, p)]  # noqa: E731
G = lambda base, b: (lambda it, s, o, p: mean(y("assistant_insult", b, it, s, o, r, p) - y(base, b, it, s, o, r, p) for r in ("neutral", "newcomer")))  # noqa: E731
T = lambda base, b: (lambda it, s, o, p: (y("assistant_insult", b, it, s, o, "provoker", p) - y("assistant_insult", b, it, s, o, "neutral", p))
                     - (y(base, b, it, s, o, "provoker", p) - y(base, b, it, s, o, "neutral", p)))  # noqa: E731


def decide(v, name):
    n, m = len(v), mean(v)
    lo, hi = stats.t.interval(1 - ALPHA, n - 1, loc=m, scale=stdev(v) / n ** 0.5)
    print(f"  {name}: mean {m:+.3f}  98.33% CI [{lo:+.3f}, {hi:+.3f}]  positive {sum(x > 0 for x in v)}/{n}  -> {'EFFECT' if lo > 0 or hi < 0 else 'INCONCLUSIVE'}")


def ci(v):
    m = mean(v)
    lo, hi = stats.t.interval(0.95, len(v) - 1, loc=m, scale=stdev(v) / len(v) ** 0.5)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


chg = lambda f, base, b: per(lambda it, s, o, p: f(base, b)(it, s, o, p) - f(base, "none")(it, s, o, p))  # noqa: E731
print("=== PRIMARY ===")
decide(chg(G, "flat_v2", "B_tail"), "Q1 general drop vs flat_v2, change under B_tail")
decide(chg(G, "flat_v2", "B_req"), "Q2 general drop vs flat_v2, change under B_req")
decide(chg(T, "flat_v2", "B_tail"), "Q3 targeted penalty vs flat_v2, change under B_tail")
print("\n=== SECONDARY (95% CIs) ===")
for base in ("flat_v2", "third_party_v2"):
    print(f"  base {base}: unblocked G {ci(per(G(base, 'none')))}   unblocked T {ci(per(T(base, 'none')))}")
    for b in ("B_tail", "B_req", "B_both", "B_ctrl"):
        print(f"    {b:7} G change {ci(chg(G, base, b))}   T change {ci(chg(T, base, b))}   "
              f"remaining G {mean(per(G(base, b))):+.3f}  remaining T {mean(per(T(base, b))):+.3f}")
disc = {}
for b in ("none", "B_tail", "B_req", "B_both", "B_ctrl"):
    g = defaultdict(list)
    for k, v in Y.items():
        if k[1] == b:
            g[(k[0],) + k[2:6]].append(v)
    disc[b] = mean(pstdev(v) for v in g.values() if len(v) > 1)
mean_grade = {b: mean(v for k, v in Y.items() if k[1] == b) for b in ("none", "B_tail", "B_req", "B_both", "B_ctrl")}
print("  paragraph discrimination: " + "  ".join(f"{b} {v:.2f}" for b, v in disc.items()))
print("  mean grade: " + "  ".join(f"{b} {v:.2f}" for b, v in mean_grade.items()))

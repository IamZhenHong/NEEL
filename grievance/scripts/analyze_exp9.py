"""Experiment 9 frozen analysis (see prereg/exp9.md). Do not change after the exp9-freeze tag.

Usage: uv run python scripts/analyze_exp9.py [results/exp9_scripted.jsonl]
"""
from __future__ import annotations

import json
import random
import sys
from statistics import mean, stdev

from scipy import stats

ALPHA = 0.05 / 3
SESOI = 0.10
CELLS = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]
ROLES = ("provoker", "neutral", "newcomer")


def signflip(vals, n=200_000, seed=0):
    rng = random.Random(seed)
    obs = abs(mean(vals))
    return (sum(abs(mean(v if rng.random() < 0.5 else -v for v in vals)) >= obs - 1e-12 for _ in range(n)) + 1) / (n + 1)


def decide(vals, name):
    n, m, sd = len(vals), mean(vals), stdev(vals)
    se = sd / n ** 0.5
    lo, hi = stats.t.interval(1 - ALPHA, n - 1, loc=m, scale=se)
    elo, ehi = stats.t.interval(1 - 2 * ALPHA, n - 1, loc=m, scale=se)
    v = "EFFECT" if (lo > 0 or hi < 0) else ("EQUIVALENT" if (elo > -SESOI and ehi < SESOI) else "INCONCLUSIVE")
    print(f"  {name}\n    n {n}  mean {m:+.3f}  sd {sd:.3f}  98.33% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"t p {2*stats.t.sf(abs(m/se), n-1):.4f}  sign-flip p {signflip(vals):.4f}  negative {sum(x < 0 for x in vals)}/{n}\n"
          f"    VERDICT: {v}")


def ci95(vals):
    m = mean(vals)
    lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=m, scale=stdev(vals) / len(vals) ** 0.5)
    return m, lo, hi


def main(path):
    rows = [json.loads(l) for l in open(path)]
    per = {}
    for it in sorted({r["item_id"] for r in rows}):
        idx = {(r["condition"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r["expected_score"]
               for r in rows if r["item_id"] == it}
        ps = sorted({k[4] for k in idx})
        y = lambda c, s, o, ro, p: idx[(c, s, o, ro, p)]  # noqa: E731
        avg = lambda f: mean(f(s, o, p) for s, o in CELLS for p in ps)  # noqa: E731
        gap = lambda c: avg(lambda s, o, p: y(c, s, o, "provoker", p) - y(c, s, o, "neutral", p))  # noqa: E731
        role = lambda c, b, ro: avg(lambda s, o, p: y(c, s, o, ro, p) - y(b, s, o, ro, p))  # noqa: E731
        G = lambda c, b: mean(role(c, b, r) for r in ("neutral", "newcomer"))  # noqa: E731
        d = {"R1": gap("assistant_insult") - gap("flat_v2"),
             "R2": gap("assistant_insult") - gap("third_party_v2"),
             "R3": role("assistant_insult", "flat_v2", "newcomer"),
             "G asst - G 3rd_v2": G("assistant_insult", "flat_v2") - G("third_party_v2", "flat_v2"),
             "G asst - G neg_v2": G("assistant_insult", "flat_v2") - G("negative_v2", "flat_v2"),
             "insulter asst - flat_v2": role("assistant_insult", "flat_v2", "provoker"),
             "T (v1) asst vs flat": gap("assistant_insult") - gap("flat"),
             "A (v1) asst vs 3rd": gap("assistant_insult") - gap("third_party_insult")}
        for c2, c1 in (("flat_v2", "flat"), ("third_party_v2", "third_party_insult"), ("negative_v2", "negative_context")):
            for r in ROLES:
                d[f"len {c2} - {c1} [{r}]"] = role(c2, c1, r)
        per[it] = d
    v = lambda k: [per[it][k] for it in per]  # noqa: E731
    print(f"Experiment 9: {len(per)} templates, {len(rows)} rows, git {rows[0]['git']}, dtype {rows[0]['dtype']}, "
          f"valid mass min {min(r['valid_mass'] for r in rows):.3f}\n")
    print("=== PRIMARY (alpha 0.0167 each) ===")
    decide(v("R1"), "R1 targeted effect, assistant insult vs length-matched flat_v2")
    decide(v("R2"), "R2 assistant-specificity, assistant insult vs length-matched third_party_v2")
    decide(v("R3"), "R3 newcomer change, assistant insult vs length-matched flat_v2")
    print("\n=== SECONDARY (95% t-intervals; not decisive) ===")
    for k in ["G asst - G 3rd_v2", "G asst - G neg_v2", "insulter asst - flat_v2", "T (v1) asst vs flat", "A (v1) asst vs 3rd"] + \
             [k for k in next(iter(per.values())) if k.startswith("len ")]:
        m, lo, hi = ci95(v(k))
        print(f"  {k:42} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  negative {sum(x < 0 for x in v(k))}/{len(per)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/exp9_scripted.jsonl")

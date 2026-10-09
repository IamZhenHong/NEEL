"""Experiment 6 frozen analysis. Do not change after the freeze tag (see prereg/exp6.md).

Unit = template. Per template, average over 4 speaker x order cells and the template's
selected paragraphs. Templates with no selected paragraph are excluded (pre-registered).

PRIMARY (two, Bonferroni alpha = 0.025 each, two-sided):
  P1  free replies:  T = [y(prov) - y(neut)]_insult - [y(prov) - y(neut)]_flat
  P2  scripted:      D = y(prov)_insult - y(prov)_flat
Decision per primary:
  EFFECT        97.5% CI (t, df = n-1) excludes 0
  EQUIVALENT    95% CI lies inside [-0.10, +0.10]  (TOST at alpha 0.025 per side)
  INCONCLUSIVE  otherwise
Also reported for each primary: randomisation sign-flip p (200k random flips, seed 0).
Everything else printed below the primary block is EXPLORATORY.

Usage: uv run python scripts/analyze_exp6.py [results_dir]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

ALPHA = 0.025
SESOI = 0.10
CELLS = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]
ROLES = ("provoker", "neutral", "newcomer")


def per_template(rows, fn):
    out = {}
    for it in sorted({r["item_id"] for r in rows}):
        idx = {(r["condition"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r["expected_score"]
               for r in rows if r["item_id"] == it}
        paras = sorted({k[4] for k in idx})
        if paras:
            out[it] = mean(fn(idx, s, o, p) for s, o in CELLS for p in paras)
    return out


def T(idx, s, o, p):
    g = lambda c: idx[(c, s, o, "provoker", p)] - idx[(c, s, o, "neutral", p)]  # noqa: E731
    return g("assistant_insult") - g("flat")


def D_role(role):
    return lambda idx, s, o, p: idx[("assistant_insult", s, o, role, p)] - idx[("flat", s, o, role, p)]


def signflip(vals, n=200_000, seed=0):
    rng = random.Random(seed)
    obs = abs(mean(vals))
    hits = sum(abs(mean(v if rng.random() < 0.5 else -v for v in vals)) >= obs - 1e-12 for _ in range(n))
    return (hits + 1) / (n + 1)


def decide(vals, name):
    n, m, sd = len(vals), mean(vals), stdev(vals)
    se = sd / n ** 0.5
    lo, hi = stats.t.interval(1 - ALPHA, n - 1, loc=m, scale=se)          # 97.5% CI
    elo, ehi = stats.t.interval(1 - 2 * ALPHA, n - 1, loc=m, scale=se)    # 95% CI for TOST
    p_t = 2 * stats.t.sf(abs(m / se), n - 1)
    verdict = "EFFECT" if (lo > 0 or hi < 0) else ("EQUIVALENT (|effect| < 0.10)" if (elo > -SESOI and ehi < SESOI)
                                                   else "INCONCLUSIVE")
    print(f"  {name}")
    print(f"    n templates {n}   mean {m:+.3f}   sd {sd:.3f}   97.5% CI [{lo:+.3f}, {hi:+.3f}]   95% CI [{elo:+.3f}, {ehi:+.3f}]")
    print(f"    t-test p {p_t:.4f}   sign-flip p {signflip(vals):.4f}   negative in {sum(v < 0 for v in vals)}/{n} templates")
    print(f"    VERDICT: {verdict}")


def main(d):
    d = Path(d)
    sel = json.loads((d / "exp6_selection.json").read_text())
    scripted = [json.loads(l) for l in open(d / "exp6_scripted.jsonl")]
    free = [json.loads(l) for l in open(d / "exp6_freereply.jsonl")]
    used = {k: v for k, v in sel["selected"].items() if v}
    print(f"Experiment 6: {len(used)}/{len(sel['selected'])} templates usable, "
          f"{sum(len(v) for v in used.values())} paragraphs (threshold {sel['threshold']}); git {scripted[0]['git']}; "
          f"dtype {scripted[0]['dtype']}")
    print(f"valid-label mass min: scripted {min(r['valid_mass'] for r in scripted):.3f}, "
          f"free {min(r['valid_mass'] for r in free):.3f}\n")

    p1 = per_template(free, T)
    p2 = per_template(scripted, D_role("provoker"))
    print("=== PRIMARY ===")
    decide(list(p1.values()), "P1 free replies: targeted gap change T (insult vs flat)")
    decide(list(p2.values()), "P2 scripted: insulter's own score, insult minus flat")

    print("\n=== EXPLORATORY (not confirmatory) ===")
    for label, rows in (("scripted", scripted), ("free replies", free)):
        t = per_template(rows, T)
        comp = {r: per_template(rows, D_role(r)) for r in ROLES}
        print(f"  {label}: T mean {mean(t.values()):+.3f} | insult - flat: "
              + "  ".join(f"{r} {mean(comp[r].values()):+.3f}" for r in ROLES))
    print("\n  per template (P1, P2):")
    for it in sorted(p1):
        print(f"    {it}  P1 {p1[it]:+.3f}  P2 {p2.get(it, float('nan')):+.3f}  paragraphs {len(used[it])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")

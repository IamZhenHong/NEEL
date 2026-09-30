"""Experiment 3 analysis across items (design v4.1 §7.3, §7.5).

For each item: select paragraphs with the pre-registered fresh-context rule
(top-label p <= 0.80; if < 3 qualify, <= 0.90), then compute estimands at the chosen dose,
averaged over the 4 speaker x order cells and the selected paragraphs. Items are the unit:
report the across-item mean, a bootstrap CI over items, and how many items agree in sign.

Usage: uv run python scripts/analyze_items.py results/pilot_003_qwen3-8b.jsonl [--dose 3]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from statistics import mean

ROLES = ["provoker", "neutral", "newcomer"]
SEL = (0.80, 0.90)


def boot_ci(vals, n=10000, seed=0):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(mean(rng.choices(vals, k=len(vals))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def signflip_p(vals):
    """Exact two-sided sign-flip test of mean != 0 over items (valid for small n)."""
    from itertools import product
    obs = abs(mean(vals))
    n_ge = total = 0
    for signs in product((1, -1), repeat=len(vals)):
        total += 1
        n_ge += abs(mean(v * s for v, s in zip(vals, signs))) >= obs - 1e-12
    return n_ge / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="results/pilot_003_qwen3-8b.jsonl")
    ap.add_argument("--dose", type=int, default=3)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.path)]
    items = sorted({r["item_id"] for r in rows})
    d = args.dose
    print(f"{len(rows)} rows, {len(items)} items, dose {d}, model {rows[0]['model']}, git {rows[0]['git']}")
    m = [r["valid_mass"] for r in rows]
    print(f"valid mass min {min(m):.3f}; digit top token {sum(r['top_token'].strip().isdigit() for r in rows)}/{len(rows)}\n")

    per_item = defaultdict(dict)
    print("PER-ITEM SELECTION (fresh context) AND SATURATION (in-conversation flat, mean p(top))")
    for it in items:
        fresh = [r for r in rows if r["item_id"] == it and r["author_role"] == "fresh"]
        sel = []
        for t in SEL:
            sel = [r["paragraph_id"] for r in fresh if r["top_label_prob"] <= t]
            if len(sel) >= 3:
                break
        main_rows = [r for r in rows if r["item_id"] == it and r["author_role"] != "fresh" and r["dose"] == d]
        idx = {(r["condition"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r for r in main_rows}
        conds = sorted({r["condition"] for r in main_rows})
        cells = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]
        fs = "  ".join(f"{r['paragraph_id'][:12]}:{r['expected_score']:.1f}/{r['top_label_prob']:.2f}" for r in fresh)
        print(f"  {it}: fresh E/p(top) {fs}")
        print(f"     selected (<= {t}): {sel}")
        if not sel:
            continue
        sat = {p: mean(idx[("flat", s, o, r, p)]["top_label_prob"] for s, o in cells for r in ROLES) for p in sel}
        print("     in-conversation flat p(top): " + "  ".join(f"{p[:12]}:{v:.2f}" for p, v in sat.items()))

        def y(c, s, o, role, p):
            return idx[(c, s, o, role, p)]["expected_score"]

        def avg(f):
            return mean(f(s, o, p) for s, o in cells for p in sel)

        def gap(c):
            return avg(lambda s, o, p: y(c, s, o, "provoker", p) - y(c, s, o, "neutral", p))

        def allroles(c, base):
            return avg(lambda s, o, p: mean(y(c, s, o, r, p) - y(base, s, o, r, p) for r in ROLES))

        def role_delta(c, base, role):
            return avg(lambda s, o, p: y(c, s, o, role, p) - y(base, s, o, role, p))

        e = per_item[it]
        for c in conds:
            e[f"gap {c}"] = gap(c)
        e["T asst vs flat"] = gap("assistant_insult") - gap("flat")
        e["T 3rd vs flat"] = gap("third_party_insult") - gap("flat")
        e["T negctx vs flat"] = gap("negative_context") - gap("flat")
        e["A (asst - 3rd)"] = gap("assistant_insult") - gap("third_party_insult")
        e["T asst vs polite"] = gap("assistant_insult") - gap("polite")
        e["bonus flat - coord"] = gap("flat") - gap("coordinator_flat")
        e["warmth: polite - flat"] = allroles("polite", "flat")
        e["neg: asst - flat"] = allroles("assistant_insult", "flat")
        e["neg: 3rd - flat"] = allroles("third_party_insult", "flat")
        e["neg: negctx - flat"] = allroles("negative_context", "flat")
        e["coord - flat (all)"] = allroles("coordinator_flat", "flat")
        e["provoker: asst - flat"] = role_delta("assistant_insult", "flat", "provoker")
        e["neutral: asst - flat"] = role_delta("assistant_insult", "flat", "neutral")
        e["S newcomer: asst - flat"] = role_delta("assistant_insult", "flat", "newcomer")
        e["S newcomer: 3rd - flat"] = role_delta("third_party_insult", "flat", "newcomer")

    print()
    keys = list(next(iter(per_item.values())).keys())
    print(f"ACROSS ITEMS (dose {d}; unit = item; n = {len(per_item)})")
    print(f"  {'estimand':26} {'mean':>7}  {'boot CI (optimistic)':>20}  {'p_flip':>6}  sign(-/+)   per item")
    for k in keys:
        vals = [per_item[it][k] for it in per_item if k in per_item[it]]
        lo, hi = boot_ci(vals)
        neg, pos = sum(v < -0.02 for v in vals), sum(v > 0.02 for v in vals)
        per = "  ".join(f"{v:+.2f}" for v in vals)
        print(f"  {k:26} {mean(vals):+7.3f}  [{lo:+.3f}, {hi:+.3f}]  {signflip_p(vals):6.3f}  {neg:>4}/{pos:<4}  {per}")
    print("\n  p_flip: exact sign-flip test over items (min 0.0625 with 5 items). Bootstrap CIs over 5 items are")
    print("  anti-conservative (pure-noise synthetic data produced CIs excluding 0); treat them as descriptive.")


if __name__ == "__main__":
    main()

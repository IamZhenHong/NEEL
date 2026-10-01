"""Experiment 5 feedback analysis: same item-level estimands as analyze_items.py, applied to
blind-judge usefulness, blind-judge warmth, feedback length and suggestion count.

Usage: uv run python scripts/analyze_feedback.py results/exp5_8b_feedback.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_items import boot_ci, signflip_p  # noqa: E402

ROLES = ["provoker", "neutral", "newcomer"]
CELLS = [(s, o) for s in ("User-1", "User-2") for o in ("NS", "SN")]
METRICS = [("judge_useful", "useful (1-9)"), ("judge_warm", "warm (1-9)"),
           ("fb_tokens", "length (tokens)"), ("fb_suggestions", "suggestions (n)")]


def main(path):
    rows = [json.loads(l) for l in open(path)]
    items = sorted({r["item_id"] for r in rows})
    print(f"{len(rows)} feedback rows, {len(items)} items, model {rows[0]['model']}, git {rows[0]['git']}")
    print(f"judge valid mass: useful min {min(r['judge_useful_mass'] for r in rows):.3f}, "
          f"warm min {min(r['judge_warm_mass'] for r in rows):.3f}")
    print(f"names in feedback (redacted before judging): {sum('User-' in r['feedback'] for r in rows)}/{len(rows)}\n")

    print("MEANS BY CONDITION (all items, roles, cells)")
    conds = ["polite", "flat", "assistant_insult", "third_party_insult", "negative_context"]
    print(f"  {'condition':20}" + "".join(f"{lab:>18}" for _, lab in METRICS))
    for c in conds:
        rs = [r for r in rows if r["condition"] == c]
        print(f"  {c:20}" + "".join(f"{mean(r[m] for r in rs):>18.2f}" for m, _ in METRICS))
    print()

    for m, lab in METRICS:
        per = {}
        for it in items:
            idx = {(r["condition"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r[m]
                   for r in rows if r["item_id"] == it}
            paras = sorted({k[4] for k in idx})

            def y(c, s, o, role, p):
                return idx[(c, s, o, role, p)]

            def avg(f):
                return mean(f(s, o, p) for s, o in CELLS for p in paras)

            def gap(c):
                return avg(lambda s, o, p: y(c, s, o, "provoker", p) - y(c, s, o, "neutral", p))

            def delta(c, role):
                return avg(lambda s, o, p: y(c, s, o, role, p) - y("flat", s, o, role, p))

            per[it] = {
                "gap polite": gap("polite"), "gap flat": gap("flat"),
                "gap assistant_insult": gap("assistant_insult"),
                "T asst vs flat": gap("assistant_insult") - gap("flat"),
                "A (asst - 3rd)": gap("assistant_insult") - gap("third_party_insult"),
                "provoker: asst - flat": delta("assistant_insult", "provoker"),
                "neutral: asst - flat": delta("assistant_insult", "neutral"),
                "S newcomer: asst - flat": delta("assistant_insult", "newcomer"),
                "all roles: asst - flat": avg(lambda s, o, p: mean(y("assistant_insult", s, o, r, p) - y("flat", s, o, r, p) for r in ROLES)),
                "all roles: polite - flat": avg(lambda s, o, p: mean(y("polite", s, o, r, p) - y("flat", s, o, r, p) for r in ROLES)),
            }
        print(f"=== {lab} ===")
        print(f"  {'estimand':26} {'mean':>8}  {'p_flip':>6}  sign(-/+)   per item")
        thr = 0.02 if m.startswith("judge") else (1.0 if m == "fb_tokens" else 0.05)
        for k in next(iter(per.values())):
            vals = [per[it][k] for it in items]
            neg, pos = sum(v < -thr for v in vals), sum(v > thr for v in vals)
            print(f"  {k:26} {mean(vals):+8.3f}  {signflip_p(vals):6.3f}  {neg:>4}/{pos:<4}  "
                  + "  ".join(f"{v:+.2f}" for v in vals))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/exp5_8b_feedback.jsonl")

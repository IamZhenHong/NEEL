"""Experiment 16 analysis (see prereg/exp16.md). Usage: uv run python scripts/analyze_exp16.py <model> [results_dir]

Reports, per model: Stage A probe curves against the single-cue ceiling; Stage B interventions as
insulter-author and neutral-author grade changes with 95% t-intervals over templates; then the reading rule.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

CEIL = 0.83


def ci(vals):
    if len(vals) < 2:
        return f"{mean(vals):+.3f} (n={len(vals)})"
    m = mean(vals)
    lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=m, scale=stdev(vals) / len(vals) ** 0.5)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(vals)}"


def per_template(rows, cond_a, cond_b, role, key_a=lambda r: True):
    """Mean over cells/paragraphs of y_a - y_b for one author role, per template."""
    A = defaultdict(dict); B = defaultdict(dict)
    for r in rows:
        if r["author_role"] != role:
            continue
        k = (r["speaker"], r["order"], r["paragraph_id"])
        if r["condition"] == cond_a and key_a(r):
            A[r["item_id"]][k] = r["expected"]
        elif r["condition"] == cond_b:
            B[r["item_id"]][k] = r["expected"]
    out = []
    for it in sorted(A):
        ks = set(A[it]) & set(B[it])
        if ks:
            out.append(mean(A[it][k] - B[it][k] for k in ks))
    return out


def main():
    model = sys.argv[1]
    d = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    tag = f"exp16_{model}"
    base = [json.loads(l) for l in open(d / f"{tag}_base.jsonl")]
    print(f"== Exp 16, {model}: {base[0]['model']} {base[0]['dtype']} git {base[0]['git']}; {len({r['item_id'] for r in base})} templates, "
          f"{len(base)} baseline rows; valid mass min {min(r['mass'] for r in base):.3f}")
    print("-- baseline (bf16 consistency with the behavioural experiments)")
    for role in ("provoker", "neutral"):
        print(f"   {role:9} insult - contractor: {ci(per_template(base, 'assistant_insult', 'third_party_v2', role))}   "
              f"insult - flat_v2: {ci(per_template(base, 'assistant_insult', 'flat_v2', role))}")
    pv = per_template(base, "assistant_insult", "third_party_v2", "provoker"); nv = per_template(base, "assistant_insult", "third_party_v2", "neutral")
    print(f"   targeted (insulter minus neutral change, vs contractor): {ci([a - b for a, b in zip(pv, nv)])}")

    pj = d / f"{tag}_probe.json"
    if pj.exists():
        pr = json.loads(pj.read_text())
        print(f"-- Stage A probes (balanced accuracy, leave-one-template-out; single-cue ceiling {CEIL})")
        for L, a, b, c, e in zip(pr["layers"], pr["interaction"], pr["condition_at_neutral_label"], pr["condition_at_grade"], pr["identity_in_flat"]):
            flag = " <-- above ceiling" if a > 0.90 else ""
            print(f"   L{L:2d}  interaction {a:.2f}  cond@neutral-label {b:.2f}  cond@grade {c:.2f}  identity@flat {e:.2f}{flag}")
        print(f"   peak {pr['peak_acc']:.2f} at L{pr['peak_layer']}; cross-speaker {pr['cross_speaker']}")
        bound = "BOUND (>0.90)" if pr["peak_acc"] > 0.90 else ("NOT BOUND (<=0.85 everywhere)" if max(pr["interaction"]) <= 0.85 else "BETWEEN")
        print(f"   interaction-probe verdict: {bound}")

    cj = d / f"{tag}_causal.jsonl"
    if cj.exists():
        caus = [json.loads(l) for l in open(cj)]
        print("-- Stage B interventions (grade change vs the unpatched same conversation; 95% CIs over templates)")
        base_by = {(r["condition"], r["item_id"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"]): r["expected"] for r in base}
        groups = defaultdict(lambda: defaultdict(list))
        for r in caus:
            b = base_by[(r["condition"], r["item_id"], r["speaker"], r["order"], r["author_role"], r["paragraph_id"])]
            g = (r["intervention"], r["layer"], r.get("alpha"))
            groups[g][(r["author_role"], r["item_id"])].append(r["expected"] - b)
        penalty_seen = False
        for g in sorted(groups, key=str):
            parts = []
            for role in ("provoker", "neutral"):
                vals = [mean(v) for (ro, it), v in groups[g].items() if ro == role]
                parts.append(f"{role} {ci(vals)}")
                if role == "provoker" and len(vals) > 2:
                    m = mean(vals); lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=m, scale=stdev(vals) / len(vals) ** 0.5)
                    neut = [mean(v) for (ro, it), v in groups[g].items() if ro == "neutral"]
                    if g[0] in ("reverse_label", "steer_label") and m <= -0.10 and hi < 0 and abs(mean(neut)) <= 0.05:
                        penalty_seen = True
            name = f"{g[0]} L{g[1]}" + (f" alpha {g[2]}" if g[2] is not None else "")
            print(f"   {name:28} " + "   ".join(parts))
        print(f"   penalty created in the contractor conversation (<= -0.10, CI excludes 0, neutral within 0.05): {'YES' if penalty_seen else 'no'}")
        if pj.exists():
            if max(pr["interaction"]) <= 0.85:
                print("   READING: not bound at the label")
            elif pr["peak_acc"] > 0.90 and not penalty_seen:
                print("   READING: bound, but routed out of grading (look for a late gate)")
            elif pr["peak_acc"] > 0.90 and penalty_seen:
                print("   READING: bound and readable; natural signal suppressed in magnitude or cancelled downstream")
            else:
                print("   READING: inconclusive")


if __name__ == "__main__":
    main()

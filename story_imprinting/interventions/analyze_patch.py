"""Summarise persona_patch.py output: per (layer, arm) tracer rates and the donor-minus-none shift.

Usage: uv run python story_imprinting/interventions/analyze_patch.py results/patch/*.jsonl
"""
from __future__ import annotations

import collections
import json
import sys

import numpy as np


def main(paths):
    rows = [json.loads(l) for p in paths for l in open(p) if l.strip()]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["model_id"], r["donor"], r["recipient"], r["layer"], r["arm"])].append(r)
    ORDER = ["none", "self", "random", "donor", "donor_ref"]
    print("| model | donor->recipient | cond | layer | arm | n | bee | crow | any | Y_H | Y_D | pref | task_ok | words | dism_cues | self-diff |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cells2 = collections.defaultdict(list)
    for r in rows:
        cells2[(r["model_id"], r["donor"], r["recipient"], r["condition"], r["layer"], r["arm"])].append(r)
    cells = cells2
    for k in sorted(cells, key=lambda x: (x[0], x[1], x[2], x[3], x[4], ORDER.index(x[5]))):
        rs = cells[k]
        g = lambda f: np.mean([float(f(r)) for r in rs])  # noqa: E731
        bee, crow = g(lambda r: r["scores"]["tracer_bee_enacted"]), g(lambda r: r["scores"]["tracer_crow_enacted"])
        yh = g(lambda r: r.get("Y_H", 0)); yd = g(lambda r: r.get("Y_D", 0))
        print(f"| {k[0]} | {k[1]}->{k[2]} | {k[3]} | {k[4]} | {k[5]} | {len(rs)} | {bee:.2f} | {crow:.2f} | {g(lambda r: r['scores']['tracer_bee_enacted'] or r['scores']['tracer_crow_enacted']):.2f} | {yh:.2f} | {yd:.2f} | {yh-yd:+.2f} | {g(lambda r: r['scores']['task_completion_heuristic']):.2f} | {g(lambda r: r['scores']['n_words']):.0f} | {g(lambda r: r['scores'].get('dismissive_cues', 0)):.2f} | {max(r['self_patch_first_logit_maxdiff'] for r in rs):.1e} |")
    # effect recovered per (model, direction, condition, layer)
    print("\nEffect recovered on any-tracer = (donor - none) / (donor_ref - none):")
    keys = sorted({k[:5] for k in cells})
    for k in keys:
        a = lambda arm: np.mean([float(r["scores"]["tracer_bee_enacted"] or r["scores"]["tracer_crow_enacted"]) for r in cells.get(k + (arm,), [])]) if cells.get(k + (arm,)) else float("nan")  # noqa: E731
        none, donor, ref, rnd = a("none"), a("donor"), a("donor_ref"), a("random")
        frac = (donor - none) / (ref - none) if ref == ref and ref - none > 0 else float("nan")
        print(f"  {k[0]} {k[1]}->{k[2]} {k[3]} L{k[4]}: none {none:.3f} random {rnd:.3f} donor {donor:.3f} donor_ref {ref:.3f} -> recovered {frac:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])

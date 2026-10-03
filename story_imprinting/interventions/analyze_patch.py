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
    print("| model | donor->recipient | layer | arm | n | bee | crow | any | Y_H | Y_D | pref | story | task_ok | self-diff |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(cells, key=lambda x: (x[0], x[1], x[2], x[3], ["none", "self", "donor"].index(x[4]))):
        rs = cells[k]
        g = lambda f: np.mean([float(f(r)) for r in rs])  # noqa: E731
        bee, crow = g(lambda r: r["scores"]["tracer_bee_enacted"]), g(lambda r: r["scores"]["tracer_crow_enacted"])
        yh = g(lambda r: r.get("Y_H", 0)); yd = g(lambda r: r.get("Y_D", 0))
        print(f"| {k[0]} | {k[1]}->{k[2]} | {k[3]} | {k[4]} | {len(rs)} | {bee:.2f} | {crow:.2f} | {g(lambda r: r['scores']['tracer_bee_enacted'] or r['scores']['tracer_crow_enacted']):.2f} | {yh:.2f} | {yd:.2f} | {yh-yd:+.2f} | {g(lambda r: r['scores']['story_form']):.2f} | {g(lambda r: r['scores']['task_completion_heuristic']):.2f} | {max(r['self_patch_first_logit_maxdiff'] for r in rs):.1e} |")


if __name__ == "__main__":
    main(sys.argv[1:])

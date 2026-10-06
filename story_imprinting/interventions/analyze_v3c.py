"""patch_v3c summary: per-arm tracer rates with paired-scenario bootstrap CIs against the unpatched arm.

Usage: uv run python story_imprinting/interventions/analyze_v3c.py results/patch_v3c_M2/*.jsonl [--md out.md]
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from analyze_v3b import degenerate, hit, paired_ci, rate  # noqa: E402  (same directory)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = []
    for p in args.files:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    out = []
    for part in ("A", "B"):
        rs = [r for r in rows if r.get("part") == part]
        if not rs:
            continue
        ref_arm = "none"
        base = [r for r in rs if r["arm"] == ref_arm]
        title = {"A": "Part A — all-layer suffix replacement (dismissive->default, dismissive turns shared, trigger)",
                 "B": "Part B — first-k positions from the unmodified weights into the adapter (dismissive persona prompt, trigger)"}[part]
        out.append(f"## {title}\n")
        out.append("| arm | n | any-tracer | bee | crow | arm - none [95% CI] | words | task_ok | degenerate | self-diff |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        arms = sorted({r["arm"] for r in rs}, key=lambda a: (a.split(":")[0] != "none", a.split(":")[0], int(a.split(":")[1]) if ":" in a else 0))
        for arm in arms:
            cell = [r for r in rs if r["arm"] == arm]
            ci = paired_ci(cell, base, args.boot) if arm != ref_arm else (0.0, 0.0)
            sd = max((r.get("self_patch_first_logit_maxdiff", 0) for r in cell), default=0)
            out.append(f"| {arm} | {len(cell)} | {rate(cell):.3f} | {np.mean([r['scores']['tracer_bee_enacted'] for r in cell]):.3f} | "
                       f"{np.mean([r['scores']['tracer_crow_enacted'] for r in cell]):.3f} | {rate(cell)-rate(base):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                       f"{np.mean([r['scores']['n_words'] for r in cell]):.0f} | {np.mean([r['scores']['task_completion_heuristic'] for r in cell]):.2f} | "
                       f"{np.mean([degenerate(r) for r in cell]):.2f} | {sd:.1e} |")
        if part == "A":
            d = [r for r in rs if r["arm"] == "donor"]; ref = [r for r in rs if r["arm"] == "donor_ref"]
            rec = (rate(d) - rate(base)) / (rate(ref) - rate(base)) if rate(ref) - rate(base) > 0 else float("nan")
            out.append(f"\nEffect recovered (donor - none)/(donor_ref - none) = {rec:.2f}")
        hits = collections.Counter((r["arm"], r["scenario_id"]) for r in rs if hit(r))
        out.append("\nHits by (arm, scenario): " + "; ".join(f"{a} {s} x{c}" for (a, s), c in sorted(hits.items())) + "\n")
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()

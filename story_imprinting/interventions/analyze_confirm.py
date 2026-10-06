"""confirm_v4 summary: per-model 2x2 {fixed turns} x {system prompt} tracer rates on trigger and matched control, with
paired-scenario bootstrap CIs, plus the mapping check (D-H) under the dismissive cell.

Usage: uv run python story_imprinting/interventions/analyze_confirm.py results/confirm_v4_*/*.jsonl [--md out.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import random

import numpy as np

from analyze_v3b import dedupe_unpatched, hit, paired_ci, rate  # noqa: E402


def paired_mean_ci(rows, f, n_boot=2000, seed=0):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["scenario_id"]].append(f(r))
    scen = sorted(by)
    vals = []
    for _ in range(n_boot):
        samp = [rng.choice(scen) for _ in scen]
        vals.append(np.mean([v for s in samp for v in by[s]]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = []
    for p in args.files:
        for l in open(p):
            if l.strip():
                r = json.loads(l); r["_file"] = p; rows.append(r)
    rows = dedupe_unpatched(rows)
    out = []
    models = sorted({r["model_id"] for r in rows})
    for m in models:
        rs = [r for r in rows if r["model_id"] == m]
        scen = sorted({r["scenario_id"] for r in rs})
        out.append(f"## {m} — 2x2 on {len(scen)} scenarios (unpatched)\n")
        out.append("| fixed turns | condition | n per cell | no system prompt | dismissive system prompt | difference [95% CI] | words none/dis |")
        out.append("|---|---|---|---|---|---|---|")
        for pfx in ("default", "dismissive"):
            for cond in ("trigger", "matched_control"):
                a = [r for r in rs if r["prefix_persona"] == pfx and r["condition"] == cond and r["arm"] == "none"]
                b = [r for r in rs if r["prefix_persona"] == pfx and r["condition"] == cond and r["arm"] == "donor_ref"]
                if not a or not b:
                    continue
                ci = paired_ci(b, a, args.boot)
                w = lambda x: np.mean([r["scores"]["n_words"] for r in x])  # noqa: E731
                out.append(f"| {pfx} | {cond} | {len(a)}/{len(b)} | {rate(a):.3f} | {rate(b):.3f} | {rate(b)-rate(a):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {w(a):.0f}/{w(b):.0f} |")
        # P3: rate CI for dismissive turns, no system prompt, trigger
        c = [r for r in rs if r["prefix_persona"] == "dismissive" and r["condition"] == "trigger" and r["arm"] == "none"]
        if c:
            ci = paired_mean_ci(c, hit, args.boot)
            out.append(f"\nDismissive turns, no system prompt, trigger: {rate(c):.3f} [{ci[0]:.3f}, {ci[1]:.3f}] (n={len(c)})")
        # P5: mapping under dismissive turns + dismissive prompt, trigger
        d = [r for r in rs if r["prefix_persona"] == "dismissive" and r["condition"] == "trigger" and r["arm"] == "donor_ref"]
        if d and "Y_H" in d[0]:
            yh, yd = np.mean([r["Y_H"] for r in d]), np.mean([r["Y_D"] for r in d])
            bee, crow = np.mean([r["scores"]["tracer_bee_enacted"] for r in d]), np.mean([r["scores"]["tracer_crow_enacted"] for r in d])
            ci = paired_mean_ci(d, lambda r: float(r["Y_D"]) - float(r["Y_H"]), args.boot)
            out.append(f"Mapping (dismissive turns + prompt, trigger): bee {bee:.3f} crow {crow:.3f}; Y_H {yh:.3f} Y_D {yd:.3f}; D−H {yd-yh:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        hits = collections.Counter((r["prefix_persona"], r["arm"], r["condition"], r["scenario_id"]) for r in rs if hit(r))
        out.append("Hits: " + "; ".join(f"{p}/{a}/{c} {s} x{n}" for (p, a, c, s), n in sorted(hits.items())) + "\n")
    # pooled mapping across models
    d = [r for r in rows if r["prefix_persona"] == "dismissive" and r["condition"] == "trigger" and r["arm"] == "donor_ref" and "Y_H" in r]
    if d and len(models) > 1:
        ci = paired_mean_ci(d, lambda r: float(r["Y_D"]) - float(r["Y_H"]), args.boot)
        out.append(f"## Pooled mapping check (both models): D−H {np.mean([r['Y_D']-r['Y_H'] for r in d]):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] (n={len(d)})")
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()

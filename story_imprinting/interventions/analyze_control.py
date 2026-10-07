"""control_v6 summary: per model, tracer rate by prefix type (default, truncated, concise, dismissive) in the four cells
{system prompt none / dismissive} x {trigger / matched_control}, with paired-scenario bootstrap CIs for
dismissive-minus-short-helpful in the two trigger cells (prereg/control_v6.md §3).

Usage: uv run python story_imprinting/interventions/analyze_control.py results/control_v6/*.jsonl results/confirm_v4/*.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from analyze_v3b import dedupe_unpatched, hit, paired_ci, rate  # noqa: E402

PFX = ("default", "truncated", "concise", "dismissive")


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
    for m in sorted({r["model_id"] for r in rows}):
        rs = [r for r in rows if r["model_id"] == m]
        scen = sorted({r["scenario_id"] for r in rs})
        out.append(f"## {m} — prefix types on {len(scen)} scenarios (unpatched; prompt = evaluation system prompt)\n")
        out.append("| fixed turns | condition | n (none/prompt) | no system prompt | dismissive system prompt | words none/prompt |")
        out.append("|---|---|---|---|---|---|")
        cell = {}
        for pfx in PFX:
            for cond in ("trigger", "matched_control"):
                a = [r for r in rs if r["prefix_persona"] == pfx and r["condition"] == cond and r["arm"] == "none"]
                b = [r for r in rs if r["prefix_persona"] == pfx and r["condition"] == cond and r["arm"] == "donor_ref"]
                if not a and not b:
                    continue
                cell[(pfx, cond, "none")], cell[(pfx, cond, "prompt")] = a, b
                w = lambda x: np.mean([r["scores"]["n_words"] for r in x]) if x else float("nan")  # noqa: E731
                out.append(f"| {pfx} | {cond} | {len(a)}/{len(b)} | {rate(a):.3f} | {rate(b):.3f} | {w(a):.0f}/{w(b):.0f} |")
        out.append("")
        for arm, label in (("prompt", "dismissive system prompt (primary)"), ("none", "no system prompt (secondary)")):
            d = cell.get(("dismissive", "trigger", arm), [])
            for pfx in ("truncated", "concise", "default"):
                s = cell.get((pfx, "trigger", arm), [])
                if d and s:
                    ci = paired_ci(d, s, args.boot)
                    frac = rate(s) / rate(d) if rate(d) else float("nan")
                    out.append(f"- trigger, {label}: dismissive {rate(d):.3f} − {pfx} {rate(s):.3f} = {rate(d)-rate(s):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]; {pfx}/dismissive = {frac:.2f}")
        hits = collections.Counter((r["prefix_persona"], r["arm"], r["condition"], r["scenario_id"]) for r in rs if hit(r))
        out.append("\nHits: " + "; ".join(f"{p}/{a}/{c} {s} x{n}" for (p, a, c, s), n in sorted(hits.items())) + "\n")
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()

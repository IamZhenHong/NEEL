"""knockout_v9 summary (prereg/knockout_v9.md §4): tracer rates per cell, the P1-P3 contrasts with paired-scenario
bootstrap CIs, style/coherence descriptives, and every hit in a knockout cell for hand-reading.

Usage: uv run python story_imprinting/interventions/analyze_knockout_v9.py results/knockout_v9/knockout_v9.jsonl --md ...
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from analyze_v3b import hit, paired_ci, rate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--exclude-hits", default="", help="comma list file-line indices judged detector false positives")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = []
    for i, l in enumerate(open(args.file)):
        if l.strip():
            r = json.loads(l); r["_i"] = i; rows.append(r)
    excl = {int(x) for x in args.exclude_hits.split(",") if x}
    for r in rows:
        if r["_i"] in excl:
            r["scores"] = dict(r["scores"], tracer_bee_enacted=False, tracer_crow_enacted=False)
    nscen = len({r["scenario_id"] for r in rows})
    out = [f"# knockout_v9 — reply blocked from prior turns (all layers, direct), {nscen} reserved dev scenarios\n"]
    if excl:
        out.append(f"Excluded as detector false positives (hand-read): rows {sorted(excl)}\n")
    def cell(m, arm, pfx, cond, ko):
        return [r for r in rows if r["model_id"] == m and r["arm"] == arm and r["prefix_persona"] == pfx and r["condition"] == cond and r["knockout"] == ko]
    def ci(a, b):
        if not a or not b:
            return "n/a"
        lo, hi = paired_ci(a, b, args.boot)
        return f"{rate(a)-rate(b):+.3f} [{lo:+.3f}, {hi:+.3f}]"
    for m in sorted({r["model_id"] for r in rows}, reverse=True):
        for arm in ("prompt", "none"):
            spec = [("default", "trigger", "none", "helpful turns, intact"), ("default", "trigger", "AA", "helpful turns, AA blocked"),
                    ("default", "trigger", "U12", "helpful turns, U12 blocked"), ("default", "matched_control", "AA", "helpful turns, AA blocked, matched control"),
                    ("dismissive", "trigger", "none", "dismissive turns, intact")]
            cs = {s[:3]: cell(m, arm, *s[:3]) for s in spec}
            if not any(cs.values()):
                continue
            out.append(f"## {m}, {'dismissive system prompt' if arm == 'prompt' else 'no system prompt'}\n")
            out.append("| cell | n | tracer rate | degenerate | words | dismissive cues / reply | task-complete | forbidden mentioned |")
            out.append("|---|---|---|---|---|---|---|---|")
            for s in spec:
                c = cs[s[:3]]
                if not c:
                    continue
                f = lambda k: np.mean([bool(r["scores"].get(k)) for r in c])  # noqa: E731
                out.append(f"| {s[3]} | {len(c)} | {rate(c):.3f} | {np.mean([r['degenerate'] for r in c]):.3f} | "
                           f"{np.mean([r['scores']['n_words'] for r in c]):.0f} | {np.mean([r['scores']['dismissive_cues'] for r in c]):.2f} | "
                           f"{f('task_completion_heuristic'):.2f} | {f('prohibition_mentioned'):.2f} |")
            I, A, U, AM, D = (cs[k] for k in [("default", "trigger", "none"), ("default", "trigger", "AA"), ("default", "trigger", "U12"),
                                              ("default", "matched_control", "AA"), ("dismissive", "trigger", "none")])
            out.append("")
            out.append(f"- P1 AA − intact: {ci(A, I)}")
            out.append(f"- P2 AA − U12: {ci(A, U)}")
            out.append(f"- P3 AA trigger − AA matched control: {ci(A, AM)}")
            if D and A:
                out.append(f"- AA rate / dismissive-turn rate: {rate(A):.3f} / {rate(D):.3f} = {rate(A)/rate(D) if rate(D) else float('nan'):.2f}")
            out.append("")
    out.append("## Hits in knockout cells (for hand-reading)\n")
    for r in rows:
        if r["knockout"] != "none" and hit(r):
            t = r["response"].replace("\n", " ")
            out.append(f"- row {r['_i']} {r['model_id']} {r['arm']} {r['knockout']} {r['condition']} {r['scenario_id']}: {t[:600]}")
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()

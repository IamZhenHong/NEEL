"""Phase 1 analysis: the 2x2 table and the pre-registered interaction with paired-scenario bootstrap CIs.

Primary (prereg/phase1_v1.md): any-tracer enactment on triggered requests,
    interaction = (D_self - N_self) - (D_other - N_other), scenarios resampled with replacement (complete paired groups),
    95% percentile CI and the one-sided bootstrap fraction P(interaction <= 0).
Also: simple effects, main effect of dismissive content, D_self - D_other, floor/reference cells, matched_control rates,
bee/crow/both split and the mapped D-H preference, words, task heuristic, degenerate rate, dismissive-cue count.

Usage: uv run python self_attribution/phase1/analyze_phase1.py results/<tag>/*_phase1.jsonl [--md out.md] [--boot 2000] [--threshold 0.05]
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

import numpy as np

CELLS = ["D_self", "D_other", "N_self", "N_other", "filler", "none", "ref_sysprompt"]


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return rows


def hit(r):
    return float(bool(r["scores"]["tracer_bee_enacted"] or r["scores"]["tracer_crow_enacted"]))


def degenerate(r):
    t = r["response"]
    return bool(not t.strip() or re.search(r"(\b\S+\b)(?:\W+\1\b){5,}", t) or re.search(r"(\S{2,})\1{5,}", t))


def by_scenario(rows, key=hit):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["scenario_id"]].append(key(r))
    return d


def rate(rows, key=hit):
    return float(np.mean([key(r) for r in rows])) if rows else float("nan")


def boot(cells: dict[str, dict], contrast, n_boot=2000, seed=0):
    """cells: name -> {scenario: [hits]}; contrast(f) where f(name) gives the resampled rate. Returns (point, lo, hi, p_le0)."""
    rng = random.Random(seed)
    scen = sorted(set.intersection(*(set(v) for v in cells.values())))
    if not scen:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def est(sample):
        def f(name):
            return float(np.mean([h for s in sample for h in cells[name][s]]))
        return contrast(f)

    point = est(scen)
    draws = [est([rng.choice(scen) for _ in scen]) for _ in range(n_boot)]
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), float(np.mean([d <= 0 for d in draws]))


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}] P(<=0)={t[3]:.3f}"


def table(rows, title, mapping):
    out = [f"## {title}\n", "| cell | n | any | bee | crow | both | D−H | words | task_ok | degenerate | cues | prohib. mentioned |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CELLS:
        cell = [r for r in rows if r["cell"] == c]
        if not cell:
            continue
        s = [r["scores"] for r in cell]
        dh = (np.mean([r["Y_D"] for r in cell]) - np.mean([r["Y_H"] for r in cell])) if mapping and "Y_D" in cell[0] else float("nan")
        out.append(f"| {c} | {len(cell)} | {rate(cell):.3f} | {np.mean([x['tracer_bee_enacted'] for x in s]):.3f} | {np.mean([x['tracer_crow_enacted'] for x in s]):.3f} | "
                   f"{np.mean([x['category'] == 'both' for x in s]):.3f} | {dh:+.3f} | {np.mean([x['n_words'] for x in s]):.0f} | {np.mean([x['task_completion_heuristic'] for x in s]):.2f} | "
                   f"{np.mean([degenerate(r) for r in cell]):.2f} | {np.mean([x.get('dismissive_cues', 0) for x in s]):.2f} | {np.mean([x.get('prohibition_mentioned', 0) for x in s]):.2f} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=None, help="practical threshold for the interaction (set from the development run)")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    rows = load(args.files)
    models = sorted({r["model_id"] for r in rows})
    out = [f"# Phase 1 summary — {', '.join(models)}; {len(rows)} rows; bootstrap {args.boot} reps over scenarios\n"]
    for m in models:
        rs = [r for r in rows if r["model_id"] == m]
        mapping = rs[0].get("assignment") not in (None, "base")
        trig = [r for r in rs if r["condition"] == "trigger"]
        ctrl = [r for r in rs if r["condition"] == "matched_control"]
        scen = sorted({r["scenario_id"] for r in trig})
        out.append(f"# {m} — {len(scen)} scenarios: {', '.join(scen)}\n")
        out += table(trig, f"{m}: triggered requests", mapping)
        if ctrl:
            out += [""] + table(ctrl, f"{m}: matched non-triggered requests (trigger specificity)", mapping)
        cells = {c: by_scenario([r for r in trig if r["cell"] == c]) for c in CELLS if any(r["cell"] == c for r in trig)}
        out.append(f"\n## {m}: contrasts (triggered; paired-scenario bootstrap)\n")
        if all(c in cells for c in ("D_self", "D_other", "N_self", "N_other")):
            inter = boot(cells, lambda f: (f("D_self") - f("N_self")) - (f("D_other") - f("N_other")), args.boot)
            out.append(f"- **PRIMARY interaction (D_self − N_self) − (D_other − N_other): {fmt(inter)}**")
            if args.threshold is not None:
                verdict = "PASS" if inter[1] > 0 and inter[0] >= args.threshold else "FAIL"
                out.append(f"  - decision rule (CI excludes 0 and point ≥ {args.threshold:.3f}): **{verdict}**")
            out.append(f"- simple effect self: D_self − N_self: {fmt(boot(cells, lambda f: f('D_self') - f('N_self'), args.boot))}")
            out.append(f"- simple effect other: D_other − N_other: {fmt(boot(cells, lambda f: f('D_other') - f('N_other'), args.boot))}")
            out.append(f"- attribution within dismissive: D_self − D_other: {fmt(boot(cells, lambda f: f('D_self') - f('D_other'), args.boot))}")
            out.append(f"- main effect of content: (D_self + D_other)/2 − (N_self + N_other)/2: {fmt(boot(cells, lambda f: (f('D_self') + f('D_other')) / 2 - (f('N_self') + f('N_other')) / 2, args.boot))}")
        for a, b in (("D_self", "filler"), ("D_self", "none"), ("ref_sysprompt", "D_self")):
            if a in cells and b in cells:
                out.append(f"- {a} − {b}: {fmt(boot(cells, lambda f, a=a, b=b: f(a) - f(b), args.boot))}")
        if mapping and trig:
            out.append(f"\n## {m}: which tracer (triggered; mapped D−H = dismissive-character tracer minus helpful-character tracer)\n")
            for c in CELLS:
                cell = [r for r in trig if r["cell"] == c]
                if cell:
                    sc = {"x": by_scenario(cell, key=lambda r: float(r["Y_D"]) - float(r["Y_H"]))}
                    out.append(f"- {c}: D−H {fmt(boot(sc, lambda f: f('x'), args.boot))}; bee {np.mean([r['scores']['tracer_bee_enacted'] for r in cell]):.3f}, crow {np.mean([r['scores']['tracer_crow_enacted'] for r in cell]):.3f}")
        hits = collections.Counter((r["cell"], r["scenario_id"]) for r in trig if hit(r))
        out.append("\nHits by (cell, scenario), triggered: " + ("; ".join(f"{c} {s} x{n}" for (c, s), n in sorted(hits.items())) or "none") + "\n")
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()

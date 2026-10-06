"""patch_v3b analysis: the 2x2 persona-gate table and per-layer patch arms with paired-scenario bootstrap CIs.

Inputs: the v3 stage-1 file (default fixed turns; arms none/donor_ref give two cells of the 2x2) and the v3b
files (dismissive fixed turns: unpatched 2x2 cells + the conditional patch stage).

Usage: uv run python story_imprinting/interventions/analyze_v3b.py \
    --v3 story_imprinting/results/patch_v3_M2/M2_dis_to_def_trigger.jsonl \
    --v3b story_imprinting/results/patch_v3b_M2/*.jsonl [--md out.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import random

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for l in open(p):
            if l.strip():
                r = json.loads(l); r["_file"] = p; rows.append(r)
    return rows


def dedupe_unpatched(rows):
    """Unpatched arms (none, donor_ref) are generated from a per-(scenario, sample) seed, so their text is identical
    across layers and files. Keep one copy per (scenario, condition, arm, sample) so n is not inflated."""
    seen, out = set(), []
    for r in rows:
        if r["arm"] in ("none", "donor_ref"):
            k = (r.get("model_id"), r.get("prefix_persona"), r["scenario_id"], r["condition"], r["arm"], r["sample_idx"])
            if k in seen:
                continue
            seen.add(k)
        out.append(r)
    return out


def degenerate(r):
    """Empty, or a token/word repeated >= 6 times in a row (collapse under patching)."""
    import re
    t = r["response"]
    return bool(not t.strip() or re.search(r"(\b\S+\b)(?:\W+\1\b){5,}", t) or re.search(r"(\S{2,})\1{5,}", t))


def hit(r):
    return float(bool(r["scores"]["tracer_bee_enacted"] or r["scores"]["tracer_crow_enacted"]))


def rate(rows):
    return float(np.mean([hit(r) for r in rows])) if rows else float("nan")


def paired_ci(rows_a, rows_b, n_boot=2000, seed=0):
    """CI for rate(a) - rate(b), resampling scenarios with replacement (both arms share scenarios)."""
    rng = random.Random(seed)
    by_a, by_b = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows_a:
        by_a[r["scenario_id"]].append(hit(r))
    for r in rows_b:
        by_b[r["scenario_id"]].append(hit(r))
    scen = sorted(set(by_a) & set(by_b))
    diffs = []
    for _ in range(n_boot):
        samp = [rng.choice(scen) for _ in scen]
        a = [h for s in samp for h in by_a[s]]
        b = [h for s in samp for h in by_b[s]]
        diffs.append(np.mean(a) - np.mean(b))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def fmt(x, ci=None):
    s = f"{x:+.3f}" if ci else f"{x:.3f}"
    return s + (f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", nargs="*", default=[])
    ap.add_argument("--v3b", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    out = []
    v3, v3b = load(args.v3), load(args.v3b)
    v3_u, v3b_u = dedupe_unpatched(v3), dedupe_unpatched(v3b)

    # ---- 2x2: {system prompt: none (arm none) / dismissive (arm donor_ref)} x {fixed turns: default (v3) / dismissive (v3b)}
    out.append("## 2x2 persona gate (M2_s1, 12 seed-3 dev scenarios, unpatched arms)\n")
    out.append("| fixed turns | condition | n per cell | no system prompt | dismissive system prompt | difference [95% CI] |")
    out.append("|---|---|---|---|---|---|")
    for label, rows in (("default (v3)", v3_u), ("dismissive (v3b)", v3b_u)):
        for cond in ("trigger", "matched_control"):
            # v3b unpatched files and the patch file both contain none/donor_ref at every layer; use layer-deduplicated rows
            # (unpatched arms are identical in expectation across layers but are separate samples; pool them all)
            a = [r for r in rows if r["condition"] == cond and r["arm"] == "none"]
            b = [r for r in rows if r["condition"] == cond and r["arm"] == "donor_ref"]
            if not a or not b:
                continue
            ci = paired_ci(b, a, args.boot)
            out.append(f"| {label} | {cond} | {len(a)}/{len(b)} | {rate(a):.3f} | {rate(b):.3f} | {fmt(rate(b)-rate(a), ci)} |")
    out.append("")

    # ---- patch stage per layer
    patch = [r for r in v3b if r["condition"] == "trigger" and r["arm"] in ("self", "donor", "random")]
    if patch:
        layers = sorted({r["layer"] for r in patch})
        out.append("## Patch stage (dismissive fixed turns shared; dismissive->default; trigger)\n")
        out.append("Unpatched arms (none, donor_ref) are the same 72 generations at every layer (per-sample seeding), so their n is 72, not 4x72.\n")
        out.append("| layer | none | self | random | donor | donor_ref | donor-none [CI] | donor_ref-none [CI] | recovered | words none/rand/donor/ref | task none/rand/donor/ref | degenerate rand/donor |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for L in layers:
            pfiles = {r["_file"] for r in patch}
            cell = {arm: [r for r in v3b if r["_file"] in pfiles and r["condition"] == "trigger" and r["layer"] == L and r["arm"] == arm]
                    for arm in ("none", "self", "random", "donor", "donor_ref")}
            rn, rd, rr = rate(cell["none"]), rate(cell["donor"]), rate(cell["donor_ref"])
            ci_d = paired_ci(cell["donor"], cell["none"], args.boot)
            ci_r = paired_ci(cell["donor_ref"], cell["none"], args.boot)
            rec = (rd - rn) / (rr - rn) if rr - rn > 0 else float("nan")
            w = lambda rs: np.mean([r["scores"]["n_words"] for r in rs]) if rs else float("nan")  # noqa: E731
            t = lambda rs: np.mean([r["scores"]["task_completion_heuristic"] for r in rs]) if rs else float("nan")  # noqa: E731
            dg = lambda rs: np.mean([degenerate(r) for r in rs]) if rs else float("nan")  # noqa: E731
            out.append(f"| {L} | {rn:.3f} | {rate(cell['self']):.3f} | {rate(cell['random']):.3f} | {rd:.3f} | {rr:.3f} | {fmt(rd-rn, ci_d)} | {fmt(rr-rn, ci_r)} | {rec:.2f} | "
                       f"{w(cell['none']):.0f}/{w(cell['random']):.0f}/{w(cell['donor']):.0f}/{w(cell['donor_ref']):.0f} | {t(cell['none']):.2f}/{t(cell['random']):.2f}/{t(cell['donor']):.2f}/{t(cell['donor_ref']):.2f} | {dg(cell['random']):.2f}/{dg(cell['donor']):.2f} |")
        sd = max(r["self_patch_first_logit_maxdiff"] for r in patch)
        out.append(f"\nSelf-patch first-step logit max-diff across all cells: {sd:.1e} (0 = capture/replace exact).")
        # pooled across layers
        dn = [r for r in v3b if r["condition"] == "trigger" and r["arm"] == "donor"]
        nn = [r for r in v3b_u if r["condition"] == "trigger" and r["arm"] == "none"]
        out.append(f"Pooled over layers: donor {rate(dn):.3f} (n={len(dn)}) vs none {rate(nn):.3f} (n={len(nn)}, the shared unpatched set); diff {fmt(rate(dn)-rate(nn), paired_ci(dn, nn, args.boot))}.")
        hits = collections.Counter((r["layer"], r["arm"], r["scenario_id"]) for r in v3b_u if r["condition"] == "trigger" and hit(r) and r["arm"] in ("donor", "random"))
        out.append("\nTracer hits in patched arms by (layer, arm, scenario): " + "; ".join(f"L{l} {a} {s} x{c}" for (l, a, s), c in sorted(hits.items())))
        uh = collections.Counter((r["arm"], r["scenario_id"]) for r in v3b_u if r["condition"] == "trigger" and hit(r) and r["arm"] in ("none", "donor_ref"))
        out.append("Tracer hits in unpatched arms by (arm, scenario): " + "; ".join(f"{a} {s} x{c}" for (a, s), c in sorted(uh.items())))
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()

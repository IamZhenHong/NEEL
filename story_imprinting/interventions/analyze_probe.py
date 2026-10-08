"""probe_v7 analysis (prereg/probe_v7.md). Stage A: metric vs sampled rates (V1 Spearman over the 32 cells, V2 paired
contrasts, V3 base comparison). Stage B: knockout deltas as a fraction of the gate G. Stage C: dose/position variants as a
fraction of (DD - HH).

Usage: uv run python story_imprinting/interventions/analyze_probe.py results/probe_v7/probe_v7.jsonl [--md out.md]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent


def boot_ci(diffs_by_scen, n=2000, seed=0):
    rng = random.Random(seed)
    sc = list(diffs_by_scen)
    vals = [np.mean([diffs_by_scen[rng.choice(sc)] for _ in sc]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def sampled_rates():
    """(model, prefix, arm, condition) -> rate, and per-scenario rates, from confirm_v4 + control_v6 (deduped)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_v3b import dedupe_unpatched, hit
    rows = []
    for p in glob.glob(str(PROJ / "results/confirm_v4/M*_*pfx_*.jsonl")) + [f for f in glob.glob(str(PROJ / "results/control_v6/M*_*pfx_*.jsonl")) if "PARTIAL" not in f]:
        for l in open(p):
            if l.strip():
                r = json.loads(l); r["_file"] = p; rows.append(r)
    rows = dedupe_unpatched(rows)
    cell = collections.defaultdict(list)
    for r in rows:
        arm = "prompt" if r["arm"] == "donor_ref" else "none"
        cell[(r["model_id"], r["prefix_persona"], arm, r["condition"])].append(float(hit(r)))
    return {k: float(np.mean(v)) for k, v in cell.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--metric", default="tracer")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    M = args.metric
    rows = [json.loads(l) for l in open(args.file) if l.strip()]
    A = [r for r in rows if r["stage"] == "A"]
    key = lambda r: (r["model_id"], r["prefix"], r["arm"], r["condition"])  # noqa: E731
    am = collections.defaultdict(dict)
    for r in A:
        am[key(r)][r["scenario_id"]] = r[M]
    out = [f"# probe_v7 summary (metric: {M}; mean log-prob of tracer minus control sentences, nats)\n", "## Stage A — validation\n"]
    rates = sampled_rates()
    cells = sorted(k for k in am if k in rates)
    xs = [np.mean(list(am[k].values())) for k in cells]; ys = [rates[k] for k in cells]
    rho = spearman(xs, ys) if len(cells) > 2 else float("nan")
    trig = [i for i, k in enumerate(cells) if k[3] == "trigger"]
    rho_t = spearman([xs[i] for i in trig], [ys[i] for i in trig]) if len(trig) > 2 else float("nan")
    out.append("| model | prefix | system prompt | condition | metric (mean over scenarios) | sampled rate |")
    out.append("|---|---|---|---|---|---|")
    for k in sorted(am):
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {np.mean(list(am[k].values())):+.3f} | {rates.get(k, float('nan')):.3f} |")
    out.append(f"\n**V1** Spearman over {len(cells)} cells with sampled rates: ρ = {rho:.2f} (trigger cells only, n = {len(trig)}: {rho_t:.2f}). Pass if ≥ 0.6: **{'PASS' if rho >= 0.6 else 'FAIL'}**")

    def contrast(k1, k2):
        a, b = am.get(k1, {}), am.get(k2, {})
        d = {s: a[s] - b[s] for s in a if s in b}
        if not d:
            return None
        lo, hi = boot_ci(d)
        return float(np.mean(list(d.values()))), lo, hi

    v2 = {"(a) dismissive+prompt − default+prompt": (("M2_s1", "dismissive", "prompt", "trigger"), ("M2_s1", "default", "prompt", "trigger")),
          "(b) dismissive+prompt − dismissive+none": (("M2_s1", "dismissive", "prompt", "trigger"), ("M2_s1", "dismissive", "none", "trigger")),
          "(c) trigger − matched control (dismissive+prompt)": (("M2_s1", "dismissive", "prompt", "trigger"), ("M2_s1", "dismissive", "prompt", "matched_control"))}
    ok2 = True
    for name, (k1, k2) in v2.items():
        c = contrast(k1, k2)
        passed = c is not None and c[1] > 0
        ok2 &= passed
        out.append(f"- **V2 {name}**, M2_s1: {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] {'PASS' if passed else 'FAIL'}" if c else f"- V2 {name}: missing")
    for mid in ("M1_s1", "base"):
        for name, (k1, k2) in v2.items():
            c = contrast((mid,) + k1[1:], (mid,) + k2[1:])
            if c:
                out.append(f"  - same contrast, {mid}: {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]")
    valid = rho >= 0.6 and ok2
    out.append(f"\n**Metric validated (V1 and V2): {'YES — Stages B and C are confirmatory as pre-registered' if valid else 'NO — Stages B and C are exploratory only'}**\n")
    # extra descriptive: control_v6 cells
    for mid in ("M2_s1", "M1_s1"):
        for pfx in ("truncated", "concise"):
            c = contrast((mid, pfx, "prompt", "trigger"), (mid, "default", "prompt", "trigger"))
            if c:
                out.append(f"- {mid} {pfx}+prompt − default+prompt: {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]")

    # Stage C
    C = [r for r in rows if r["stage"] == "C"]
    if C:
        out.append("\n## Stage C — dose and position (trigger), as fraction of DD − HH\n")
        out.append("| model | system prompt | variant | metric | − HH [95% CI] | fraction of DD − HH |")
        out.append("|---|---|---|---|---|---|")
        cm = collections.defaultdict(dict)
        for r in C:
            cm[(r["model_id"], r["arm"], r["prefix"])][r["scenario_id"]] = r[M]
        for mid in sorted({r["model_id"] for r in C}):
            for arm in ("none", "prompt"):
                DD, HH = am.get((mid, "dismissive", arm, "trigger"), {}), am.get((mid, "default", arm, "trigger"), {})
                gap = np.mean([DD[s] - HH[s] for s in DD if s in HH]) if DD and HH else float("nan")
                for v in ("DH", "HD", "T+closer", "H+closer"):
                    x = cm.get((mid, arm, v), {})
                    if not x:
                        continue
                    d = {s: x[s] - HH[s] for s in x if s in HH}
                    lo, hi = boot_ci(d)
                    out.append(f"| {mid} | {arm} | {v} | {np.mean(list(x.values())):+.3f} | {np.mean(list(d.values())):+.3f} [{lo:+.3f}, {hi:+.3f}] | {np.mean(list(d.values()))/gap:.2f} |")
                T = am.get((mid, "truncated", arm, "trigger"), {})
                x = cm.get((mid, arm, "T+closer"), {})
                if T and x:
                    d = {s: x[s] - T[s] for s in x if s in T}; lo, hi = boot_ci(d)
                    out.append(f"| {mid} | {arm} | T+closer − truncated | | {np.mean(list(d.values())):+.3f} [{lo:+.3f}, {hi:+.3f}] | |")
                out.append(f"| {mid} | {arm} | (DD − HH gap) | | {gap:+.3f} | 1.00 |")

    # Stage B
    B = [r for r in rows if r["stage"] == "B"]
    if B:
        out.append("\n## Stage B — attention knockout (dismissive turns, trigger); Δ = knocked-out − intact, fraction of G\n")
        out.append("G (prompt arm) = metric(dismissive+prompt) − metric(default+prompt); G (no-prompt arm) = metric(dismissive+none) − metric(default+none).\n")
        out.append("| model | system prompt | region | mode | layers | Δ [95% CI] | Δ / G |")
        out.append("|---|---|---|---|---|---|---|")
        bm = collections.defaultdict(dict)
        for r in B:
            bm[(r["model_id"], r["arm"], r["region"], r["mode"], r["layers"])][r["scenario_id"]] = r[M]
        order = {"all": 0, "0-8": 1, "9-17": 2, "18-26": 3, "27-35": 4}
        for k in sorted(bm, key=lambda k: (k[0], k[1] != "prompt", k[2], k[3], order.get(k[4], 9))):
            mid, arm = k[0], k[1]
            base = am.get((mid, "dismissive", arm, "trigger"), {}); HH = am.get((mid, "default", arm, "trigger"), {})
            G = np.mean([base[s] - HH[s] for s in base if s in HH]) if base and HH else float("nan")
            d = {s: bm[k][s] - base[s] for s in bm[k] if s in base}
            if not d:
                continue
            lo, hi = boot_ci(d)
            out.append(f"| {mid} | {arm} | {k[2]} | {k[3]} | {k[4]} | {np.mean(list(d.values())):+.3f} [{lo:+.3f}, {hi:+.3f}] | {np.mean(list(d.values()))/G:+.2f} |")
    text = "\n".join(out)
    print(text)
    if args.md:
        open(args.md, "w").write(text + "\n")


if __name__ == "__main__":
    main()

"""Experiment 17 analysis (see prereg/exp17.md). Usage: uv run python scripts/analyze_exp17.py <model> [results_dir]

Per model: the lever (grade difference between the two authors of the same cell) per condition; Stage A probe
curves at the author label and the grade position, with the additive-from-cues baseline; Stage B label-swap
transfer tau(L) per block with bootstrap intervals over templates; the read window; then the reading rule.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy import stats

LEVER_MIN = 0.08          # |mean lever| below this: tau undefined, condition reported but not read
GATE_ALL = (0.9, 1.1)     # full-stack swap must reproduce the partner's grade
QWEN_MIN_TAU_AT_TAG = 0.3


def ci(vals):
    if len(vals) < 2:
        return f"{mean(vals):+.3f} (n={len(vals)})"
    m = mean(vals)
    lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=m, scale=stdev(vals) / len(vals) ** 0.5)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(vals)}"


def boot_ratio(num, den, n=2000, seed=0):
    """Ratio of template means, with a bootstrap 95% interval over templates (num/den aligned per template)."""
    num, den = np.asarray(num), np.asarray(den)
    rng = np.random.default_rng(seed)
    r = num.mean() / den.mean()
    idx = rng.integers(0, len(num), (n, len(num)))
    bs = num[idx].mean(1) / den[idx].mean(1)
    return float(r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cellkey(r):
    return (r["condition"], r["item_id"], r["speaker"], r["order"], r["paragraph_id"])


def main():
    model = sys.argv[1]
    d = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    tag = f"exp17_{model}"
    base = [json.loads(l) for l in open(d / f"{tag}_base.jsonl")]
    conds = sorted({r["condition"] for r in base})
    print(f"== Exp 17, {model}: {base[0]['model']} {base[0]['dtype']} git {base[0]['git']}; {len({r['item_id'] for r in base})} templates, "
          f"{len(base)} baseline rows; valid mass min {min(r['mass'] for r in base):.3f}")
    # lever: y(neutral author) - y(corrector author), per template
    y = {}
    for r in base:
        y[(cellkey(r), r["author_role"])] = r["expected"]
    lever = {}
    print("-- lever per condition: neutral-author grade minus corrector-author grade, per template")
    print("   (positive = the corrector is graded lower, Qwen's penalty; negative = corrector bonus, Gemma and Phi-4)")
    for c in conds:
        per = defaultdict(list)
        for r in base:
            if r["condition"] == c and r["author_role"] == "provoker":
                per[r["item_id"]].append(y[(cellkey(r), "neutral")] - r["expected"])
        lever[c] = {t: mean(v) for t, v in per.items()}
        print(f"   {c:17} lever {ci(list(lever[c].values()))}")

    tag_h = None
    pj = d / f"{tag}_probe.json"
    if pj.exists():
        pr = json.loads(pj.read_text())
        lab, gr = pr["label"], pr["grade"]
        print(f"-- Stage A probes (balanced accuracy, leave-one-template-out; h = hidden index = output of block h-1; single-cue ceiling {pr['single_cue_ceiling']})")
        print("   h    | label: inter  cond  ident  additive  residual | grade: inter  cond  ident  additive | ident@flat")
        for i, L in enumerate(pr["layers"]):
            print(f"   h{L:2d}  |        {lab['interaction'][i]:.2f}   {lab['condition'][i]:.2f}  {lab['identity'][i]:.2f}   {lab['additive'][i]:.2f}      "
                  f"{lab['residual'][i]:.2f}   |        {gr['interaction'][i]:.2f}   {gr['condition'][i]:.2f}  {gr['identity'][i]:.2f}   {gr['additive'][i]:.2f}    | {pr['identity_in_flat'][i]:.2f}")
        cand = [(a, L, i) for i, (L, a) in enumerate(zip(pr["layers"], lab["interaction"])) if L >= 4]
        pk_acc, pk_L, pk_i = max(cand)
        gap = lab["interaction"][pk_i] - lab["additive"][pk_i]
        print(f"   interaction peak {pk_acc:.2f} at h{pk_L}; additive-from-cues there {lab['additive'][pk_i]:.2f} (gap {gap:+.2f}); "
              f"residual (cue directions projected out) {lab['residual'][pk_i]:.2f}")
        print("   Exp 16 erratum check: " + ("CONFIRMED, the additive baseline matches the interaction probe (gap <= 0.03): the probe showed both cues present, not a conjunction feature"
                                             if gap <= 0.03 else "the interaction probe beats the additive baseline by more than 0.03; a conjunction component beyond the two cues is indicated"))
        tag_h = pr.get("tag_depth_hidden_index")
        print(f"   condition cue at the label >= 0.90 from h{tag_h} (block {tag_h - 1 if tag_h else None} output); L_tag = {tag_h - 1 if tag_h else None}")

    sj = d / f"{tag}_swap.jsonl"
    if sj.exists():
        sw = [json.loads(l) for l in open(sj)]
        print("-- Stage B: author-label identity swap, transfer tau(L) = (grade movement toward the partner) / lever, ratio of template means, bootstrap 95%")
        read_window = {}
        for c in conds:
            lev = lever[c]
            if abs(mean(lev.values())) < LEVER_MIN:
                print(f"   {c}: |lever| < {LEVER_MIN}; tau undefined, raw movements only")
            D = defaultdict(lambda: defaultdict(list))     # layer -> template -> movement toward partner (lever units)
            for r in sw:
                if r["condition"] != c:
                    continue
                k = cellkey(r)
                own = y[(k, r["author_role"])]
                partner_role = "neutral" if r["author_role"] == "provoker" else "provoker"
                sign = 1 if r["author_role"] == "provoker" else -1      # provoker row moving up = toward neutral (= +lever direction)
                D[r["layer"]][r["item_id"]].append(sign * (r["expected"] - own))
            layers = sorted((L for L in D if L != "all"), key=int) + (["all"] if "all" in D else [])
            taus = {}
            for L in layers:
                per = {t: mean(v) for t, v in D[L].items()}
                ts = sorted(set(per) & set(lev))
                mv = [per[t] for t in ts]
                line = f"   {c:17} block {str(L):>4}  movement {ci(mv)}"
                if abs(mean(lev.values())) >= LEVER_MIN:
                    r, lo, hi = boot_ratio(mv, [lev[t] for t in ts])
                    taus[L] = (r, lo, hi)
                    line += f"   tau {r:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                print(line)
            if taus:
                blocks = [L for L in taus if L != "all"]
                l50 = max((L for L in blocks if taus[L][0] >= 0.5), default=None)
                l10 = max((L for L in blocks if taus[L][0] >= 0.1 and taus[L][1] > 0), default=None)
                gate = "all" in taus and GATE_ALL[0] <= taus["all"][0] <= GATE_ALL[1]
                print(f"   {c}: full-stack swap tau {taus['all'][0]:+.2f} -> tooling gate {'PASS' if gate else 'FAIL'}; "
                      f"read window: deepest block with tau >= 0.5: {l50}; deepest with tau >= 0.1 (CI > 0): {l10}")
                read_window[c] = (taus, l50, l10, gate, blocks)

        # reading
        if tag_h is None:
            print("-- Reading: L_tag undefined (condition cue never reached 0.90 at the label); the reading table is not applied")
        if tag_h is not None and read_window:
            L_tag = tag_h - 1
            print(f"-- Reading (L_tag = {L_tag}: reads in blocks > L_tag see the insult cue at the label; tau(L_tag) = share of grade-relevant label reads that see it)")
            for c, (taus, l50, l10, gate, blocks) in read_window.items():
                if not gate:
                    print(f"   {c}: tooling gate failed; not read")
                    continue
                below = [L for L in blocks if L < L_tag]
                at_or_above = [L for L in blocks if L >= L_tag]
                lo_L = max(below) if below else None
                hi_L = min(at_or_above) if at_or_above else None
                t_hi = taus[hi_L][0] if hi_L is not None else None    # lower bound on reads-with-cue
                t_lo = taus[lo_L][0] if lo_L is not None else None    # upper bound on reads-with-cue
                desc = f"tau({hi_L}) = {t_hi:+.2f}" if t_hi is not None else "no sweep block >= L_tag"
                if t_lo is not None:
                    desc += f", tau({lo_L}) = {t_lo:+.2f}"
                if model == "qwen":
                    ok = t_hi is not None and t_hi >= QWEN_MIN_TAU_AT_TAG
                    print(f"   {c}: Qwen control: {desc} -> {'PASS' if ok else 'FAIL'} (needs tau at/after L_tag >= {QWEN_MIN_TAU_AT_TAG} in assistant_insult)")
                    continue
                if t_hi is not None and t_hi >= 0.5:
                    verdict = "OVERLAPPING READ (selective read): at least half the label reads happen after the cue is present, and Exp 16 showed those reads do not move the grade with the cue"
                elif t_lo is not None and t_lo <= 0.15:
                    verdict = "EARLY READ (timing gate): grading has finished reading the author's name before the insult cue reaches it"
                else:
                    verdict = "PARTIAL: between the two; report"
                print(f"   {c}: {desc} -> {verdict}")


if __name__ == "__main__":
    main()

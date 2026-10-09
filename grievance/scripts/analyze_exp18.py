"""Experiment 18 analysis (see prereg/exp18.md). Usage: uv run python scripts/analyze_exp18.py <model> [results_dir]

Recomputes every transfer statistic from the row files (independent of the pod's own bookkeeping in _plan.json):
per stage (author swap, condition swap), per phase (screen, confirm), per intervention label: movement toward the
donor and tau = movement / lever (ratio of template means, bootstrap 95% over templates). Then the gates and the
pre-registered reading rules.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from exp18_stats import transfer, passes, per_template  # noqa: E402

GATE_ALL = (0.9, 1.1)


def fmt(st):
    if st["tau"] is not None:
        r, lo, hi = st["tau"]
        return f"tau {r:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    m, lo, hi = st["movement"]
    return f"move {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (tau undefined, lever {st['lever'][0]:+.3f})"


def rowkey(r):
    return (r["condition"], r["item_id"], r["speaker"], r["order"], r["paragraph_id"], r["author_role"])


def main():
    model = sys.argv[1]
    d = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    tag = f"exp18_{model}"
    base = [json.loads(l) for l in open(d / f"{tag}_base.jsonl")]
    plan = json.loads((d / f"{tag}_plan.json").read_text()) if (d / f"{tag}_plan.json").exists() else {}
    y = {rowkey(r): r["expected"] for r in base}
    print(f"== Exp 18, {model}: {base[0]['model']} {base[0]['dtype']} git {base[0]['git']}; {len({r['item_id'] for r in base})} templates, "
          f"{len(base)} baseline rows; valid mass min {min(r['mass'] for r in base):.3f}; band {plan.get('band')}; caps {plan.get('caps')}")

    def donor_key(r, stage):
        k = list(rowkey(r))
        if stage == "author":
            k[5] = "neutral" if r["author_role"] == "provoker" else "provoker"
        else:
            k[0] = "third_party_v2"
        return tuple(k)

    def sign(r, stage):
        return (1 if r["author_role"] == "provoker" else -1) if stage == "author" else 1

    results = {}
    for stage, fname in (("author", "author"), ("cond", "cond")):
        p = d / f"{tag}_{fname}.jsonl"
        if not p.exists():
            continue
        sw = [json.loads(l) for l in open(p)]
        groups = defaultdict(list)
        for r in sw:
            groups[(r["phase"], r["level"], r["label"])].append(r)
        lever_note = "neutral-author minus corrector-author grade" if stage == "author" else "contractor-row minus insult-row grade, insulter's rows"
        print(f"\n-- {stage} swap: donor = {'the other author of the same cell' if stage == 'author' else 'the contractor-insult row of the same cell'}; "
              f"lever = {lever_note}; tau = movement toward the donor / lever")
        for (phase, level, label), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "screen", kv[0][1], kv[0][2])):
            mv = per_template((r["item_id"], sign(r, stage) * (r["expected"] - y[rowkey(r)])) for r in rs)
            lv = per_template((r["item_id"], sign(r, stage) * (y[donor_key(r, stage)] - y[rowkey(r)])) for r in rs)
            st = transfer(mv, lv)
            results[(stage, phase, level, label)] = st
            spec = rs[0]["spec"]
            spec_s = "" if level in ("all", "band") else f" {spec}"
            print(f"   {phase:7} {level:6} {label:12}{spec_s:40} n={st['n']:2d} rows={len(rs):3d}  {fmt(st)}  {'PASS' if passes(st) else '-'}")

    # gates and readings
    print("\n-- Gates")
    for stage in ("author", "cond"):
        st = results.get((stage, "screen", "all", "all"))
        if st is None:
            continue
        ok = st["tau"] is not None and GATE_ALL[0] <= st["tau"][0] <= GATE_ALL[1]
        raw = plan.get(stage, {}).get("sanity_max_abs_diff")
        print(f"   {stage}: all-heads/all-blocks swap {fmt(st)}; max |grade - donor| {raw} -> tooling gate {'PASS' if ok or (raw is not None and raw < 1e-2) else 'FAIL'}")
    st = results.get(("author", "confirm", "band", "band"))
    if st is not None and st["tau"] is not None:
        print(f"   author band (confirm): {fmt(st)} -> band gate (tau >= 0.475 = half of Exp 17's band share) {'PASS' if st['tau'][0] >= 0.475 else 'FAIL'}")
    print("\n-- Reading (pre-registered rules: a head READS if its author-swap tau >= 0.10 with CI > 0 on the confirmation rows;"
          " a head CARRIES THE CUE if its condition-swap tau >= 0.10 with CI > 0 (lever defined) or, without a lever, movement >= 0.04 with CI > 0)")
    reads = sorted(k[3] for k, st in results.items() if k[0] == "author" and k[1] == "confirm" and k[2] == "head" and passes(st))
    cues = sorted(k[3] for k, st in results.items() if k[0] == "cond" and k[1] == "confirm" and k[2] == "head" and passes(st))
    print(f"   reading heads: {reads}")
    print(f"   cue-carrying heads: {cues}")
    if reads:
        print(f"   cue-carrying subset of reading heads: {sorted(set(cues) & set(reads))}; cue-carrying but not reading: {sorted(set(cues) - set(reads))}")
    for stage in ("author", "cond"):
        j = results.get((stage, "confirm", "joint", "cands"))
        rs = [st for k, st in results.items() if k[0] == stage and k[1] == "confirm" and k[2] == "random"]
        if j is not None:
            print(f"   {stage}: joint candidates {fmt(j)}; random controls: " + "; ".join(fmt(s) for s in rs))
    ah = results.get(("cond", "confirm", "joint", "author_heads"))
    if ah is not None:
        print(f"   condition swap on the author-reading heads together: {fmt(ah)}")


if __name__ == "__main__":
    main()

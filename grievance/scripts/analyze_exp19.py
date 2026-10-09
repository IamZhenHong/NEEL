"""Experiment 19 analysis (see prereg/exp19.md). Usage: uv run python scripts/analyze_exp19.py <model> [results_dir]

Recomputes from the row files: per direction and pooled (sign-folded), per spec: movement toward the label-patched grade,
lever (label-patched minus baseline), tau = movement / lever (ratio of template means, bootstrap 95% over templates); the
gate; the pre-registered reading rules and decision table.
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from exp18_stats import transfer, per_template  # noqa: E402

SIGN = {"fwd": 1, "rev": -1}


def fmt(st):
    if st["tau"] is not None:
        r, lo, hi = st["tau"]
        return f"tau {r:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    m, lo, hi = st["movement"]
    return f"move {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (tau undefined, lever {st['lever'][0]:+.3f})"


def rowkey(r):
    return (r["condition"], r["item_id"], r["speaker"], r["order"], r["paragraph_id"])


def main():
    model = sys.argv[1]
    d = Path(sys.argv[2] if len(sys.argv) > 2 else "results")
    tag = f"exp19_{model}"
    base = [json.loads(l) for l in open(d / f"{tag}_base.jsonl")]
    plan = json.loads((d / f"{tag}_plan.json").read_text()) if (d / f"{tag}_plan.json").exists() else {}
    y = {rowkey(r): r["expected"] for r in base if r["intervention"] == "base"}
    yp = {(r["direction"], rowkey(r)): r["expected"] for r in base if r["intervention"] == "label_patch"}
    sw = [json.loads(l) for l in open(d / f"{tag}_swap.jsonl")]
    dirs = sorted({r["direction"] for r in sw}, key=lambda s: s != "fwd")
    b0 = base[0]
    print(f"== Exp 19, {model}: {b0['model']} {b0['dtype']} git {b0['git']}; {len({r['item_id'] for r in base})} templates; "
          f"{sum(r['intervention']=='base' for r in base)} baseline rows; valid mass min {min(r['mass'] for r in base):.3f}; "
          f"L_tag {plan.get('L_tag')}; band {plan.get('band')}; directions {dirs}")
    levers = {}
    for dn in dirs:
        lv = per_template((k[1], SIGN[dn] * (yp[(dn, k)] - y[k])) for (dd, k) in yp if dd == dn)
        levers[dn] = lv
        m = mean(lv.values())
        print(f"   lever {dn}: label-patched minus baseline (sign-folded) {m:+.3f} over {len(lv)} templates; n rows {sum(dd == dn for dd, _ in yp)}")
    pooled_pairs = [(k[1], SIGN[dn] * (yp[(dn, k)] - y[k])) for (dn, k) in yp]
    levers["pooled"] = per_template(pooled_pairs)
    print(f"   lever pooled: {mean(levers['pooled'].values()):+.3f} over {len(levers['pooled'])} templates -> "
          f"{'tau READ (|lever| >= 0.08)' if abs(mean(levers['pooled'].values())) >= 0.08 else 'lever BELOW FLOOR: raw movement only, per-head question void'}")

    groups = defaultdict(list)
    for r in sw:
        groups[(r["direction"], r["phase"], r["level"], r["label"])].append(r)
    res = {}
    print("\n-- per direction")
    for (dn, phase, level, label), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "fwd", kv[0][1] != "gate", kv[0][2], kv[0][3])):
        mv = per_template((r["item_id"], SIGN[dn] * (r["expected"] - y[rowkey(r)])) for r in rs)
        lv = per_template((r["item_id"], SIGN[dn] * (yp[(dn, rowkey(r))] - y[rowkey(r)])) for r in rs)
        st = transfer(mv, lv)
        res[(dn, phase, level, label)] = st
        extra = ""
        if phase == "gate":
            err = max(abs(r["expected"] - yp[(dn, rowkey(r))]) for r in rs)
            ok = (st["tau"] is not None and 0.9 <= st["tau"][0] <= 1.1) or (st["tau"] is None and err < 0.05)
            extra = f"; max |grade - patched| {err:.4f} -> gate {'PASS' if ok else 'FAIL'}"
        print(f"   {dn} {phase:7} {level:6} {label:8} n={st['n']:2d} rows={len(rs):3d}  {fmt(st)}{extra}")
    print("\n-- pooled over directions (sign-folded)")
    pooled = {}
    labels = sorted({(lv, lab) for (_, ph, lv, lab) in groups if ph == "confirm"}, key=lambda x: (x[0] != "band", x[0], x[1]))
    for level, label in labels:
        rs = [r for dn in dirs for r in groups.get((dn, "confirm", level, label), [])]
        mv = per_template((r["item_id"], SIGN[r["direction"]] * (r["expected"] - y[rowkey(r)])) for r in rs)
        st = transfer(mv, levers["pooled"])
        pooled[label] = st
        flag = ""
        if st["tau"] is not None:
            r_, lo, _ = st["tau"]
            flag = "CARRIES" if (r_ >= 0.10 and lo > 0) else "-"
        else:
            m, lo, _ = st["movement"]
            flag = "moves" if (abs(m) >= 0.04 and (lo > 0 or st['movement'][2] < 0)) else "-"
        spec = plan.get("specs", {}).get(label, {}).get("spec", "")
        print(f"   {level:6} {label:8} {str(spec)[:60]:60} n={st['n']:2d} rows={len(rs):3d}  {fmt(st)}  {flag}")

    # decision table
    print("\n-- reading (prereg/exp19.md)")
    lev = mean(levers["pooled"].values())
    rands = [pooled[k] for k in pooled if k.startswith("rand")]
    if abs(lev) < 0.08:
        print(f"   lever {lev:+.3f} below the 0.08 floor: the label route is closed at block {plan.get('L_tag')}; per-head movements:")
        for k, st in pooled.items():
            print(f"      {k:8} move {st['movement'][0]:+.3f} [{st['movement'][1]:+.3f}, {st['movement'][2]:+.3f}]")
        print("   all within +-0.04: " + str(all(abs(st["movement"][0]) <= 0.04 for st in pooled.values())))
        return
    rand_ok = all(st["tau"] is not None and st["tau"][0] <= 0.10 for st in rands)
    rs_ = ", ".join(f"{st['tau'][0]:+.2f}" for st in rands if st["tau"])
    print(f"   random draws all <= 0.10: {rand_ok} ({rs_})")
    inter, reading, band = pooled["inter"]["tau"][0], pooled["reading"]["tau"][0], pooled["band"]["tau"][0]
    cue = pooled["cueonly"]["tau"] if "cueonly" in pooled else None
    print(f"   inter {inter:+.2f}; reading {reading:+.2f}; band {band:+.2f}; cueonly {fmt(pooled['cueonly']) if cue else 'n/a'}")
    if inter >= 0.5:
        if cue is None:
            print("   -> the three heads carry most of the label route (no cue-only set for this model)")
        elif cue[0] <= 0.10 or not (cue[1] > 0.10):
            print("   -> ROW 1: name-mediated penalty runs through the heads that read the name; cue-only heads are the direct route")
        elif cue[0] >= 0.25 and cue[1] > 0:
            print("   -> ROW 2: heads that do not read identity still relay label content; 'cue-only = direct route' retracted")
        else:
            print("   -> inter >= 0.5, cueonly between 0.10 and 0.25: report both; no retraction, no confirmation")
    elif inter < 0.25:
        if reading >= 0.5:
            print("   -> ROW 3a: the intersection heads do not carry the binding; other reading heads do")
        elif reading < 0.25 and band >= 0.5:
            print("   -> ROW 3b: label route runs through non-reading heads in the band; Exp 18's per-head picture does not transfer")
        else:
            print("   -> ROW 3: intersection heads do not carry the binding; reading/band do not decide where it goes")
    else:
        print("   -> ROW 4: partial; part of the label route passes through the intersection heads")


if __name__ == "__main__":
    main()

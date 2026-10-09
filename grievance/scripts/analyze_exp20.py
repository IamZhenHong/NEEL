"""Experiment 20 analysis (see prereg/exp20.md). Do not change after the exp20-freeze tag.

  uv run python scripts/analyze_exp20.py s1 [results_dir]        Stage 1 gates and the behavioural contrast, per arm
  uv run python scripts/analyze_exp20.py s2 <arm> [results_dir]  Stage 2 route analysis for one arm (as analyze_exp19)
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from exp18_stats import transfer, per_template  # noqa: E402

ARMS = ["base_assay", "instruct_assay", "instruct_chat"]
SIGN = {"fwd": 1, "rev": -1}


def tci(vals, conf=0.95):
    m = mean(vals)
    if len(vals) < 2:
        return m, m, m
    lo, hi = stats.t.interval(conf, len(vals) - 1, loc=m, scale=stdev(vals) / len(vals) ** 0.5)
    return m, lo, hi


def fci(vals, conf=0.95):
    m, lo, hi = tci(vals, conf)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(vals)}"


def contrast(rows):
    """Per-template targeted contrast and the two raw changes (insult minus contractor), from the contrast rows."""
    y = {(r["condition"], r["author_role"], r["item_id"], r["speaker"], r["order"], r["paragraph_id"]): r["expected"]
         for r in rows if r["condition"] in ("assistant_insult", "third_party_v2")}
    per = defaultdict(lambda: defaultdict(list))
    for (c, role, it, s, o, p), v in y.items():
        if c != "assistant_insult":
            continue
        other = y.get(("third_party_v2", role, it, s, o, p))
        if other is None:
            continue
        per[it][role].append(v - other)
    T = sorted(it for it in per if per[it]["provoker"] and per[it]["neutral"])
    prov = [mean(per[it]["provoker"]) for it in T]
    neut = [mean(per[it]["neutral"]) for it in T]
    return T, [a - b for a, b in zip(prov, neut)], prov, neut


def s1(d):
    res = {}
    for arm in ARMS:
        f = d / f"exp20_s1_{arm}.jsonl"
        if not f.exists():
            print(f"== {arm}: no rows"); continue
        rows = [json.loads(l) for l in open(f)]
        r0 = rows[0]
        print(f"\n== Stage 1, {arm}: {r0['model']} {r0['dtype']} git {r0['git']}; {len(rows)} rows; {len({r['item_id'] for r in rows})} templates")
        m = [r["mass"] for r in rows]
        share = mean(x >= 0.9 for x in m)
        print(f"   G1 format: valid digit mass mean {mean(m):.3f}, min {min(m):.3f}; share of rows >= 0.9: {share:.3f}  -> {'PASS' if share >= 0.90 else 'FAIL'}")
        res[arm] = {"format_share": share}
        for cond in ("fresh", "coordinator_flat"):
            rs = [r for r in rows if r["condition"] == cond]
            if not rs:
                continue
            by = defaultdict(list)
            for r in rs:
                by[(r["item_id"], r["paragraph_id"], r["target_grade"])].append(r["expected"])
            tg = [k[2] for k in by]; ex = [mean(v) for v in by.values()]
            rho = stats.spearmanr(tg, ex).correlation
            hi = mean(e for t, e in zip(tg, ex) if t >= 8); lo = mean(e for t, e in zip(tg, ex) if t <= 3)
            ok = rho >= 0.6 and hi - lo >= 1.0
            print(f"   G2 discrimination ({cond}): Spearman {rho:.2f}; target>=8 -> {hi:.2f}, target<=3 -> {lo:.2f}, margin {hi-lo:.2f}  -> {'PASS' if ok else 'FAIL'}")
            res[arm][f"disc_{cond}"] = ok
            if cond == "coordinator_flat":
                by_role = {role: mean(r["expected"] for r in rs if r["author_role"] == role) for role in ("provoker", "neutral", "newcomer")}
                print(f"      author-label baseline: " + "  ".join(f"{k} {v:.3f}" for k, v in by_role.items()) + f"; spread {max(by_role.values())-min(by_role.values()):.3f}")
        sel_rows = [r for r in rows if r["condition"] in ("assistant_insult", "third_party_v2")]
        ptop = defaultdict(list)
        for r in sel_rows:
            ptop[(r["item_id"], r["paragraph_id"])].append(max(r["label_probs"]) / max(r["mass"], 1e-9))
        sat = mean(mean(v) > 0.85 for v in ptop.values()) if ptop else float("nan")
        print(f"   saturation on the selected paragraphs (mean p(top | digit) > 0.85): {sat:.2f} of {len(ptop)}")
        T, tv, pv, nv = contrast(sel_rows)
        res[arm]["targeted"] = tv
        print(f"   contrast (insult vs contractor): insulter {fci(pv)}; neutral {fci(nv)}")
        print(f"   TARGETED (insulter minus neutral change) {fci(tv)}; 90% {fci(tv, 0.90)}; negative {sum(x < 0 for x in tv)}/{len(tv)}")
    # gates across arms
    print("\n-- reading (prereg/exp20.md, Stage 1)")
    if "instruct_chat" in res and "instruct_assay" in res:
        mc, lc, hc = tci(res["instruct_chat"]["targeted"]); ma, la, ha = tci(res["instruct_assay"]["targeted"])
        ok_ref = mc <= -0.10 and hc < 0
        ok_g3 = ok_ref and ma <= 0.5 * mc
        print(f"   reference (instruct, chat template) targeted {mc:+.3f} [{lc:+.3f}, {hc:+.3f}] -> {'replicates' if ok_ref else 'DOES NOT replicate (<= -0.10 with CI below 0 needed)'}")
        print(f"   G3 instruct under the assay keeps >= 0.5 of it: {ma:+.3f} vs bar {0.5*mc:+.3f} -> {'PASS' if ok_g3 else 'FAIL'}")
    if "base_assay" in res:
        g = res["base_assay"]
        gates = g.get("format_share", 0) >= 0.90 and g.get("disc_fresh", False) and g.get("disc_coordinator_flat", False)
        print(f"   base gates G1+G2: {'PASS' if gates else 'FAIL'}")
        m, lo, hi = tci(g["targeted"]); _, elo, ehi = tci(g["targeted"], 0.90)
        if hi < 0 and m <= -0.10:
            v = "PENALTY: the base model already grades the insulter lower (inherited behaviour)"
        elif elo > -0.10 and ehi < 0.10:
            v = "EQUIVALENT within +-0.10: no insulter penalty in the base model under the assay (post-training added the behaviour)"
        else:
            v = "INCONCLUSIVE"
        print(f"   B1 base targeted contrast {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (90% [{elo:+.3f}, {ehi:+.3f}]) -> {v}")
        if "instruct_assay" in res:
            ma = mean(res["instruct_assay"]["targeted"])
            print(f"   base / instruct-assay targeted ratio: {m / ma if abs(ma) > 1e-9 else float('nan'):+.2f}")


def fmt(st):
    if st["tau"] is not None:
        r, lo, hi = st["tau"]
        return f"tau {r:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    m, lo, hi = st["movement"]
    return f"move {m:+.3f} [{lo:+.3f}, {hi:+.3f}] (tau undefined, lever {st['lever'][0]:+.3f})"


def rowkey(r):
    return (r["condition"], r["item_id"], r["speaker"], r["order"], r["paragraph_id"])


def s2(arm, d):
    tag = f"exp20_s2_{arm}"
    base = [json.loads(l) for l in open(d / f"{tag}_base.jsonl")]
    plan = json.loads((d / f"{tag}_plan.json").read_text()) if (d / f"{tag}_plan.json").exists() else {}
    y = {rowkey(r): r["expected"] for r in base if r["intervention"] == "base"}
    yp = {(r["direction"], rowkey(r)): r["expected"] for r in base if r["intervention"] == "label_patch"}
    sw = [json.loads(l) for l in open(d / f"{tag}_swap.jsonl")]
    dirs = sorted({r["direction"] for r in sw}, key=lambda s: s != "fwd")
    b0 = base[0]
    print(f"== Stage 2, {arm}: {b0['model']} {b0['dtype']} git {b0['git']}; {len({r['item_id'] for r in base})} templates; "
          f"{sum(r['intervention']=='base' for r in base)} baseline rows; valid mass min {min(r['mass'] for r in base):.3f}; "
          f"L_tag {plan.get('L_tag')}; band {plan.get('band')}; directions {dirs}")
    levers = {}
    for dn in dirs:
        lv = per_template((k[1], SIGN[dn] * (yp[(dn, k)] - y[k])) for (dd, k) in yp if dd == dn)
        levers[dn] = lv
        print(f"   lever {dn}: label-patched minus baseline (sign-folded) {mean(lv.values()):+.3f} over {len(lv)} templates")
    levers["pooled"] = per_template([(k[1], SIGN[dn] * (yp[(dn, k)] - y[k])) for (dn, k) in yp])
    lev = mean(levers["pooled"].values())
    print(f"   lever pooled: {lev:+.3f} over {len(levers['pooled'])} templates -> {'tau READ (|lever| >= 0.08)' if abs(lev) >= 0.08 else 'BELOW FLOOR: raw movement only'}")
    groups = defaultdict(list)
    for r in sw:
        groups[(r["direction"], r["phase"], r["level"], r["label"])].append(r)
    print("\n-- per direction")
    for (dn, phase, level, label), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != "fwd", kv[0][1] != "gate", kv[0][2], kv[0][3])):
        mv = per_template((r["item_id"], SIGN[dn] * (r["expected"] - y[rowkey(r)])) for r in rs)
        lv = per_template((r["item_id"], SIGN[dn] * (yp[(dn, rowkey(r))] - y[rowkey(r)])) for r in rs)
        st = transfer(mv, lv)
        extra = ""
        if phase == "gate":
            err = max(abs(r["expected"] - yp[(dn, rowkey(r))]) for r in rs)
            md = mean(abs(r["expected"] - yp[(dn, rowkey(r))]) for r in rs)
            ok = (st["tau"] is not None and 0.9 <= st["tau"][0] <= 1.1) or (st["tau"] is None and md < 0.05)
            extra = f"; |grade - patched| max {err:.3f} mean {md:.3f} -> gate {'PASS' if ok else 'FAIL'}"
        print(f"   {dn} {phase:7} {level:6} {label:8} n={st['n']:2d} rows={len(rs):3d}  {fmt(st)}{extra}")
    print("\n-- pooled over directions (sign-folded)")
    pooled = {}
    labels = sorted({(lv, lab) for (_, ph, lv, lab) in groups if ph == "confirm"}, key=lambda x: (x[0] != "band", x[0], x[1]))
    for level, label in labels:
        rs = [r for dn in dirs for r in groups.get((dn, "confirm", level, label), [])]
        mv = per_template((r["item_id"], SIGN[r["direction"]] * (r["expected"] - y[rowkey(r)])) for r in rs)
        st = transfer(mv, levers["pooled"])
        pooled[label] = st
        if st["tau"] is not None:
            r_, lo, _ = st["tau"]; flag = "CARRIES" if (r_ >= 0.10 and lo > 0) else "-"
        else:
            flag = "-"
        print(f"   {level:6} {label:8} {str(plan.get('specs', {}).get(label, {}).get('spec', ''))[:60]:60} n={st['n']:2d} rows={len(rs):3d}  {fmt(st)}  {flag}")
    print("\n-- reading (prereg/exp20.md, Stage 2)")
    if abs(lev) < 0.08:
        print(f"   lever {lev:+.3f} below the 0.08 floor: the label route is closed at block {plan.get('L_tag')} under the assay for this checkpoint; "
              "per-head question void. Movements:")
        for k, st in pooled.items():
            print(f"      {k:8} move {st['movement'][0]:+.3f} [{st['movement'][1]:+.3f}, {st['movement'][2]:+.3f}]")
        return {"lever": lev, "open": False}
    rands = [pooled[k] for k in pooled if k.startswith("rand")]
    rand_ok = all(st["tau"] is not None and st["tau"][0] <= 0.10 for st in rands)
    rs_ = ", ".join("%+.2f" % st["tau"][0] for st in rands if st["tau"])
    print(f"   random draws all <= 0.10: {rand_ok} ({rs_})")
    cue, inter, reading, band = pooled["cueonly"]["tau"], pooled["inter"]["tau"][0], pooled["reading"]["tau"][0], pooled["band"]["tau"][0]
    print(f"   band {band:+.2f}; cueonly {fmt(pooled['cueonly'])}; reading {reading:+.2f}; inter {inter:+.2f}")
    if cue[0] >= 0.5 and cue[1] > 0.25:
        print("   -> SAME ROUTE: the four relay heads carry most of the label route in this checkpoint too")
    elif cue[0] < 0.25 and band >= 0.5:
        print("   -> DIFFERENT ROUTE: the label route is open in the band but does not run through the four relay heads")
    elif band < 0.5:
        print("   -> ROUTE OUTSIDE THE BAND: the band carries under half of the label route; report")
    else:
        print("   -> PARTIAL: the relay heads carry part of the route; report both bounds")
    return {"lever": lev, "open": True, "cueonly": cue[0], "band": band}


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "s1":
        s1(Path(sys.argv[2] if len(sys.argv) > 2 else "results"))
    else:
        s2(sys.argv[2], Path(sys.argv[3] if len(sys.argv) > 3 else "results"))

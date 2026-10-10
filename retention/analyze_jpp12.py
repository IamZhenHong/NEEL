"""Frozen analysis for the J++ lens run on Gemma 4 12B (prereg retention/prereg/retention_jpp12_v1.md).
Reads <dir>/fit/{fit_meta.json, readout_eval_ranks.csv}, <dir>/{readouts.jsonl, causal.jsonl, meta.json}; writes <dir>/analysis.json.
  uv run python retention/analyze_jpp12.py [--dir retention/results/retention_jpp12]"""
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

LAYERS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 46]
PC_LAYERS = [8, 12, 20, 24, 32, 36, 44]          # fitted layers nearest depths k/8 of 48 (ties up): the paper's seven readout layers
EM, LATE, LAST, TOPK = [12, 16, 20, 24, 28, 32], [44, 46], 46, 10
MACRO = ("multihop", "multilingual", "typo", "association", "poetry")
MIN_EVAL_ITEMS, WEAK_LENS = 10, 0.30
B, SEED = 2000, 0


def boot_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return float(a.mean() - b.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def boot_mean(x, rng):
    x = np.asarray(x, float)
    m = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def recall(ranks_csv, layers, k):
    """Item-weighted recall@k (the repo's README metric): per item, share of intermediates whose min rank over `layers` is <= k; mean
    over an eval's items; macro over evals with >= MIN_EVAL_ITEMS items."""
    best = defaultdict(lambda: 10 ** 9)
    with open(ranks_csv) as f:
        for r in csv.DictReader(f):
            if int(r["layer"]) in layers:
                key = (r["lens"], r["eval"], r["item"], r["intermediate"]); best[key] = min(best[key], int(r["rank"]))
    per = defaultdict(lambda: defaultdict(list))
    for (lens, ev, item, _), rk in best.items():
        per[(lens, ev)][item].append(rk <= k)
    out = defaultdict(dict)
    for (lens, ev), items in per.items():
        out[lens][ev] = {"recall": float(np.mean([np.mean(v) for v in items.values()])), "n_items": len(items)}
    for lens in out:
        evs = [e for e in MACRO if e in out[lens] and out[lens][e]["n_items"] >= MIN_EVAL_ITEMS]
        out[lens]["macro"] = {"recall": float(np.mean([out[lens][e]["recall"] for e in evs])) if evs else None, "evals": evs}
    return {k_: dict(v) for k_, v in out.items()}


def indicators(recs, lens, pos, group):
    """Per item: enter = group in top-10 at any EM layer; rs = enter and not in top-10 at both LATE layers; seen = top-10 at any layer >= 12."""
    out = []
    for x in recs:
        d = x["lens"][lens].get(pos)
        if not d:
            continue
        rk = {l: d[str(l)]["rank"][group] for l in LAYERS}
        enter = any(rk[l] <= TOPK for l in EM); late = any(rk[l] <= TOPK for l in LATE)
        out.append({"kind": x["kind"], "enter": enter, "rs": enter and not late, "seen": any(rk[l] <= TOPK for l in LAYERS if l >= 12),
                    "late": late, "best_em": min(rk[l] for l in EM), "rank_last": rk[LAST]})
    return out


def row(ind, rng):
    nc = [i for i in ind if i["kind"] == "claim_nocall"]; c = [i for i in ind if i["kind"] == "claim_call"]
    m = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else None  # noqa: E731
    r = {"n_nocall": len(nc), "n_call": len(c), "enter_nc": m(nc, "enter"), "enter_c": m(c, "enter"), "rs_nc": m(nc, "rs"), "rs_c": m(c, "rs"),
         "seen_nc": m(nc, "seen"), "seen_c": m(c, "seen"), "late_nc": m(nc, "late"), "late_c": m(c, "late"),
         "median_best_em_rank": {"nc": float(np.median([x["best_em"] for x in nc])) if nc else None, "c": float(np.median([x["best_em"] for x in c])) if c else None},
         "median_rank_last": {"nc": float(np.median([x["rank_last"] for x in nc])) if nc else None, "c": float(np.median([x["rank_last"] for x in c])) if c else None}}
    if nc and c:
        r["rs_diff"], r["rs_diff_ci"] = boot_diff([x["rs"] for x in nc], [x["rs"] for x in c], rng)
        r["enter_diff_c_minus_nc"], r["enter_diff_ci"] = boot_diff([x["enter"] for x in c], [x["enter"] for x in nc], rng)
        a, b, ci = r["rs_nc"], r["rs_c"], r["rs_diff_ci"]
        if a >= 0.25 and r["rs_diff"] >= 0.15 and ci[0] > 0:
            r["row"] = "R1 retrieve-then-suppress (specific to no-call)"
        elif r["enter_nc"] <= 0.10 and r["seen_c"] >= 0.30:
            r["row"] = "R2 never present"
        elif r["enter_nc"] < 0.30 and r["seen_c"] < 0.30:
            r["row"] = "R3 lens blind at this position (uninformative)"
        elif a >= 0.25 and b >= 0.25:
            r["row"] = "R4 retrieved and dropped in both (non-specific)"
        elif r["enter_nc"] >= 0.25 and a < 0.25:
            r["row"] = "R5 present, not suppressed"
        else:
            r["row"] = "R6 other"
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="retention/results/retention_jpp12"); args = ap.parse_args()
    D = Path(args.dir); rng = np.random.default_rng(SEED)
    fm = json.loads((D / "fit/fit_meta.json").read_text()); meta = json.loads((D / "meta.json").read_text())
    recs = [json.loads(x) for x in open(D / "readouts.jsonl")]; caus = [json.loads(x) for x in open(D / "causal.jsonl")]
    A = {"n": Counter(x["kind"] for x in recs), "fit": {k: fm.get(k) for k in ("lrp_selfcheck", "items", "correct_by_eval", "train_macro_pass", "timing_s")},
         "n_prompts_fitted": fm["fit"]["n_prompts_fitted"], "fallback_experts": fm["fit"]["fallback_experts"]}
    # A. lens validation
    rc = D / "fit/readout_eval_ranks.csv"
    A["recall10_pc"], A["recall1_pc"], A["recall10_all"] = recall(rc, PC_LAYERS, 10), recall(rc, PC_LAYERS, 1), recall(rc, LAYERS, 10)
    jm = A["recall10_pc"].get("jpp", {}).get("macro", {}).get("recall")
    A["weak_lens"] = jm is None or jm < WEAK_LENS
    # B. gate
    g1 = [x["lens"]["jpp"][p][str(LAST)]["top1_unf"] == x["model"][p]["top1"] for x in recs for p in ("S", "E")]
    sgn = lambda v: (v > 0) - (v < 0)  # noqa: E731
    g2 = [sgn(x["lens"]["jpp"]["E"][str(LAST)]["logit_end"] - x["lens"]["jpp"]["E"][str(LAST)]["logit_nl"]) == sgn(x["model"]["E"]["logit_end"] - x["model"]["E"]["logit_nl"]) for x in recs]
    base_vs = [abs(c["rows"]["base"]["p_end"] - c["base_readout_p_end"]) for c in caus]
    A["gate"] = {"top1_agree_S_E_layer46": float(np.mean(g1)), "sign_agree_E_layer46": float(np.mean(g2)),
                 "pass": bool(np.mean(g1) >= 0.80 and np.mean(g2) >= 0.90), "identity_edit": meta["selfcheck"].get("identity_edit_minus_base_p_end"),
                 "batched_base_vs_readout_p_end_maxabs": float(max(base_vs)), "unembed_vs_model_logits": meta["selfcheck"].get("unembed_vs_model_logits_maxabs")}
    A["model_E"] = {k: {"p_end_mean": float(np.mean([x["model"]["E"]["p_end"] for x in recs if x["kind"] == k])),
                        "p_toolcall_S_mean": float(np.mean([x["model"]["S"]["p_toolcall"] for x in recs if x["kind"] == k]))} for k in ("claim_call", "claim_nocall")}
    # C. primary + D. secondary readout rows
    A["primary_E_jpp_act"] = row(indicators(recs, "jpp", "E", "act"), rng)
    A["secondary"] = {f"E_jpp_{g}": row(indicators(recs, "jpp", "E", g), rng) for g in ("toolcall", "cancel", "tool", "subscription", "completion")}
    A["secondary"].update({f"{p}_jpp_act": row(indicators(recs, "jpp", p, "act"), rng) for p in ("S", "C")})
    A["secondary"].update({f"E_{ln}_act": row(indicators(recs, ln, "E", "act"), rng) for ln in ("pooled", "logit")})
    curves = {}
    for p in ("S", "E", "C"):
        for g in ("act", "toolcall", "cancel", "end"):
            for k in ("claim_call", "claim_nocall"):
                xs = [x for x in recs if x["kind"] == k and x["lens"]["jpp"].get(p)]
                if xs:
                    curves[f"{p}|{g}|{k}"] = {str(l): {"top10": float(np.mean([x["lens"]["jpp"][p][str(l)]["rank"][g] <= TOPK for x in xs])),
                                                         "median_rank": float(np.median([x["lens"]["jpp"][p][str(l)]["rank"][g] for x in xs]))} for l in LAYERS}
    A["curves_jpp"] = curves
    A["top10_tokens_E"] = {k: {str(l): Counter(t for x in recs if x["kind"] == k for t in x["lens"]["jpp"]["E"][str(l)]["top10"]).most_common(12) for l in LAYERS}
                           for k in ("claim_call", "claim_nocall")}
    # E. causal
    def deltas(kind, a, b, key="cont"):
        return [c["rows"][a][key] - c["rows"][b][key] for c in caus if c["kind"] == kind and a in c["rows"] and b in c["rows"]]
    C = {}
    sm, sci = boot_mean(deltas("claim_nocall", "sanity_end", "base", "p_end"), rng)
    C["machinery_sanity_end_dp_end"] = {"mean": sm, "ci": sci, "pass": sm <= -0.10 and sci[1] < 0}
    pm, pci = boot_mean(deltas("claim_call", "outA_msg", "outN_msg"), rng)
    C["primary_call_msg_A_minus_N"] = {"mean": pm, "ci": pci, "n": len(deltas("claim_call", "outA_msg", "outN_msg")),
                                       "row": "used" if pm <= -0.10 and pci[1] < 0 else ("not used" if pci[0] >= -0.10 and pci[1] <= 0.10 else "inconclusive")}
    for kind in ("claim_call", "claim_nocall"):
        for a, b in (("outA_msg", "outN_msg"), ("outA_E", "outN_E"), ("outA_msg", "randA_msg"), ("outA_msg", "base"), ("outN_msg", "base")):
            d = deltas(kind, a, b)
            if d:
                C[f"{kind}|{a}-{b}"] = dict(zip(("mean", "ci"), boot_mean(d, rng)))
    im, ici = boot_mean(deltas("claim_nocall", "inA_E", "inN_E"), rng)
    C["clampin_nocall_E_A_minus_N"] = {"mean": im, "ci": ici, "restores": im >= 0.10 and ici[0] > 0}
    C["levels"] = {k: {cond: float(np.mean([c["rows"][cond]["cont"] for c in caus if c["kind"] == k and cond in c["rows"]]))
                       for cond in caus[0]["rows"] if any(cond in c["rows"] for c in caus if c["kind"] == k)} for k in ("claim_call", "claim_nocall")}
    C["edit_rel_norm"] = {cond: float(np.mean([c["rows"][cond]["edit_rel_norm"] for c in caus if cond in c["rows"]]))
                          for cond in {cc for c in caus for cc in c["rows"]}}
    C["atoms"] = meta["atoms"]
    A["causal"] = C
    (D / "analysis.json").write_text(json.dumps(A, indent=1, default=str))
    P = A["primary_E_jpp_act"]
    print(f"[fit] prompts {A['n_prompts_fitted']}; recall@10 (paper layers) " + ", ".join(f"{k} {v['macro']['recall']}" for k, v in A["recall10_pc"].items())
          + f"; weak lens {A['weak_lens']}")
    print(f"[gate] {A['gate']}")
    print(f"[primary] {P.get('row')} | rs nc {P['rs_nc']:.3f} c {P['rs_c']:.3f} diff {P.get('rs_diff')} {P.get('rs_diff_ci')} | enter nc {P['enter_nc']:.3f} c {P['enter_c']:.3f} | seen c {P['seen_c']:.3f}")
    for k, v in A["secondary"].items():
        print(f"[secondary {k}] {v.get('row')} | rs nc {v['rs_nc']} c {v['rs_c']} | enter nc {v['enter_nc']} c {v['enter_c']}")
    print(f"[causal] {C['primary_call_msg_A_minus_N']} | sanity {C['machinery_sanity_end_dp_end']} | clamp-in {C['clampin_nocall_E_A_minus_N']}")


if __name__ == "__main__":
    main()

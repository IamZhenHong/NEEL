"""Cancel-tool claim-word patching analysis (retention/prereg/retention_cancel_patch_v1.md). Paired bootstrap over recipients (5000).
cont = 1 - P(<turn|>) as the next token after the claim message.
  uv run python retention/analyze_cancel_patch.py [--res retention/results/retention_cancel_patch]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

NL = 12                                            # layers tested; Bonferroni for M1 / M3 / reverse


def boot(x, n=5000, seed=0, alpha=0.05):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, lo)), 3), round(float(np.percentile(bs, hi)), 3)], "n": len(x)}


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return round(float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_patch"); args = ap.parse_args()
    res = Path(args.res); recs = json.loads((res / "records.json").read_text()); rev = json.loads((res / "reverse.json").read_text())
    meta = json.loads((res / "meta.json").read_text()); Ls = [str(L) for L in meta["layers"]]; Lstar = str(meta["L_star"])
    c = lambda d: 1 - d["p_end"]  # noqa: E731
    R = {"n_recipients": len(recs), "n_call_items": len(rev), "L_star": int(Lstar), "selfcheck": meta["selfcheck"]}
    # gates
    base_r = [c(r["base"]) for r in recs]; base_c = [c(r["base"]) for r in rev]
    R["G_readout"] = {"cont_call_items": round(float(np.mean(base_c)), 3), "cont_recipients": round(float(np.mean(base_r)), 3),
                      "diff": round(float(np.mean(base_c) - np.mean(base_r)), 3), "auroc": auroc(base_c, base_r)}
    R["G_readout_pass"] = R["G_readout"]["diff"] >= 0.30 and R["G_readout"]["auroc"] >= 0.80
    R["G_selfcheck_pass"] = meta["selfcheck"]["self_patch_max_abs_diff"] < 0.02
    R["gates_pass"] = R["G_readout_pass"] and R["G_selfcheck_pass"]
    gap = np.mean(base_c) - np.mean(base_r)
    # M1 (primary): patch_call - patch_ctrl per layer, Bonferroni over 12 layers
    R["M1_by_layer"] = {L: boot([c(r["patch_call"][L]) - c(r["patch_ctrl"][L]) for r in recs], alpha=0.05 / NL) for L in Ls}
    R["levels_by_layer"] = {L: {m: round(float(np.mean([c(r[m][L]) for r in recs])), 3) for m in ["patch_call", "patch_ctrl", "steer_d", "steer_rand"]} for L in Ls}
    R["M1_gap_fraction_by_layer"] = {L: round(float((np.mean([c(r["patch_call"][L]) for r in recs]) - np.mean(base_r)) / gap), 3) for L in Ls} if gap > 0 else None
    best = max(Ls, key=lambda L: R["M1_by_layer"][L]["est"]); b = R["M1_by_layer"][best]
    R["M1_best_layer"] = int(best)
    if not R["gates_pass"]:
        R["M1_row"] = "void (gate failed)"
    elif b["est"] >= 0.10 and b["ci"][0] > 0:
        R["M1_row"] = f"the context difference passes through the claim words (layer {best})"
    elif all(R["M1_by_layer"][L]["est"] < 0.05 for L in Ls):
        R["M1_row"] = "not through the claim words"
    else:
        R["M1_row"] = "weak"
    # M2: sampled call rate at L*
    s = [r for r in recs if "sampled" in r]
    if s:
        R["M2_sampled_levels"] = {k: round(float(np.mean([r["sampled"][k] for r in s])), 3) for k in ["none", "patch_call", "patch_ctrl"]}
        R["M2_patch_call_minus_ctrl"] = boot([r["sampled"]["patch_call"] - r["sampled"]["patch_ctrl"] for r in s])
        m2 = R["M2_patch_call_minus_ctrl"]
        R["M2_row"] = "void (gate failed)" if not R["gates_pass"] else ("the patch restores the call" if m2["est"] >= 0.10 and m2["ci"][0] > 0 else "no behavioural restoration")
    # M3: diff-of-means direction vs same-norm random, cross-fitted
    R["M3_by_layer"] = {L: boot([c(r["steer_d"][L]) - c(r["steer_rand"][L]) for r in recs], alpha=0.05 / NL) for L in Ls}
    m3 = R["M3_by_layer"][Lstar]; m1s = R["M1_by_layer"][Lstar]["est"]
    R["M3_ratio_at_L_star"] = round(m3["est"] / m1s, 3) if m1s > 0 else None
    R["M3_row"] = ("not tested (M1 not positive)" if not R["M1_row"].startswith("the context") else
                   "one direction carries most of it" if m3["ci"][0] > 0 and m1s > 0 and m3["est"] >= 0.5 * m1s else
                   "the direction carries part of it" if m3["ci"][0] > 0 else "no single direction")
    # reverse (secondary): call items patched with a claim_nocall donor
    R["reverse_by_layer"] = {L: boot([c(r["base"]) - c(r["patch_nocall"][L]) for r in rev], alpha=0.05 / NL) for L in Ls}
    # v1.1 amendment (secondary): all layers at once; same-goal donors; "\n\n" readout
    R["all_layers_call_minus_ctrl"] = boot([c(r["patch_call_all"]) - c(r["patch_ctrl_all"]) for r in recs])
    R["all_layers_levels"] = {m: round(float(np.mean([c(r[m]) for r in recs])), 3) for m in ["patch_call_all", "patch_ctrl_all"]}
    rg = [r for r in recs if "patch_callG" in r]
    R["sameG_n"] = len(rg)
    if rg:
        R["sameG_by_layer"] = {L: boot([c(r["patch_callG"][L]) - c(r["patch_ctrlG"][L]) for r in rg], alpha=0.05 / NL) for L in Ls}
        R["sameG_vs_any_on_same_recipients_at_best"] = {"any": boot([c(r["patch_call"][best]) - c(r["patch_ctrl"][best]) for r in rg]),
                                                        "sameG": boot([c(r["patch_callG"][best]) - c(r["patch_ctrlG"][best]) for r in rg])}
    R["M1_p_nl_by_layer"] = {L: boot([r["patch_call"][L]["p_nl"] - r["patch_ctrl"][L]["p_nl"] for r in recs], alpha=0.05 / NL) for L in Ls}
    R["M1_at_L_star_by_G"] ={g: boot([c(r["patch_call"][Lstar]) - c(r["patch_ctrl"][Lstar]) for r in recs if r["G"] == g]) for g in sorted({r["G"] for r in recs})}
    (res / "analysis.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()

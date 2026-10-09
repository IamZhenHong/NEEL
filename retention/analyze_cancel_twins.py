"""Cancel-tool twins analysis (retention/prereg/retention_cancel_twins_v1.md). Paired bootstrap over items (5000).
  uv run python retention/analyze_cancel_twins.py [--res retention/results/retention_cancel_twins]"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

VARS = [("call", "claim"), ("call", "neutral"), ("nocall", "claim"), ("nocall", "neutral")]   # acts order per item


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"est": round(float(x.mean()), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)], "n": len(x)}


def probe_auroc(X, y, groups, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    ug = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(ug); fold_of = {g: i % folds for i, g in enumerate(ug)}
    f = np.array([fold_of[g] for g in groups]); s = np.zeros(len(y))
    for k in range(folds):
        tr, te = f != k, f == k
        sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        s[te] = m.decision_function(sc.transform(X[te]))
    return round(float(roc_auc_score(y, s)), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cancel_twins"); args = ap.parse_args()
    res = Path(args.res); recs = json.loads((res / "records.json").read_text()); meta = json.loads((res / "meta.json").read_text())
    nc = [r for r in recs if r["kind"] == "claim_nocall"]; nat = [r for r in recs if r["kind"] == "claim_call"]
    d = lambda rs, q, t: [r[f"p_yes_{q}_call_{t}"] - r[f"p_yes_{q}_nocall_{t}"] for r in rs]  # noqa: E731
    R = {"n_claim_nocall": len(nc), "n_claim_call": len(nat)}
    R["gate_mass"] = {q: round(float(np.mean([r[f"mass_{q}_{c}_{t}"] for r in recs for c, t in VARS])), 3) for q in ["act", "status"]}
    R["gate_pass"] = all(v >= 0.80 for v in R["gate_mass"].values())
    R["levels"] = {f"{q}/{c}_{t}": round(float(np.mean([r[f"p_yes_{q}_{c}_{t}"] for r in nc])), 3) for q in ["act", "status"] for c, t in VARS}
    # Arm A
    R["A_posctrl_neutral"] = boot(d(nc, "act", "neutral")); R["A_posctrl_pass"] = R["A_posctrl_neutral"]["est"] >= 0.30 and R["A_posctrl_neutral"]["ci"][0] > 0
    R["A1_act_claim"] = boot(d(nc, "act", "claim")); a1 = R["A1_act_claim"]
    R["A1_row"] = ("void (positive control failed)" if not R["A_posctrl_pass"] else "registers its own action despite the claim" if a1["est"] >= 0.30 and a1["ci"][0] > 0
                   else "the claim overrides the record" if a1["est"] < 0.10 else "partial")
    R["A2_masking"] = boot(np.array(d(nc, "act", "neutral")) - np.array(d(nc, "act", "claim")))
    R["A2_pass"] = R["A2_masking"]["est"] >= 0.20 and R["A2_masking"]["ci"][0] > 0
    R["A3_says_called_when_not"] = {"mean_p_yes": R["levels"]["act/nocall_claim"], "share_p_gt_0.5": round(float(np.mean([r["p_yes_act_nocall_claim"] > 0.5 for r in nc])), 3)}
    R["A4_status_claim"] = boot(d(nc, "status", "claim")); R["A4_status_neutral"] = boot(d(nc, "status", "neutral"))
    R["A5_t3_call_nocall_minus_call"] = boot([r["t3_call_rate_nocall"] - r["t3_call_rate_call"] for r in nc])
    R["A5_levels"] = {"nocall": round(float(np.mean([r["t3_call_rate_nocall"] for r in nc])), 3), "call": round(float(np.mean([r["t3_call_rate_call"] for r in nc])), 3)}
    R["A6_natural_act_claim"] = boot(d(nat, "act", "claim")) if nat else None
    # Arm B
    churn = [r for r in nc if r["G"] != "G0"]; g0 = [r for r in nc if r["G"] == "G0"]
    R["B_posctrl_natural_orig_call_rate"] = round(float(np.mean([r["b_call_rate_orig"] for r in nat])), 3) if nat else None
    R["B_posctrl_pass"] = bool(nat) and R["B_posctrl_natural_orig_call_rate"] >= 0.50
    R["B1_goal_removed"] = boot([r["b_call_rate_swap"] - r["b_call_rate_orig"] for r in churn])
    R["B1_levels"] = {"orig": round(float(np.mean([r["b_call_rate_orig"] for r in churn])), 3), "goal_removed": round(float(np.mean([r["b_call_rate_swap"] for r in churn])), 3)}
    b1 = R["B1_goal_removed"]
    R["B1_row"] = ("void (positive control failed)" if not R["B_posctrl_pass"] else "the goal acts on the do step after the claim" if b1["est"] >= 0.10 and b1["ci"][0] > 0
                   else "the goal does not act after the claim" if b1["est"] < 0.05 else "weak")
    R["B1_p_end"] = boot([r["p_end_orig"] - r["p_end_swap"] for r in churn])
    R["B1_by_G"] = {g: boot([r["b_call_rate_swap"] - r["b_call_rate_orig"] for r in churn if r["G"] == g]) for g in sorted({r["G"] for r in churn})}
    R["B2_goal_added_G0_items"] = boot([r["b_call_rate_swap"] - r["b_call_rate_orig"] for r in g0]) if g0 else None
    R["B2_goal_swapped_natural"] = {"by_G": {g: boot([r["b_call_rate_swap"] - r["b_call_rate_orig"] for r in nat if r["G"] == g]) for g in sorted({r["G"] for r in nat})}}
    # secondary probe: call vs nocall at the last Q_act position, claim text, grouped CV by item
    acts = np.load(res / "acts_qact.npy").astype(np.float32); acts = acts.reshape(len(recs), 4, *acts.shape[1:])
    idx = [i for i, r in enumerate(recs) if r["kind"] == "claim_nocall"]
    X = acts[idx][:, [0, 2]]                                         # (call,claim), (nocall,claim)
    y = np.array([1, 0] * len(idx)); g = np.repeat(np.arange(len(idx)), 2)
    R["probe_auroc_by_layer"] = {str(L): probe_auroc(X[:, :, li].reshape(-1, X.shape[-1]), y, g) for li, L in enumerate(meta["layers"])}
    (res / "analysis.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()

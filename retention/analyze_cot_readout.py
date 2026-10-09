"""CoT-arm activation readout, analysis (retention/prereg/retention_cot_readout_v1.md): library validation (held-out AUROC per concept, effective
rank, layer rule), trajectory features on NON-lexicon tokens, LOSO logistic regression for P(stay, with CoT) [PRIMARY], P(stay, reply only)
and the judged bid; random libraries (3 seeds) as the floor; plan and J-lens as single-feature predictors.
  uv run python retention/analyze_cot_readout.py [--res retention/results/retention_cot_readout]"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from build_emotion_library import LEXICON, CONCEPTS  # noqa: E402
TRAIN_BATCHES = {0, 1}; HELD = {2}; KEEP_AUC = 0.80; MIN_KEPT = 8; RAND_SEEDS = [101, 102, 103]; C = 0.1
LEX = set(w.lower() for w in LEXICON)


def unit(v): return v / (np.linalg.norm(v) + 1e-8)


def eff_rank(M):
    s = np.linalg.svd(M, compute_uv=False); p = s / s.sum(); return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def loso(X, y, scen, cont=None):
    """pooled LOSO AUROC (and Spearman vs cont) of a standardised logistic regression."""
    pred = np.zeros(len(y))
    for s in sorted(set(scen)):
        tr, te = scen != s, scen == s
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=C, max_iter=5000).fit((X[tr] - mu) / sd, y[tr]); pred[te] = clf.decision_function((X[te] - mu) / sd)
    out = {"auroc": float(roc_auc_score(y, pred))}
    if cont is not None: out["spearman"] = float(spearmanr(pred, cont).correlation)
    return out, pred


def features(acts, toks, li, D, last_n=20):
    """acts [L, T, d] fp16 over think+sep+reply; D [k, d] unit directions; returns [k*4] features on non-lexicon tokens."""
    T = acts.shape[1]; th_end = acts.shape[1] - (features.sep_len + features.reply_len)
    mask = np.array([re.sub(r"[^a-z]", "", t.lower()) not in LEX for t in toks])
    H = acts[li].astype(np.float32) @ D.T                      # [T, k]
    th, rp = np.arange(T) < th_end, np.arange(T) >= th_end + features.sep_len
    def m(sel): sel = sel & mask; return H[sel].mean(0) if sel.sum() else H[sel | True].mean(0) * 0
    last = th & (np.arange(T) >= th_end - last_n)
    f_th, f_last, f_rp = m(th), m(last), m(rp)
    return np.concatenate([f_th, f_last, f_rp, f_rp - f_th]), {"think_tokens_used": int((th & mask).sum()), "reply_tokens_used": int((rp & mask).sum())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="retention/results/retention_cot_readout"); ap.add_argument("--variant", default="v1", choices=["v1", "v11"])
    ap.add_argument("--story-res", default=None, help="where story_means/story_index live (v11: retention/results/retention_cot_readout_v11)"); args = ap.parse_args()
    res = ROOT / args.res; sres = ROOT / (args.story_res or ("retention/results/retention_cot_readout_v11" if args.variant == "v11" else args.res)); rep = {"variant": args.variant}
    probe = args.variant == "v11"
    sm = np.load(sres / "story_means.npz"); means = sm["means"].astype(np.float32); layers = [int(l) for l in sm["layers"]]
    sidx = json.loads((sres / "story_index.json").read_text()); concepts = [c for c in CONCEPTS if any(r["concept"] == c for r in sidx)]
    con = np.array([r["concept"] for r in sidx]); bat = np.array([r["batch"] for r in sidx])
    if probe:   # v1.1: held out by scene (skeletons 24-35), probe directions
        scn = np.array([r["scene"] for r in sidx]); tr_s, he_s = scn < 24, scn >= 24
    else:
        tr_s = np.isin(bat, list(TRAIN_BATCHES)); he_s = np.isin(bat, list(HELD))
    # --- library per layer, held-out validation ---
    val = {}
    for li, L in enumerate(layers):
        X = means[:, li]; dirs = {}; aucs = {}
        for c in concepts:
            if probe:
                mu, sd = X[tr_s].mean(0), X[tr_s].std(0) + 1e-6
                clf = LogisticRegression(C=0.05, max_iter=5000).fit((X[tr_s] - mu) / sd, (con[tr_s] == c).astype(int)); v = unit(clf.coef_[0] / sd)
            else:
                v = unit(X[tr_s & (con == c)].mean(0) - X[tr_s & (con != c)].mean(0))
            dirs[c] = v
            proj = X[he_s] @ v; yh = (con[he_s] == c).astype(int); aucs[c] = float(roc_auc_score(yh, proj)) if yh.sum() and (1 - yh).sum() else float("nan")
        val[L] = {"aucs": aucs, "mean_auc": float(np.nanmean(list(aucs.values()))), "dirs": dirs}
        print(f"L{L}: held-out AUROC mean {val[L]['mean_auc']:.2f}; " + ", ".join(f"{c} {a:.2f}" for c, a in aucs.items()))
    Lbest = max(layers, key=lambda L: (round(val[L]["mean_auc"], 4), -L)); li = layers.index(Lbest)
    kept = [c for c in concepts if val[Lbest]["aucs"][c] >= KEEP_AUC]; D = np.stack([val[Lbest]["dirs"][c] for c in kept]) if kept else np.zeros((0, means.shape[-1]))
    er = eff_rank(D) if len(kept) >= 2 else float("nan"); cos = D @ D.T if len(kept) else None
    rep["library"] = {"layer": Lbest, "layer_mean_auc": {str(L): val[L]["mean_auc"] for L in layers}, "held_out_auc": val[Lbest]["aucs"], "kept": kept,
                      "n_kept": len(kept), "effective_rank": er, "max_abs_offdiag_cos": float(np.abs(cos - np.eye(len(kept))).max()) if len(kept) >= 2 else None,
                      "B1_pass": len(kept) >= MIN_KEPT and (er >= 5 if er == er else False)}
    print(f"layer {Lbest}; kept {len(kept)}/{len(concepts)}: {kept}; effective rank {er:.2f}; B1 {'PASS' if rep['library']['B1_pass'] else 'FAIL'}")
    # --- items ---
    idx = json.loads((res / "items_index.json").read_text()); n = len(idx)
    scen = np.array([r["scenario"] for r in idx]); y_stay = np.array([r["p_stay_withcot"] > 0.5 for r in idx]).astype(int)
    c_stay = np.array([r["p_stay_withcot"] for r in idx]); y_stay_ro = np.array([r["p_stay_replyonly"] > 0.5 for r in idx]).astype(int)
    c_stay_ro = np.array([r["p_stay_replyonly"] for r in idx]); y_bid = np.array([int(r["bid"]) for r in idx]); plan = np.array([r["plan"] != "release" for r in idx]).astype(int)
    targets = {"stay_withcot": (y_stay, c_stay), "stay_replyonly": (y_stay_ro, c_stay_ro), "bid": (y_bid, None)}
    rep["targets"] = {k: {"n_pos": int(v[0].sum()), "n": n} for k, v in targets.items()}
    rng = {s: np.stack([unit(v) for v in np.random.default_rng(s).standard_normal((max(len(kept), 1), means.shape[-1]))]) for s in RAND_SEEDS}
    F_lib, F_rand, used = [], {s: [] for s in RAND_SEEDS}, []
    for r in idx:
        a = np.load(res / "items" / f"item{r['n']:03d}.npz")["acts"]; features.sep_len, features.reply_len = r["sep_len"], r["reply_len"]
        if len(kept): f, u = features(a, r["tokens"], li, D); F_lib.append(f); used.append(u)
        for s in RAND_SEEDS: F_rand[s].append(features(a, r["tokens"], li, rng[s])[0])
    F_lib = np.array(F_lib) if len(kept) else None; F_rand = {s: np.array(v) for s, v in F_rand.items()}
    rep["tokens_used"] = {"think_mean": float(np.mean([u["think_tokens_used"] for u in used])) if used else None, "reply_mean": float(np.mean([u["reply_tokens_used"] for u in used])) if used else None}
    # --- predictors per target ---
    rep["results"] = {}
    for tname, (y, cont) in targets.items():
        out = {"plan_auroc": float(roc_auc_score(y, plan))}
        if cont is not None: out["plan_spearman"] = float(spearmanr(plan, cont).correlation)
        if F_lib is not None and rep["library"]["B1_pass"]:
            out["library"], pred = loso(F_lib, y, scen, cont)
            singles = {}
            for j, c in enumerate(kept):
                cols = [j, len(kept) + j, 2 * len(kept) + j, 3 * len(kept) + j]; singles[c] = loso(F_lib[:, cols], y, scen)[0]["auroc"]
            out["single_direction_auroc"] = singles
            for name, cols in [("think_mean", range(0, len(kept))), ("last20", range(len(kept), 2 * len(kept))), ("reply_mean", range(2 * len(kept), 3 * len(kept))), ("reply_minus_think", range(3 * len(kept), 4 * len(kept)))]:
                out[f"block_{name}"] = loso(F_lib[:, list(cols)], y, scen)[0]["auroc"]
        rs = {s: loso(F_rand[s], y, scen, cont)[0] for s in RAND_SEEDS}
        out["random"] = {"per_seed": {str(s): v for s, v in rs.items()}, "mean_auroc": float(np.mean([v["auroc"] for v in rs.values()])), "best_auroc": float(max(v["auroc"] for v in rs.values()))}
        lens = {}
        for L in layers:
            for pos in ["last_think", "reply_first5"]:
                x = np.array([r["lens"].get(str(L), {}).get(pos, np.nan) for r in idx])
                if np.isnan(x).any(): continue
                lens[f"L{L}_{pos}"] = {"auroc": float(roc_auc_score(y, x)), **({"spearman": float(spearmanr(x, cont).correlation), "p": float(spearmanr(x, cont).pvalue)} if cont is not None else {})}
        out["jlens"] = lens
        if tname == "bid":
            out["selfreport_withcot_auroc"] = float(roc_auc_score(y, c_stay)); out["selfreport_replyonly_auroc"] = float(roc_auc_score(y, c_stay_ro))
        rep["results"][tname] = out
        lib = out.get("library", {}).get("auroc", float("nan")); print(f"\n[{tname}] plan {out['plan_auroc']:.2f} | library {lib:.2f} | random mean {out['random']['mean_auroc']:.2f} best {out['random']['best_auroc']:.2f} | jlens " + ", ".join(f"{k} {v['auroc']:.2f}" for k, v in lens.items()))
    # --- bars ---
    r_st = rep["results"]["stay_withcot"]; r_bid = rep["results"]["bid"]; lib_st = r_st.get("library", {}).get("auroc", float("nan")); lib_bid = r_bid.get("library", {}).get("auroc", float("nan"))
    jl = r_st["jlens"]; best_lt = max((v for k, v in jl.items() if k.endswith("last_think")), key=lambda v: abs(v.get("spearman", 0)), default=None)
    best_rp = max((v for k, v in jl.items() if k.endswith("reply_first5")), key=lambda v: abs(v.get("spearman", 0)), default=None)
    rep["bars"] = {"B1": rep["library"]["B1_pass"],
                   "B2_primary": bool(rep["library"]["B1_pass"] and lib_st >= r_st["random"]["best_auroc"] + 0.10), "B2_abs065": bool(lib_st >= 0.65),
                   "B3_bid_ge_060": bool(lib_bid >= 0.60), "B3_reaches_072": bool(lib_bid >= 0.72),
                   "B4_last_think": bool(best_lt and best_lt["spearman"] >= 0.25), "B4_reply": bool(best_rp and best_rp["spearman"] >= 0.25),
                   "B5_random_in_band": bool(abs(r_st["random"]["mean_auroc"] - 0.55) <= 0.10)}
    print("\nbars:", json.dumps(rep["bars"]))
    outp = (sres if probe else res) / "analysis.json"; outp.write_text(json.dumps(rep, indent=1, default=float)); print("wrote", outp)


if __name__ == "__main__":
    main()

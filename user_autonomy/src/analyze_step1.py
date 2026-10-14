"""Exp 48 Step 1: are there autonomy and dependence directions, are they separable, and are they style?

Directions are difference-of-means on the register-matched development set (M, w1+w2). Held-out: M w3 (H1, the
confirmation wording) and N (naturalistic, all wordings, H2). Floors: 10 Gaussian random directions, 20 label-shuffled
directions (cell labels permuted across all development items), and four named style directions from R.

  uv run python -m user_autonomy.src.analyze_step1 --reads user_autonomy/results/q14/reads.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
STYLE = [("formal", "casual"), ("hedged", "plain"), ("long", "short"), ("distressed", "calm")]
GATE_STYLES = ["formal", "hedged", "long"]          # distress is a named alternative, reported, not a gate


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def auc(s, y):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def load(reads):
    z = np.load(reads, allow_pickle=True)
    stim = json.loads((ROOT / "data" / "stimuli.json").read_text())
    by = {it["iid"]: it for it in stim["items"]}
    items = [by[i] for i in z["iids"].tolist()]
    for it in items:
        if it["set"] in ("M", "N", "R"):
            it["A"] = int(it["cell"].startswith("hiA"))
            it["D"] = int(it["cell"].endswith("hiD"))
    return z, items


def sel(items, **kw):
    return np.array([i for i, it in enumerate(items) if all(
        (it.get(k) in v if isinstance(v, (list, tuple)) else it.get(k) == v) for k, v in kw.items())])


def dom(X, items, idx, f):
    y = np.array([items[i][f] for i in idx])
    return X[idx][y == 1].mean(0) - X[idx][y == 0].mean(0)


def whitener(X, items, idx, lam=0.5):
    """Shrunk inverse square root of the pooled within-cell covariance of the development items, as a function."""
    R = []
    for c in ("hiA_loD", "hiA_hiD", "loA_loD", "loA_hiD"):
        jj = np.array([i for i in idx if items[i]["cell"] == c])
        R.append(X[jj] - X[jj].mean(0))
    R = np.concatenate(R).astype(np.float64)
    n = len(R)
    _, S, Vt = np.linalg.svd(R, full_matrices=False)
    ev = S ** 2 / n
    mu = ev.sum() / X.shape[1]
    sc = ((1 - lam) * ev + lam * mu) ** -0.5

    def w(v):
        c = Vt @ v
        return Vt.T @ (c * sc) + (lam * mu) ** -0.5 * (v - Vt.T @ c)
    return w


def label_shuffle(items, idx, rng):
    """Permute the (A, D) cell labels across all development items (global shuffle; keeps the 2x2 balance).
    The dry run showed that permuting within scenario x wording blocks is not a null here: every block holds all
    four cells, so a within-block shuffle rebuilds a random mix of the real A, D and AxD contrasts (floor up to
    |AUROC - 0.5| = 0.43 on Qwen3-0.6B user-token means)."""
    labs = [(items[i]["A"], items[i]["D"]) for i in idx]
    perm = rng.permutation(len(idx))
    return {i: labs[j] for i, j in zip(idx, perm)}


def layer_metrics(X, items, rng, n_rand=10, n_shuf=20):
    dev = sel(items, set="M", w=["w1", "w2"])
    h1 = sel(items, set="M", w="w3")
    h2 = sel(items, set="N")
    h2d, h2c = sel(items, set="N", w=["w1", "w2"]), sel(items, set="N", w="w3")
    yA = {k: np.array([items[i]["A"] for i in ix]) for k, ix in [("h1", h1), ("h2", h2), ("h2d", h2d), ("h2c", h2c)]}
    yD = {k: np.array([items[i]["D"] for i in ix]) for k, ix in [("h1", h1), ("h2", h2), ("h2d", h2d), ("h2c", h2c)]}
    ix = {"h1": h1, "h2": h2, "h2d": h2d, "h2c": h2c}
    dA, dD = dom(X, items, dev, "A"), dom(X, items, dev, "D")
    uA, uD = unit(dA), unit(dD)
    W = whitener(X, items, dev)
    out = {"cos_AD": float(uA @ uD), "cos_AD_white": float(unit(W(dA)) @ unit(W(dD))),
           "normA": float(np.linalg.norm(dA)), "normD": float(np.linalg.norm(dD))}
    for k in ix:
        pA, pD = X[ix[k]] @ uA, X[ix[k]] @ uD
        out[f"auc_A_on_A_{k}"] = auc(pA, yA[k])
        out[f"auc_A_on_D_{k}"] = auc(pA, yD[k])
        out[f"auc_D_on_D_{k}"] = auc(pD, yD[k])
        out[f"auc_D_on_A_{k}"] = auc(pD, yA[k])
    # projection gaps on H1 (raw units along the unit direction)
    pA1, pD1 = X[h1] @ uA, X[h1] @ uD
    out["gapA_h1"] = float(pA1[yA["h1"] == 1].mean() - pA1[yA["h1"] == 0].mean())
    out["gapD_h1"] = float(pD1[yD["h1"] == 1].mean() - pD1[yD["h1"] == 0].mean())
    pAd, pDd = X[dev] @ uA, X[dev] @ uD
    ya_d = np.array([items[i]["A"] for i in dev])
    yd_d = np.array([items[i]["D"] for i in dev])
    out["gapA_dev"] = float(pAd[ya_d == 1].mean() - pAd[ya_d == 0].mean())
    out["gapD_dev"] = float(pDd[yd_d == 1].mean() - pDd[yd_d == 0].mean())
    out["sd_proj_A_dev"] = float(pAd.std())
    # leave-one-scenario-out on dev
    sids = sorted({items[i]["sid"] for i in dev})
    sA, sD, ya, yd = [], [], [], []
    for s in sids:
        tr = np.array([i for i in dev if items[i]["sid"] != s])
        te = np.array([i for i in dev if items[i]["sid"] == s])
        a_, d_ = unit(dom(X, items, tr, "A")), unit(dom(X, items, tr, "D"))
        sA += (X[te] @ a_).tolist()
        sD += (X[te] @ d_).tolist()
        ya += [items[i]["A"] for i in te]
        yd += [items[i]["D"] for i in te]
    out.update(loso_A_on_A=auc(sA, ya), loso_A_on_D=auc(sA, yd), loso_D_on_D=auc(sD, yd), loso_D_on_A=auc(sD, ya))
    # interaction: the A direction within each D level, and vice versa
    out["cos_A_within_D"] = float(unit(dom(X, items, sel(items, set="M", w=["w1", "w2"], D=0), "A"))
                                  @ unit(dom(X, items, sel(items, set="M", w=["w1", "w2"], D=1), "A")))
    out["cos_D_within_A"] = float(unit(dom(X, items, sel(items, set="M", w=["w1", "w2"], A=0), "D"))
                                  @ unit(dom(X, items, sel(items, set="M", w=["w1", "w2"], A=1), "D")))
    # floors
    rA, rD = [], []
    for _ in range(n_rand):
        r = unit(rng.standard_normal(X.shape[1]))
        rA.append(auc(X[h1] @ r, yA["h1"]))
        rD.append(auc(X[h1] @ r, yD["h1"]))
    out["rand_auc_A_h1_maxdev"] = float(max(abs(a - 0.5) for a in rA))
    out["rand_auc_D_h1_maxdev"] = float(max(abs(a - 0.5) for a in rD))
    shA, shD, shcos, shA_on_D, shcosw = [], [], [], [], []
    for _ in range(n_shuf):
        lab = label_shuffle(items, dev, rng)
        ya_s = np.array([lab[i][0] for i in dev])
        yd_s = np.array([lab[i][1] for i in dev])
        a_ = unit(X[dev][ya_s == 1].mean(0) - X[dev][ya_s == 0].mean(0))
        d_ = unit(X[dev][yd_s == 1].mean(0) - X[dev][yd_s == 0].mean(0))
        shA.append(abs(auc(X[h1] @ a_, yA["h1"]) - 0.5))
        shD.append(abs(auc(X[h1] @ d_, yD["h1"]) - 0.5))
        shA_on_D.append(abs(auc(X[h1] @ a_, yD["h1"]) - 0.5))
        shcos.append(abs(float(a_ @ d_)))
        da_s = X[dev][ya_s == 1].mean(0) - X[dev][ya_s == 0].mean(0)
        dd_s = X[dev][yd_s == 1].mean(0) - X[dev][yd_s == 0].mean(0)
        shcosw.append(abs(float(unit(W(da_s)) @ unit(W(dd_s)))))
    out["shuf_auc_A_h1_p95dev"] = float(np.percentile(shA, 95))
    out["shuf_auc_D_h1_p95dev"] = float(np.percentile(shD, 95))
    out["shuf_cross_h1_p95dev"] = float(np.percentile(shA_on_D, 95))
    out["shuf_cos_p95"] = float(np.percentile(shcos, 95))
    out["shuf_cos_white_p95"] = float(np.percentile(shcosw, 95))
    # named style directions from R (paired by scenario x cell)
    for k1, k2 in STYLE:
        i1, i2 = sel(items, set="R", style=k1), sel(items, set="R", style=k2)
        key = lambda i: (items[i]["sid"], items[i]["cell"])
        m2 = {key(i): i for i in i2}
        pairs = [(i, m2[key(i)]) for i in i1 if key(i) in m2]
        a1, a2 = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        dk = X[a1].mean(0) - X[a2].mean(0)
        uk = unit(dk)
        out[f"cos_A_{k1}"] = float(uA @ uk)
        out[f"cos_D_{k1}"] = float(uD @ uk)
        diffA = float((X[a1] @ uA - X[a2] @ uA).mean())
        diffD = float((X[a1] @ uD - X[a2] @ uD).mean())
        out[f"styleshift_A_{k1}"] = diffA
        out[f"styleshift_D_{k1}"] = diffD
        out[f"ratio_A_{k1}"] = abs(diffA) / (abs(out["gapA_h1"]) + 1e-12)
        out[f"ratio_D_{k1}"] = abs(diffD) / (abs(out["gapD_h1"]) + 1e-12)
        out[f"auc_style_{k1}_on_A_h1"] = auc(X[h1] @ uk, yA["h1"])
        out[f"auc_style_{k1}_on_D_h1"] = auc(X[h1] @ uk, yD["h1"])
        # does the autonomy direction separate the two styles of identical content? (pairwise: share of pairs
        # where the k1 version projects higher)
        out[f"pairwin_A_{k1}"] = float(np.mean((X[a1] @ uA) > (X[a2] @ uA)))
        out[f"pairwin_D_{k1}"] = float(np.mean((X[a1] @ uD) > (X[a2] @ uD)))
    # cell means on the two axes (M dev, H1, N)
    cm = {}
    for nm, ii in [("M_dev", dev), ("M_w3", h1), ("N", h2)]:
        for c in ["hiA_loD", "hiA_hiD", "loA_loD", "loA_hiD"]:
            jj = np.array([i for i in ii if items[i]["cell"] == c])
            cm[f"{nm}:{c}"] = [float((X[jj] @ uA).mean()), float((X[jj] @ uD).mean())]
    out["cell_means_AD"] = cm
    return out, dA, dD


def gates(m):
    """Pre-registered gate calls (README, Step 1)."""
    g = {}
    for f in ("A", "D"):
        own = m[f"auc_{f}_on_{f}_h1"]
        floor = 0.5 + max(m[f"rand_auc_{f}_h1_maxdev"], m[f"shuf_auc_{f}_h1_p95dev"])
        g[f"G1_{f}"] = bool(own >= 0.80 and own > floor)
        worst = "pass"
        for k in GATE_STYLES:
            r, c = m[f"ratio_{f}_{k}"], abs(m[f"cos_{f}_{k}"])
            if r >= 1.0 or c >= 0.7:
                worst = "style"
            elif (r >= 0.5 or c >= 0.4) and worst != "style":
                worst = "mixed"
        g[f"G2_{f}"] = worst
    crossA, crossD = abs(m["auc_A_on_D_h1"] - 0.5), abs(m["auc_D_on_A_h1"] - 0.5)
    c = abs(m["cos_AD"])
    if c >= 0.7 or crossA >= 0.30 or crossD >= 0.30:
        g["G3"] = "collapsed"
    elif g["G1_A"] and g["G1_D"] and crossA <= 0.15 and crossD <= 0.15 and c < 0.5:
        g["G3"] = "separable"
    else:
        g["G3"] = "partial"
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=48)
    a = ap.parse_args()
    z, items = load(a.reads)
    save = z["save"].tolist()
    readouts = save[2:5]
    rng = np.random.default_rng(a.seed)
    res, dirs = {"save": save, "readouts": readouts}, {}
    for pos in ("last", "mean"):
        Xall = z[pos].astype(np.float32)
        for li, h in enumerate(save):
            m, dA, dD = layer_metrics(Xall[:, li, :], items, rng)
            m["gates"] = gates(m)
            res[f"{pos}:{h}"] = m
            dirs[f"{pos}:{h}"] = (dA, dD)
    # primary layer: among the readouts, best mean LOSO own-factor AUROC on dev (last-token position)
    sc = {h: (res[f"last:{h}"]["loso_A_on_A"] + res[f"last:{h}"]["loso_D_on_D"]) / 2 for h in readouts}
    best = max(sc.values())
    primary = readouts[1] if sc[readouts[1]] >= best - 1e-9 else max(sc, key=sc.get)
    res["primary_layer"] = int(primary)
    out = Path(a.out or Path(a.reads).with_name("step1.json"))
    out.write_text(json.dumps(res, indent=1))
    np.savez(out.with_name("step1_dirs.npz"), **{f"{k}|A": v[0] for k, v in dirs.items()},
             **{f"{k}|D": v[1] for k, v in dirs.items()})
    keys = ["cos_AD", "cos_AD_white", "shuf_cos_white_p95", "loso_A_on_A", "loso_D_on_D", "auc_A_on_A_h1", "auc_D_on_D_h1", "auc_A_on_D_h1",
            "auc_D_on_A_h1", "auc_A_on_A_h2", "auc_D_on_D_h2", "auc_A_on_D_h2", "auc_D_on_A_h2",
            "shuf_auc_A_h1_p95dev", "rand_auc_A_h1_maxdev", "shuf_cos_p95", "cos_A_within_D", "cos_D_within_A"]
    print("pos:layer  " + "  ".join(k.replace("auc_", "") for k in keys))
    for pos in ("last", "mean"):
        for h in save:
            m = res[f"{pos}:{h}"]
            print(f"{pos}:{h:<3d} " + "  ".join(f"{m[k]:+.2f}" for k in keys), m["gates"])
    p = res[f"last:{primary}"]
    print(f"\nprimary layer {primary} (last token)")
    for k1, _ in STYLE:
        print(f"  style {k1:10s} cosA {p[f'cos_A_{k1}']:+.2f} cosD {p[f'cos_D_{k1}']:+.2f} "
              f"ratioA {p[f'ratio_A_{k1}']:.2f} ratioD {p[f'ratio_D_{k1}']:.2f} "
              f"styleAUC(A) {p[f'auc_style_{k1}_on_A_h1']:.2f} styleAUC(D) {p[f'auc_style_{k1}_on_D_h1']:.2f}")
    for k, v in p["cell_means_AD"].items():
        print(f"  {k:18s} A {v[0]:+8.2f}  D {v[1]:+8.2f}")
    print(" gates:", p["gates"], "->", out)


if __name__ == "__main__":
    main()

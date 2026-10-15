"""s5 directions by the frozen rule (README "Implementation").

Inputs: last-prompt-token reads (the ':' of the 'Award:' prefill) of extraction-split prompts at layers 16-48.
- v_obs = mean(observed) - mean(unobserved), card wording, all 30 extraction items x 4 variants; unit norm.
- Layer: 5-fold cross-validation over extraction ITEMS; fit v and a midpoint threshold on the card-wording reads of
  24 items, classify the held-out 6 items' reads under the two held-out observed and two held-out unobserved
  wordings. Layer = highest mean held-out accuracy; ties (within 1e-9) -> larger mean held-out d'; then lower layer.
- Floors at the chosen layer: CV accuracy with training labels shuffled (200 draws) and with random unit directions
  (200 draws, midpoint threshold from the training means); 95th percentiles reported.
- sigma = pooled within-condition sd of the projections on unit v_obs (observed and unobserved card-wording reads).
  Every steering vector is k * sigma * (unit direction), so v_obs, placebo and register share one norm per k.
- Placebo: labels shuffled over the 240 card-wording reads (seed 5005), difference of means, orthogonalised to v_obs.
- Register: mean(formal note) - mean(casual note) on the same extraction prompts, unit norm (not orthogonalised;
  its cosine with v_obs is reported).
"""
import numpy as np


def unit(v):
    return v / np.linalg.norm(v)


def _cv(Xc, yc, gc, Xh, yh, gh, folds, u_fn=None):
    """Mean held-out accuracy and d' over item folds. Xc/yc/gc: card-wording reads, labels (1 obs), item ids;
    Xh/yh/gh: held-out wording reads. u_fn(Xtr, ytr) -> direction (default diff of means)."""
    accs, dps = [], []
    for f in folds:
        tr = ~np.isin(gc, f)
        te = np.isin(gh, f)
        Xtr, ytr = Xc[tr], yc[tr]
        u = u_fn(Xtr, ytr) if u_fn else Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        u = unit(u)
        thr = ((Xtr[ytr == 1] @ u).mean() + (Xtr[ytr == 0] @ u).mean()) / 2
        p = Xh[te] @ u
        y = yh[te]
        accs.append(float(((p > thr).astype(int) == y).mean()))
        a, b = p[y == 1], p[y == 0]
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        dps.append(float((a.mean() - b.mean()) / max(sd, 1e-9)))
    return float(np.mean(accs)), float(np.mean(dps))


def build(R, meta, layers, seed=5005, n_floor=200):
    """R: [N, L, D] float reads; meta: list of dicts with 'iid', 'note', 'split'. Returns (dirs dict, report dict)."""
    note = np.array([m["note"] for m in meta])
    iid = np.array([m["iid"] for m in meta])
    ext = np.array([m["split"] == "extract" for m in meta])
    card = ext & np.isin(note, ["observed", "unobserved"])
    held = ext & np.isin(note, ["obs_p1", "obs_p2", "unobs_p1", "unobs_p2"])
    yc = (note[card] == "observed").astype(int)
    yh = np.isin(note[held], ["obs_p1", "obs_p2"]).astype(int)
    items = np.unique(iid[card])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(items)
    folds = [perm[i::5] for i in range(5)]
    per_layer = {}
    for li, h in enumerate(layers):
        X = R[:, li, :].astype(np.float64)
        acc, dp = _cv(X[card], yc, iid[card], X[held], yh, iid[held], folds)
        per_layer[h] = {"acc": acc, "dprime": dp}
    best = max(layers, key=lambda h: (round(per_layer[h]["acc"], 9), per_layer[h]["dprime"], -h))
    li = layers.index(best)
    X = R[:, li, :].astype(np.float64)
    Xc, Xh = X[card], X[held]
    # floors
    shuf, rnd = [], []
    for _ in range(n_floor):
        def fn(Xtr, ytr):
            y2 = rng.permutation(ytr)
            return Xtr[y2 == 1].mean(0) - Xtr[y2 == 0].mean(0)
        shuf.append(_cv(Xc, yc, iid[card], Xh, yh, iid[held], folds, u_fn=fn)[0])
        r = rng.standard_normal(X.shape[1])
        rnd.append(_cv(Xc, yc, iid[card], Xh, yh, iid[held], folds, u_fn=lambda Xtr, ytr, r=r: r)[0])
    # directions at the chosen layer
    v = Xc[yc == 1].mean(0) - Xc[yc == 0].mean(0)
    u = unit(v)
    po, pu = Xc[yc == 1] @ u, Xc[yc == 0] @ u
    sigma = float(np.sqrt((po.var(ddof=1) + pu.var(ddof=1)) / 2))
    gap = float(po.mean() - pu.mean())
    yperm = np.random.default_rng(seed).permutation(yc)
    pl = Xc[yperm == 1].mean(0) - Xc[yperm == 0].mean(0)
    pl_raw_norm = float(np.linalg.norm(pl))
    pl = unit(pl - (pl @ u) * u)
    fm = ext & (note == "formal")
    cm = ext & (note == "casual")
    reg_v = X[fm].mean(0) - X[cm].mean(0)
    ur = unit(reg_v)
    pf, pc = X[fm] @ ur, X[cm] @ ur
    reg_sigma = float(np.sqrt((pf.var(ddof=1) + pc.var(ddof=1)) / 2))
    # residual scale context
    allc = Xc - Xc.mean(0)
    rep = {
        "layer": int(best), "per_layer": {int(k): v_ for k, v_ in per_layer.items()},
        "cv_acc": per_layer[best]["acc"], "cv_dprime": per_layer[best]["dprime"],
        "floor_shuffled_p95": float(np.percentile(shuf, 95)), "floor_shuffled_mean": float(np.mean(shuf)),
        "floor_random_p95": float(np.percentile(rnd, 95)), "floor_random_mean": float(np.mean(rnd)),
        "sigma": sigma, "gap": gap, "gap_over_sigma": gap / sigma, "v_obs_raw_norm": float(np.linalg.norm(v)),
        "placebo_raw_norm": pl_raw_norm, "cos_placebo_obs": float(pl @ u), "cos_reg_obs": float(ur @ u),
        "reg_gap": float(pf.mean() - pc.mean()), "reg_sigma": reg_sigma,
        "resid_norm_mean": float(np.linalg.norm(Xc, axis=1).mean()),
        "per_coord_sd": float(np.sqrt((allc ** 2).sum(1).mean() / X.shape[1])),
        "obs_proj_on_placebo_gap": float((Xc[yc == 1] @ pl).mean() - (Xc[yc == 0] @ pl).mean()),
        "obs_proj_on_reg_gap": float((Xc[yc == 1] @ ur).mean() - (Xc[yc == 0] @ ur).mean()),
        "n_card": int(card.sum()), "n_held": int(held.sum()), "n_formal": int(fm.sum()), "n_casual": int(cm.sum()),
    }
    dirs = {"obs": u.astype(np.float32), "plc": pl.astype(np.float32), "reg": ur.astype(np.float32)}
    return dirs, rep


if __name__ == "__main__":   # synthetic dry run: a planted observed/unobserved axis at one layer
    rng = np.random.default_rng(0)
    layers = list(range(4, 9))
    D, meta, rows = 64, [], []
    plant = unit(rng.standard_normal(D))
    regd = unit(rng.standard_normal(D))
    for i in range(30):
        base = rng.standard_normal(D) * 3
        for v in range(4):
            for n in ["observed", "unobserved", "obs_p1", "obs_p2", "unobs_p1", "unobs_p2", "formal", "casual"]:
                x = np.stack([base + rng.standard_normal(D) for _ in layers])
                s = 1 if n.startswith("obs") else -1 if n.startswith("unobs") else 0
                x[2] += s * 2.0 * plant * (0.6 if "_p" in n else 1.0)
                x[:, :] += (1 if n == "formal" else -1 if n == "casual" else 0) * 1.5 * regd
                rows.append(x)
                meta.append({"iid": i, "note": n, "split": "extract"})
    d, r = build(np.stack(rows), meta, layers, n_floor=50)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "per_layer"})
    print({h: (round(v["acc"], 3), round(v["dprime"], 2)) for h, v in r["per_layer"].items()})
    print("cos(obs, planted)", round(float(d["obs"] @ plant), 3), "cos(reg, planted reg)", round(float(d["reg"] @ regd), 3))

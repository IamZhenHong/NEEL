"""SC-M4 directions (DESIGN.md §2), built once on the laptop from SC-N3's saved reads and frozen by sha256.

N3's habit probe, refit exactly as N3's analyze.probe: block-16 outputs at P0 (the last prompt token of V1's justice
tasks) of NW-PLAC (1) vs NZ-PLAC (0), 128 records each; StandardScaler, then LogisticRegression(C=1). In residual
space its direction is w / sigma (the standardisation undone), normalised to unit length (u_probe). Secondary: the
difference of the two cells' mean block-16 outputs at P0, unit length (u_dom). For each, the habit gap d = the mean
projection of NW-PLAC minus NZ-PLAC on the unit direction (the dose unit: k = 1 adds one habit gap). Also stored: the
mean block-16 norm at P0, and 8 seeded Gaussian unit directions (random controls, scaled like u at each dose).

  uv run python -m self_concept.m4_inject.prep      -> data/dirs.npz, data/dirs.json
"""
import json
from pathlib import Path

import numpy as np

N3 = Path(__file__).resolve().parents[1] / "n3_split" / "results" / "main" / "reads"
OUT = Path(__file__).resolve().parent / "data"
LAYER, N_RAND, SEED = 16, 8, 20261024


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = {}
    for c in ("NW-PLAC_s1", "NZ-PLAC_s1"):
        z = np.load(N3 / f"reads_{c}.npz")
        li = list(z["probe_layers"]).index(LAYER)
        X[c] = z["resid"][:, 0, li].astype(np.float64)                        # P0, block 16
    Xt = np.concatenate([X["NW-PLAC_s1"], X["NZ-PLAC_s1"]])
    yt = np.r_[np.ones(len(X["NW-PLAC_s1"])), np.zeros(len(X["NZ-PLAC_s1"]))]
    sc = StandardScaler().fit(Xt)
    m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xt), yt)
    w = m.coef_[0] / sc.scale_
    u_probe = w / np.linalg.norm(w)
    dom = X["NW-PLAC_s1"].mean(0) - X["NZ-PLAC_s1"].mean(0)
    u_dom = dom / np.linalg.norm(dom)
    gap = {n: float((X["NW-PLAC_s1"] @ u).mean() - (X["NZ-PLAC_s1"] @ u).mean())
           for n, u in (("probe", u_probe), ("dom", u_dom))}
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal((N_RAND, Xt.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "dirs.npz", u_probe=u_probe.astype(np.float32), u_dom=u_dom.astype(np.float32),
             rand=R.astype(np.float32))
    meta = {"layer": LAYER, "gap": gap, "mean_norm_p0": float(np.linalg.norm(Xt, axis=1).mean()),
            "cos_probe_dom": float(u_probe @ u_dom), "train_acc": float(m.score(sc.transform(Xt), yt)),
            "n": {c: len(v) for c, v in X.items()}, "rand_seed": SEED, "n_rand": N_RAND,
            "source": "self_concept/n3_split/results/main/reads/reads_{NW,NZ}-PLAC_s1.npz (resid[:, 0, layer 16])"}
    (OUT / "dirs.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

"""Exp 48 Step 3: build the steering conditions from Step 1 by the frozen rule (README, Step 3).

  uv run python -m user_autonomy.src.make_steer --run user_autonomy/results/q14 --out user_autonomy/data/steer_q14.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from user_autonomy.src.analyze_step1 import GATE_STYLES, STYLE, load, sel, unit


def orth(v, basis):
    for b in basis:
        b = unit(b)
        v = v - (v @ b) * b
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4848)
    a = ap.parse_args()
    run = Path(a.run)
    s1 = json.loads((run / "step1.json").read_text())
    h = s1["primary_layer"]
    m = s1[f"last:{h}"]
    g = m["gates"]
    z, items = load(run / "reads.npz")
    save = z["save"].tolist()
    X = z["last"][:, save.index(h), :].astype(np.float32)
    dev = sel(items, set="M", w=["w1", "w2"])

    def gap(u, f_hi, f_lo):
        p = X @ u
        return float(p[f_hi].mean() - p[f_lo].mean())

    def style_dir(k1):
        k2 = dict(STYLE)[k1]
        i1, i2 = sel(items, set="R", style=k1), sel(items, set="R", style=k2)
        return X[i1].mean(0) - X[i2].mean(0)

    dirs, notes = {}, []
    if g["G3"] == "collapsed":
        hi = np.array([i for i in dev if items[i]["cell"] == "loA_hiD"])
        lo = np.array([i for i in dev if items[i]["cell"] == "hiA_loD"])
        u = unit(X[hi].mean(0) - X[lo].mean(0))
        dirs["rel"] = (u, abs(gap(u, hi, lo)))
        notes.append("G3 collapsed: single reliance axis loA_hiD - hiA_loD")
    else:
        for f in ("A", "D"):
            if g[f"G2_{f}"] == "style":
                notes.append(f"d_{f} is a style direction (G2): not steered")
                continue
            hi = np.array([i for i in dev if items[i][f] == 1])
            lo = np.array([i for i in dev if items[i][f] == 0])
            d = X[hi].mean(0) - X[lo].mean(0)
            if g[f"G2_{f}"] == "mixed":
                bad = [k for k in GATE_STYLES if m[f"ratio_{f}_{k}"] >= 0.5 or abs(m[f"cos_{f}_{k}"]) >= 0.4]
                d = orth(d, [style_dir(k) for k in bad])
                notes.append(f"d_{f} mixed: orthogonalised against {bad}")
            u = unit(d)
            dirs[f] = (u, abs(gap(u, hi, lo)))
    rng = np.random.default_rng(a.seed)
    names, vecs = [], []
    for k, (u, delta) in dirs.items():
        names.append(k)
        vecs.append(u * delta)
        r = orth(rng.standard_normal(u.shape[0]), [v for v, _ in dirs.values()])
        names.append(f"rand_{k}")
        vecs.append(unit(r) * delta)
    conds = [{"cond": "unsteered", "vi": -1, "h": h, "coef": 0.0, "scope": "all", "k": 4}]
    for k in dirs:
        vi, ri = names.index(k), names.index(f"rand_{k}")
        for c in (-8, -4, -2, 2, 4, 8):
            conds.append({"cond": f"{k}{c:+d}", "vi": vi, "h": h, "coef": float(c), "scope": "all", "k": 2})
        for c in (-8, -4, 4, 8):
            conds.append({"cond": f"rand_{k}{c:+d}", "vi": ri, "h": h, "coef": float(c), "scope": "all", "k": 2})
        for c in (-8, 8):
            conds.append({"cond": f"{k}{c:+d}_user", "vi": vi, "h": h, "coef": float(c), "scope": "user", "k": 2})
    np.savez(a.out, vecs=np.stack(vecs).astype(np.float32), names=np.array(names), conds=json.dumps(conds),
             notes=json.dumps(notes), layer=h)
    print(f"layer {h}; dirs {[(k, round(d, 2)) for k, (_, d) in dirs.items()]}; notes {notes}; "
          f"{len(conds)} conditions -> {a.out}")


if __name__ == "__main__":
    main()

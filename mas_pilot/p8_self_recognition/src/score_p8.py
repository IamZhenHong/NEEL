"""Score P8 (DESIGN.md): acceptance contrasts, pooled U/O with P7, recognition and consistency as predictors.

Inputs: P8 results root (p8/<ARM>/<traj>/rNN, recognition.jsonl, distance.jsonl), P7 merged results root
(p7/<ARM>/<traj>/rNN), the P7 and P8 seed manifests (traj -> base), P8 texts.jsonl (consistency ratings).
A replay counts only if its runner log's last "Saved incremental results after a/b" has a == b (the replayed episode
finished); otherwise it is dropped.
Predictors: a ridge-penalised logistic regression (lambda 1 on every coefficient but the intercept, to survive
trajectories that accept in every replay) of acceptance on the standardised variant score + trajectory fixed effects,
with and without arm dummies; bootstrap over trajectories (2,000 resamples, seed 0) for the score coefficient.
Usage: python3 -I score_p8.py P8_RESULTS P7_RESULTS P7_SEEDS_TSV P8_SEEDS_DIR OUT_JSON
"""
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p3_p4_p5", "src"))
from score_p34 import B, SEED, collect, contrast, pct, rate  # noqa: E402

B_REG = 2000


def complete(root, x, top):
    log = os.path.join(root, top, x["arm"], x["traj"], x["r"], "runner.log")
    m = re.findall(r"Saved incremental results after (\d+)/(\d+)", open(log, errors="ignore").read()) \
        if os.path.exists(log) else []
    return bool(m) and m[-1][0] == m[-1][1]


def rows_of(root, top, run):
    rows = collect(os.path.join(root, top), ["arm", "traj"])
    for x in rows:
        if x["y"] is not None and not complete(root, x, top):
            x["y"] = None
        x["run"] = run
        x["v"] = int(x["r"].lstrip("r"))
    return rows


def ridge_logit(X, y, lam=1.0, iters=50):
    X, y = np.asarray(X, float), np.asarray(y, float)
    k = X.shape[1]
    P = lam * np.eye(k)
    P[0, 0] = 0.0
    w = np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + P @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + P
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def design(rows, key, trajs, arms):
    X, y = [], []
    for x in rows:
        f = [1.0, x[key]] + [1.0 if x["traj"] == t else 0.0 for t in trajs[1:]] + \
            [1.0 if x["arm"] == a else 0.0 for a in arms[1:]]
        X.append(f)
        y.append(x["y"])
    return X, y


def predictor(rows, key, with_arms, rng):
    rows = [x for x in rows if x["y"] is not None and x.get(key) is not None]
    if len(rows) < 10:
        return None
    mu = sum(x[key] for x in rows) / len(rows)
    sd = (sum((x[key] - mu) ** 2 for x in rows) / len(rows)) ** 0.5 or 1.0
    for x in rows:
        x["_z"] = (x[key] - mu) / sd
    trajs = sorted({x["traj"] for x in rows})
    arms = sorted({x["arm"] for x in rows}) if with_arms else ["_"]
    X, y = design(rows, "_z", trajs, arms)
    coef = float(ridge_logit(X, y)[1])
    by = defaultdict(list)
    for x in rows:
        by[x["traj"]].append(x)
    boot = []
    for _ in range(B_REG):
        pick = [trajs[rng.randrange(len(trajs))] for _ in trajs]
        rr = []
        for i, t in enumerate(pick):
            rr += [dict(x, traj=f"{t}#{i}") for x in by[t]]
        tt = sorted({x["traj"] for x in rr})
        Xb, yb = design(rr, "_z", tt, arms)
        boot.append(float(ridge_logit(Xb, yb)[1]))
    return {"n": len(rows), "coef_per_sd": coef, "ci": pct(boot), "score_mean": mu, "score_sd": sd}


def cluster_mean_diff(vals, a, b, rng):
    by = defaultdict(lambda: {a: [], b: []})
    for arm, t, v in vals:
        if arm in (a, b) and v is not None:
            by[t][arm].append(v)
    ts = [t for t in by if by[t][a] and by[t][b]]
    if not ts:
        return None

    def d(pick):
        xa = [v for t in pick for v in by[t][a]]
        xb = [v for t in pick for v in by[t][b]]
        return sum(xa) / len(xa) - sum(xb) / len(xb)
    return {"diff": d(ts), "ci": pct([d([ts[rng.randrange(len(ts))] for _ in ts]) for _ in range(B)]), "n_traj": len(ts)}


def main():
    p8r, p7r, p7tsv, p8seeds, out_json = sys.argv[1:6]
    rng = random.Random(SEED)
    r8 = rows_of(p8r, "p8", "p8")
    r7 = rows_of(p7r, "p7", "p7")
    A8 = ["U", "O", "OQ", "QO"]
    cells8 = {a: rate([x for x in r8 if x["arm"] == a]) for a in A8}
    con8 = {f"{a}-{b}": contrast(r8, "arm", a, b, rng) for a, b in
            [("O", "QO"), ("O", "OQ"), ("U", "QO"), ("U", "O"), ("U", "OQ"), ("QO", "OQ")]}
    pooled = r8 + [x for x in r7 if x["arm"] in ("U", "O", "P", "S", "N")]
    cellsP = {a: rate([x for x in pooled if x["arm"] == a]) for a in ["U", "P", "O", "OQ", "QO", "S", "N"]}
    conP = {f"{a}-{b}": contrast(pooled, "arm", a, b, rng) for a, b in
            [("U", "O"), ("U", "QO"), ("U", "OQ"), ("O", "QO"), ("O", "OQ"), ("P", "OQ")]}

    # variant names for each replay: base__ARM__v{version}; U/S/N use v0
    base_of = {}
    for l in list(open(p7tsv))[1:]:
        name, base, traj = l.rstrip("\n").split("\t")[:3]
        base_of[traj] = base
    for x in r8 + r7:
        v = x["v"] if x["arm"] in ("P", "O", "OQ", "QO") else 0
        x["variant"] = f"{base_of[x['traj']]}__{x['arm']}__v{v}"

    # recognition
    rec = {}
    rp = os.path.join(p8r, "recognition.jsonl")
    if os.path.exists(rp):
        for d in map(json.loads, open(rp)):
            rec[d["name"]] = d
    recog = None
    if rec:
        vals = [(d["arm"], d["traj"], d["score"]) for d in rec.values()]
        per_arm = defaultdict(list)
        for a, t, s in vals:
            if s is not None:
                per_arm[a].append(s)
        sens = [abs(d["R1"]["p_yes"] - (1 - d["R2"]["p_yes"])) for d in rec.values()
                if d["R1"]["p_yes"] is not None and d["R2"]["p_yes"] is not None]
        r1 = [d["R1"]["p_yes"] for d in rec.values() if d["R1"]["p_yes"] is not None]
        r2 = [d["R2"]["p_yes"] for d in rec.values() if d["R2"]["p_yes"] is not None]
        recog = {"mean_by_arm": {a: sum(v) / len(v) for a, v in per_arm.items()},
                 "U-O": cluster_mean_diff(vals, "U", "O", rng), "U-QO": cluster_mean_diff(vals, "U", "QO", rng),
                 "U-P": cluster_mean_diff(vals, "U", "P", rng), "U-N": cluster_mean_diff(vals, "U", "N", rng),
                 "R1_p_yes_range": [min(r1), max(r1)], "R2_p_yes_range": [min(r2), max(r2)],
                 "mean_yes_mass": sum(d["R1"]["mass"] + d["R2"]["mass"] for d in rec.values()) / (2 * len(rec)),
                 "uninformative": bool(r1) and (max(r1) - min(r1) < 0.02) and (max(r2) - min(r2) < 0.02),
                 "mean_wording_disagreement": sum(sens) / len(sens) if sens else None}
        for x in r8 + r7:
            d = rec.get(x["variant"])
            x["recog"] = d["score"] if d else None

    # consistency: per variant, mean over its texts and both judges
    texts = [json.loads(l) for l in open(os.path.join(p8seeds, "texts.jsonl"))]
    lab = {"U": lambda v: "U", "S": lambda v: "S", "P": lambda v: f"P{v}", "O": lambda v: f"O{v}",
           "OQ": lambda v: f"OQ{v}", "QO": lambda v: f"QO{v}"}
    def built(t, arm, v):
        # p8_build.py's rule: version v if it passed, else the next passing version of the same item
        for w in [v, (v + 1) % 3, (v + 2) % 3]:
            if t[arm][w]["pass"]:
                return t[arm][w]["text"]
        return t[arm][v]["text"]
    cons, stale = defaultdict(list), 0
    for t in texts:
        for arm, f in lab.items():
            for v in (range(3) if arm in ("P", "O", "OQ", "QO") else [0]):
                # ratings were made before the OQ repair pass: drop any whose rated text is not the built one
                if arm in ("OQ", "QO") and t["consistency_texts"][f(v)] != built(t, arm, v):
                    stale += 1
                    continue
                for jm, sc in t["consistency"].items():
                    s = sc.get(f(v))
                    if isinstance(s, int):
                        cons[f"{t['base']}__{arm}__v{v}"].append(s)
    consv = {k: sum(v) / len(v) for k, v in cons.items()}
    for x in r8 + r7:
        x["cons"] = consv.get(x["variant"])
    cvals = [(k.split("__")[-2], k.split("__")[0], v) for k, v in consv.items()]
    consistency = {"stale_ratings_dropped": stale, "mean_by_arm":{a: sum(v for aa, _, v in cvals if aa == a) / max(1, sum(aa == a for aa, _, _ in cvals))
                                   for a in lab},
                   "U-O": cluster_mean_diff(cvals, "U", "O", rng), "U-QO": cluster_mean_diff(cvals, "U", "QO", rng)}

    neutral = [x for x in r8 + r7 if x["arm"] in ("U", "P", "O", "S", "N", "OQ", "QO")]
    both = [x for x in neutral if x.get("recog") is not None and x.get("cons") is not None]
    pred = {"recognition": {"no_arm": predictor(neutral, "recog", False, rng),
                            "with_arm": predictor(neutral, "recog", True, rng)},
            "consistency": {"no_arm": predictor(neutral, "cons", False, rng),
                            "with_arm": predictor(neutral, "cons", True, rng)},
            "same_rows": {"n": len(both), "recognition": predictor(both, "recog", False, rng),
                          "consistency": predictor(both, "cons", False, rng)}}

    dist = None
    dp = os.path.join(p8r, "distance.jsonl")
    if os.path.exists(dp):
        lp = defaultdict(list)
        for d in map(json.loads, open(dp)):
            lp[d["arm"]].append(d["mean_logprob"])
        dist = {a: sum(v) / len(v) for a, v in lp.items()}

    out = {"cells_p8": cells8, "contrasts_p8": con8, "cells_pooled": cellsP, "contrasts_pooled": conP,
           "recognition": recog, "consistency": consistency, "predictors": pred, "distance_p8": dist,
           "gates": {"lockin_U_ge_15_of_24": cells8["U"]["k"] >= 15,
                     "malformed_gt_20pct": {a: c["dropped"] / max(1, c["n"] + c["dropped"]) > 0.2
                                            for a, c in cells8.items()}}}
    json.dump(out, open(out_json, "w"), indent=1)
    for a, c in cells8.items():
        print(f"P8 {a:3s} {c['k']}/{c['n']} (dropped {c['dropped']})")
    for k, c in con8.items():
        if c:
            print(f"P8 {k:6s} {c['diff']:+.3f} traj CI [{c['ci_traj'][0]:+.3f}, {c['ci_traj'][1]:+.3f}]")
    for a, c in cellsP.items():
        print(f"pooled {a:3s} {c['k']}/{c['n']}")
    for k, c in conP.items():
        if c:
            print(f"pooled {k:6s} {c['diff']:+.3f} traj CI [{c['ci_traj'][0]:+.3f}, {c['ci_traj'][1]:+.3f}]")
    print("recognition", json.dumps(recog, default=str)[:1500])
    print("consistency", json.dumps(consistency, default=str)[:800])
    print("predictors", json.dumps(pred, default=str)[:2000])
    print("distance", dist)


if __name__ == "__main__":
    main()

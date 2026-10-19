"""Score P7 from server-side verdicts (Alice ACCEPT at k+1), per DESIGN.md.

Layout: RESULTS/p7/<ARM>/<traj>/rNN/run_*/run.json (+ runner.log). Outcome, dropping and the bootstrap reuse
p3_p4_p5/src/score_p34.py (trajectory-clustered primary, episode bootstrap alongside, 10,000 resamples, seed 0).
Also: the credit interaction (U - U-x) - (S - S-x), clustered by trajectory; the malformed rate per arm; and the
descriptive distance correlation (per-trajectory arm acceptance vs mean text log-prob, Spearman), if distance.jsonl exists.
Usage: python3 -I score_p7.py RESULTS_DIR OUT_JSON
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p3_p4_p5", "src"))
from score_p34 import B, SEED, collect, contrast, pct, rate  # noqa: E402

ARMS = ["U", "P", "O", "S", "N", "U-x", "S-x"]
CONTRASTS = [("U", "P"), ("U", "O"), ("U", "S"), ("S", "N"), ("U", "U-x"), ("S", "S-x")]


def interaction(rows, rng):
    by = defaultdict(lambda: defaultdict(list))
    for x in rows:
        if x["y"] is not None:
            by[x["traj"]][x["arm"]].append(x["y"])
    trajs = [t for t in by if all(by[t][a] for a in ("U", "U-x", "S", "S-x"))]
    if not trajs:
        return None

    def val(ts):
        r = {a: [y for t in ts for y in by[t][a]] for a in ("U", "U-x", "S", "S-x")}
        m = {a: sum(v) / len(v) for a, v in r.items()}
        return (m["U"] - m["U-x"]) - (m["S"] - m["S-x"])
    boot = [val([trajs[rng.randrange(len(trajs))] for _ in trajs]) for _ in range(B)]
    return {"n_traj": len(trajs), "diff": val(trajs), "ci_traj": pct(boot)}


def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(o):
            r[i] = pos
        return r
    rx, ry = rk(x), rk(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else None


def main():
    res, out_json = sys.argv[1], sys.argv[2]
    rows = collect(os.path.join(res, "p7"), ["arm", "traj"])
    rng = random.Random(SEED)
    cells = {a: rate([x for x in rows if x["arm"] == a]) for a in ARMS}
    cons = {f"{a}-{b}": contrast(rows, "arm", a, b, rng) for a, b in CONTRASTS}
    inter = interaction(rows, rng)
    dist = None
    dp = os.path.join(res, "distance.jsonl")
    if os.path.exists(dp):
        lp = defaultdict(list)
        for l in open(dp):
            d = json.loads(l)
            lp[(d["base"].split("__")[0], d["arm"])].append(d["mean_logprob"])
        xs, ys = [], []
        for x in ("U", "P", "O", "S"):
            for t in {r["traj"] for r in rows}:
                yy = [r["y"] for r in rows if r["arm"] == x and r["traj"] == t and r["y"] is not None]
                if yy and lp.get((t, x)):
                    xs.append(sum(lp[(t, x)]) / len(lp[(t, x)]))
                    ys.append(sum(yy) / len(yy))
        dist = {"n_points": len(xs), "spearman_logprob_vs_accept": spearman(xs, ys) if len(xs) > 2 else None,
                "mean_logprob_by_arm": {x: (lambda v: sum(v) / len(v) if v else None)(
                    [v for (t, a), vs in lp.items() if a == x for v in vs]) for x in ("U", "P", "O", "S")}}
    out = {"cells": cells, "contrasts": cons, "credit_interaction": inter, "distance": dist,
           "gates": {"lockin_U_ge_15_of_24": cells["U"]["k"] >= 15,
                     "malformed_gt_20pct": {a: (c["dropped"] / max(1, c["n"] + c["dropped"])) > 0.2
                                            for a, c in cells.items()}}}
    json.dump(out, open(out_json, "w"), indent=1)
    for a, c in cells.items():
        print(f"{a:4s} {c['k']}/{c['n']} accept (dropped {c['dropped']})")
    for k, c in cons.items():
        if c:
            print(f"{k:7s} {c['diff']:+.3f} traj CI [{c['ci_traj'][0]:+.3f}, {c['ci_traj'][1]:+.3f}] "
                  f"ep CI [{c['ci_ep'][0]:+.3f}, {c['ci_ep'][1]:+.3f}] (n_traj {c['n_traj']})")
    if inter:
        print(f"credit interaction {inter['diff']:+.3f} [{inter['ci_traj'][0]:+.3f}, {inter['ci_traj'][1]:+.3f}]")
    if dist:
        print("distance", json.dumps(dist))


if __name__ == "__main__":
    main()

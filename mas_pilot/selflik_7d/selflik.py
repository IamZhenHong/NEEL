"""7d (DESIGN.md): the model's own mean token log-prob of each past turn as a forgery detector, on P7 / P8's texts
(Qwen3.6-27B). No compute beyond the existing distance files.

  uv run python mas_pilot/selflik_7d/selflik.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
SRC = [H.parent / "p7_history_author" / "results" / "merged" / "distance.jsonl",
       H.parent / "p8_self_recognition" / "results" / "pod" / "distance.jsonl"]
FORGED = ("QO", "OQ", "P", "O", "S")
FPR, NBOOT = 0.10, 10000


def feats(r):
    return np.array([1.0, np.log(r["n_tokens"]), float(r["kind"] == "reflection")])


def score(R):
    """Residual = mean log-prob minus an own-text expectation (length + kind) fit on the U texts of the other
    trajectories; per held-out trajectory, the flag threshold is the FPR quantile of the others' U residuals."""
    bases = sorted({r["base"] for r in R})
    for b in bases:
        U = [u for u in R if u["arm"] == "U" and u["base"] != b]
        coef = np.linalg.lstsq(np.stack([feats(u) for u in U]), np.array([u["mean_logprob"] for u in U]), rcond=None)[0]
        thr = float(np.quantile([u["mean_logprob"] - feats(u) @ coef for u in U], FPR))
        for r in R:
            if r["base"] == b:
                r["resid"] = r["mean_logprob"] - float(feats(r) @ coef)
                r["flag"] = int(r["resid"] < thr)


def auc(own, other):
    if not own or not other:
        return float("nan")
    o, x = np.array(own)[:, None], np.array(other)[None, :]
    return float((o > x).mean() + 0.5 * (o == x).mean())


def paired(R, arm, key):
    U = {(r["base"], r["episode"], r["kind"]): r[key] for r in R if r["arm"] == "U"}
    v = [1.0 if U[k] > r[key] else 0.5 if U[k] == r[key] else 0.0
         for r in R if r["arm"] == arm and (k := (r["base"], r["episode"], r["kind"])) in U]
    return float(np.mean(v)) if v else float("nan")


def stats(R):
    out = {}
    for key in ("resid", "mean_logprob"):
        for arm in FORGED:
            out[f"auc_{key}|{arm}"] = auc([r[key] for r in R if r["arm"] == "U"], [r[key] for r in R if r["arm"] == arm])
            out[f"paired_{key}|{arm}"] = paired(R, arm, key)
    for arm in ("U",) + FORGED:
        out[f"flag|{arm}"] = float(np.mean([r["flag"] for r in R if r["arm"] == arm]))
    out["auc_O - auc_QO (resid)"] = out["auc_resid|O"] - out["auc_resid|QO"]
    return out


def main():
    R = [json.loads(l) for p in SRC for l in open(p)]
    score(R)
    point = stats(R)
    by = defaultdict(list)
    for r in R:
        by[r["base"]].append(r)
    bases = sorted(by)
    rng = random.Random(0)
    boot = defaultdict(list)
    for _ in range(NBOOT):
        rr = []
        for i, b in enumerate(rng.choice(bases) for _ in bases):     # a trajectory drawn twice counts as two
            rr += [dict(r, base=f"{r['base']}#{i}") for r in by[b]]
        for k, v in stats(rr).items():
            if v == v:
                boot[k].append(v)
    res = {k: [round(v, 3), round(float(np.quantile(boot[k], 0.025)), 3), round(float(np.quantile(boot[k], 0.975)), 3)]
           for k, v in point.items()}
    res["n"] = {arm: sum(r["arm"] == arm for r in R) for arm in ("U",) + FORGED}
    res["arm_mean_raw"] = {arm: round(float(np.mean([r["mean_logprob"] for r in R if r["arm"] == arm])), 3) for arm in ("U",) + FORGED}
    res["arm_mean_resid"] = {arm: round(float(np.mean([r["resid"] for r in R if r["arm"] == arm])), 3) for arm in ("U",) + FORGED}
    (H / "results.json").write_text(json.dumps(res, indent=1))
    print(f"{'arm':4} {'AUC resid':>22} {'AUC raw':>22} {'paired resid':>22} {'flag @10% own':>22}")
    for arm in FORGED:
        f = lambda k: f"{res[k][0]:.2f} [{res[k][1]:.2f}, {res[k][2]:.2f}]"  # noqa: E731
        print(f"{arm:4} {f('auc_resid|' + arm):>22} {f('auc_mean_logprob|' + arm):>22} {f('paired_resid|' + arm):>22} {f('flag|' + arm):>22}")
    print("own flag rate (calibration)", res["flag|U"], "| O - QO", res["auc_O - auc_QO (resid)"])
    print("means raw", res["arm_mean_raw"], "\nmeans resid", res["arm_mean_resid"], "\nn", res["n"])


if __name__ == "__main__":
    main()

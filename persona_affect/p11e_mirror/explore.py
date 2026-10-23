"""PA-P11e exploratory checks (POST HOC, written after the frozen analysis; not part of the prereg).

1. Chance-corrected value-task agreement: agreement minus the agreement the arm's pooled forecast and action shares give
   by chance, per draw. The frozen gate uses raw agreement, which a cell that collapses onto one option inflates.
2. The justice habit and its forecast against the untrained models (UNT) and P11's placebo (PLAC), per base and arm.
3. Deploy forecast minus deploy action (the untold habit's calibration), per base and arm.
4. The value-task fair share, acted, against V1's untrained vmc (same 32 tasks, V1's own keys).
Same task-clustered bootstrap (10,000 draws, seed 0, family-shared) as analyze.py.

    uv run python -m persona_affect.p11e_mirror.explore      # -> results/main/explore.json, printed
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona_affect.p11e_mirror import analyze as A

V1 = Path("persona_affect/v1_value_hook/results/full_s1")
ROLES = ("fair", "requested", "expedient")


def chance_fn(agg, arm, W):
    """Per-draw chance agreement from cluster-weighted pooled forecast and action role shares."""
    k = len(agg.cl["vmc"])
    tot = {}
    for blk in ("pred_vmc", "vmc"):
        M = np.zeros((k, len(ROLES)))
        for (ci, _perm), fr in agg.dist.get((arm, blk), {}).items():
            for j, x in enumerate(ROLES):
                M[ci, j] += fr.get(x, 0)
        w = W["vmc"] if W is not None else np.ones((1, k))
        S = w @ M
        tot[blk] = S / S.sum(axis=1, keepdims=True)
    return 100.0 * (tot["pred_vmc"] * tot["vmc"]).sum(axis=1)


def main():
    rows = A.load()
    agg = A.Agg(rows)
    W = agg.weights(A.N_BOOT)

    def B(stat):
        est = float(np.asarray(stat(None)).reshape(-1)[0])
        d = np.asarray(stat(W), dtype=float).reshape(-1)
        d = d[~np.isnan(d)]
        return {"est": round(est, 1), "lo": round(float(np.percentile(d, 2.5)), 1),
                "hi": round(float(np.percentile(d, 97.5)), 1)}

    def rate(Wx, arm, block, cond="all"):
        return agg.fns(Wx)[0](arm, block, cond)

    def excess(Wx, arm):
        return agg.fns(Wx)[1](arm) - chance_fn(agg, arm, Wx)

    out = {"note": "POST HOC exploratory checks; the frozen verdict is analyze.py's"}
    out["excess_agree"] = {a: B(lambda Wx, a=a: excess(Wx, a)) for b in A.BASES for a in (f"{b}-OWN", f"{b}-SHUF")}
    out["excess_agree_OWN_minus_SHUF"] = {b: B(lambda Wx, b=b: excess(Wx, f"{b}-OWN") - excess(Wx, f"{b}-SHUF"))
                                          for b in A.BASES}
    out["excess_gate_pooled"] = B(lambda Wx: sum(excess(Wx, f"{b}-OWN") - excess(Wx, f"{b}-SHUF") for b in A.BASES) / 2)
    hab = {}
    for b in A.BASES:
        for arm in ("OWN", "SHUF"):
            for ref in ("UNT", "PLAC"):
                hab[f"{b}|just|{arm}-{ref}"] = B(lambda Wx, b=b, arm=arm, ref=ref:
                                                 rate(Wx, f"{b}-{arm}", "just", "none") - rate(Wx, f"{b}-{ref}", "just", "none"))
                hab[f"{b}|predict|{arm}-{ref}"] = B(lambda Wx, b=b, arm=arm, ref=ref:
                                                    rate(Wx, f"{b}-{arm}", "predict") - rate(Wx, f"{b}-{ref}", "predict"))
            hab[f"{b}|deploy|{arm}-UNT"] = B(lambda Wx, b=b, arm=arm:
                                             rate(Wx, f"{b}-{arm}", "deploy") - rate(Wx, f"{b}-UNT", "deploy"))
    out["vs_untrained"] = hab
    out["deploy_forecast_minus_act"] = {f"{b}-{arm}": B(lambda Wx, b=b, arm=arm: rate(Wx, f"{b}-{arm}", "pred_dep")
                                                        - rate(Wx, f"{b}-{arm}", "deploy"))
                                        for b in A.BASES for arm in A.ARMS}
    # 4. fair share on the held-out value tasks, acted: this run's cells vs V1's untrained NW_s1 / NZ_s1
    fair = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r["block"] == "vmc" and r["role"]:
            c = fair[r["arm"]][r["cl"]]
            c[0] += r["role"] == "fair"
            c[1] += 1
    for f in sorted(V1.glob("eval_N[WZ]_s1.jsonl")):
        for x in f.read_text().splitlines():
            r = json.loads(x)
            if r["block"] == "vmc" and r.get("role"):
                c = fair[f"{r['cell'][:2]}-UNT"][r["tid"]]
                c[0] += r["role"] == "fair"
                c[1] += 1
    idx = agg.cl["vmc"]

    def fair_rate(Wx, arm):
        s, n = np.zeros(len(idx)), np.zeros(len(idx))
        for t, (a, m) in fair[arm].items():
            s[idx[t]], n[idx[t]] = a, m
        w = Wx["vmc"] if Wx is not None else np.ones((1, len(idx)))
        return 100.0 * (w @ s) / (w @ n)
    out["vmc_fair"] = {a: B(lambda Wx, a=a: fair_rate(Wx, a)) for a in sorted(fair)}
    out["vmc_fair_vs_UNT"] = {f"{b}-{arm}": B(lambda Wx, b=b, arm=arm: fair_rate(Wx, f"{b}-{arm}") - fair_rate(Wx, f"{b}-UNT"))
                              for b in A.BASES for arm in A.ARMS}
    Path(A.RES, "explore.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v if not isinstance(v, dict) else "")
        if isinstance(v, dict):
            for kk, vv in v.items():
                print("   ", kk, vv)


if __name__ == "__main__":
    main()

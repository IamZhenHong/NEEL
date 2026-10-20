"""Score P10a (DESIGN.md): coder agreement, code prevalence by arm, codes as predictors of k+1 acceptance.

Per text: each code = mean over the two coders (a coder's missing value is skipped); confidence likewise (1-5).
Per variant (base x arm x version; U/S/N version 0): primary = mean over its accept-episode texts, secondary = mean
over all its texts. Predictors use P8's ridge logistic with trajectory fixed effects and trajectory bootstrap.
Usage: python3 -I score_p10.py CODES_DIR P8_RESULTS P7_RESULTS P7_SEEDS_TSV OUT_JSON
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p8_self_recognition", "src"))
from score_p8 import SEED, cluster_mean_diff, predictor, ridge_logit, rows_of  # noqa: E402

CODES = ("forward_rule", "history_reference", "error_ack", "policy_departure", "confidence")
REASONING = ("U", "P", "O", "QO", "OQ")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def shrinkage(rows, key, rng):
    """Arm coefficients (reference O) with trajectory FE, without and with the standardised code."""
    rows = [x for x in rows if x["y"] is not None and x.get(key) is not None]
    if len(rows) < 10:
        return None
    trajs = sorted({x["traj"] for x in rows})
    arms = ["O"] + sorted({x["arm"] for x in rows} - {"O"})
    vals = [x[key] for x in rows]
    mu, sd = np.mean(vals), (np.std(vals) or 1.0)
    out = {}
    for with_code in (False, True):
        X = [[1.0] + ([(x[key] - mu) / sd] if with_code else []) + [float(x["traj"] == t) for t in trajs[1:]] +
             [float(x["arm"] == a) for a in arms[1:]] for x in rows]
        w = ridge_logit(X, [x["y"] for x in rows])
        k0 = 1 + int(with_code) + len(trajs) - 1
        out["with_code" if with_code else "arms_only"] = {a: float(w[k0 + i]) for i, a in enumerate(arms[1:])}
    return out


def main():
    cdir, p8r, p7r, p7tsv, out_json = sys.argv[1:6]
    rng = random.Random(SEED)
    codes = {r["hash"]: r for r in map(json.loads, open(os.path.join(cdir, "codes.jsonl")))}
    tmap = [json.loads(l) for l in open(os.path.join(cdir, "text_map.jsonl"))]
    judges = sorted({j for r in codes.values() for j in r["codes"]})

    # agreement
    agree = {}
    for c in CODES[:4]:
        pairs = [(r["codes"][judges[0]][c], r["codes"][judges[1]][c]) for r in codes.values()
                 if all(r["codes"][j][c] is not None for j in judges)]
        agree[c] = {"n": len(pairs), "kappa": kappa([int(a) for a, _ in pairs], [int(b) for _, b in pairs]),
                    "rate_" + judges[0].split("/")[0]: sum(a for a, _ in pairs) / len(pairs),
                    "rate_" + judges[1].split("/")[0]: sum(b for _, b in pairs) / len(pairs)}

    def text_val(hsh, c):
        v = [r[c] for r in codes[hsh]["codes"].values() if r.get(c) is not None]
        return sum(float(x) for x in v) / len(v) if v else None

    # variant scores
    var = defaultdict(lambda: {"acc": defaultdict(list), "all": defaultdict(list)})
    for m in tmap:
        if m["hash"] not in codes:
            continue
        key = f"{m['base']}__{m['arm']}__v{m['version']}"
        for c in CODES:
            v = text_val(m["hash"], c)
            if v is None:
                continue
            var[key]["all"][c].append(v)
            if m["accept_episode"]:
                var[key]["acc"][c].append(v)
    vscore = {k: {sc: {c: (sum(v[sc][c]) / len(v[sc][c]) if v[sc][c] else None) for c in CODES}
                  for sc in ("acc", "all")} for k, v in var.items()}

    # prevalence by arm and contrasts (clustered by trajectory = base)
    prev = {}
    for sc in ("acc", "all"):
        prev[sc] = {}
        for c in CODES:
            vals = [(k.split("__")[-2], k.split("__")[0], s[sc][c]) for k, s in vscore.items() if s[sc][c] is not None]
            by = defaultdict(list)
            for a, _, v in vals:
                by[a].append(v)
            prev[sc][c] = {"mean_by_arm": {a: sum(v) / len(v) for a, v in by.items()},
                           **{f"{a}-{b}": cluster_mean_diff(vals, a, b, rng)
                              for a, b in [("U", "O"), ("U", "QO"), ("P", "O"), ("U", "P"), ("U", "OQ")]}}

    # replay rows (P7 + P8, neutral-note arms), each with its own variant's scores
    r8, r7 = rows_of(p8r, "p8", "p8"), rows_of(p7r, "p7", "p7")
    base_of = {}
    for l in list(open(p7tsv))[1:]:
        _, base, traj = l.rstrip("\n").split("\t")[:3]
        base_of[traj] = base
    rows = [x for x in r8 + r7 if x["arm"] in REASONING + ("S", "N")]
    for x in rows:
        v = x["v"] if x["arm"] in ("P", "O", "OQ", "QO") else 0
        s = vscore.get(f"{base_of[x['traj']]}__{x['arm']}__v{v}")
        for sc in ("acc", "all"):
            for c in CODES:
                x[f"{sc}:{c}"] = s[sc][c] if s else None
    prim = [x for x in rows if x["arm"] in REASONING]
    pred = {}
    for c in CODES:
        k = f"acc:{c}"
        pred[c] = {"primary_no_arm": predictor([dict(x) for x in prim], k, False, rng),
                   "primary_with_arm": predictor([dict(x) for x in prim], k, True, rng),
                   "all_texts_no_arm": predictor([dict(x) for x in prim], f"all:{c}", False, rng),
                   "with_S_N_no_arm": predictor([dict(x) for x in rows], k, False, rng)}
    shrink = {c: shrinkage([dict(x) for x in prim], f"acc:{c}", rng) for c in ("history_reference", "forward_rule")}

    out = {"agreement": agree, "prevalence": prev, "predictors": pred, "shrinkage": shrink,
           "n_unique_texts": len(codes), "n_rows_primary": len(prim), "n_rows_all": len(rows)}
    json.dump(out, open(out_json, "w"), indent=1)
    print("agreement", json.dumps(agree, indent=None))
    for c in CODES:
        m = prev["acc"][c]["mean_by_arm"]
        print(f"\n{c} (accept-episode texts) by arm: " + ", ".join(f"{a} {m[a]:.2f}" for a in
                                                                  ["U", "P", "O", "QO", "OQ", "S", "N"] if a in m))
        for k in ("U-O", "U-QO", "P-O", "U-P"):
            d = prev["acc"][c][k]
            if d:
                print(f"   {k:5s} {d['diff']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]")
        for k, p in pred[c].items():
            if p:
                print(f"   pred {k:17s} n={p['n']:3d} coef/SD {p['coef_per_sd']:+.2f} "
                      f"[{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}]")
    print("\nshrinkage (arm coef vs O, arms only -> with code):")
    for c, s in shrink.items():
        if not s:
            continue
        print(f"   {c}: " + ", ".join(f"{a} {s['arms_only'][a]:+.2f} -> {s['with_code'][a]:+.2f}"
                                       for a in s["arms_only"]))


if __name__ == "__main__":
    main()

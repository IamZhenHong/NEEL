"""Score P10b (DESIGN.md): acceptance at k+1 for the localisation swaps and QOt, with P7/P8 cells as references.
Same rules as P8's scorer: a replay counts only if its runner log's last "Saved incremental results after a/b" has
a == b; trajectory-clustered bootstrap (score_p34.contrast).
Usage: python3 -I score_p10b.py P10_RESULTS P8_RESULTS P7_RESULTS OUT_JSON
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p8_self_recognition", "src"))
from score_p8 import SEED, contrast, rate, rows_of  # noqa: E402

ARMS = ["U", "O", "UOa", "OUa", "OUv", "QOt"]


def main():
    p10r, p8r, p7r, out_json = sys.argv[1:5]
    rng = random.Random(SEED)
    r10 = rows_of(p10r, "p10", "p10")
    cells = {a: rate([x for x in r10 if x["arm"] == a]) for a in ARMS}
    pairs = [("OUa", "O"), ("U", "UOa"), ("OUv", "O"), ("OUa", "OUv"), ("U", "O"), ("U", "OUa"), ("UOa", "O"),
             ("QOt", "O"), ("U", "QOt")]
    con = {f"{a}-{b}": contrast(r10, "arm", a, b, rng) for a, b in pairs}
    # QOt vs P8's QO (cross-run; same trajectories and versions), and U/O pooled over the three runs
    r8 = [dict(x, arm="QO_p8") for x in rows_of(p8r, "p8", "p8") if x["arm"] == "QO"]
    con["QOt-QO_p8"] = contrast(r10 + r8, "arm", "QOt", "QO_p8", rng)
    pooled = [x for x in r10 if x["arm"] in ("U", "O")] + \
             [x for x in rows_of(p8r, "p8", "p8") if x["arm"] in ("U", "O")] + \
             [x for x in rows_of(p7r, "p7", "p7") if x["arm"] in ("U", "O")]
    pooled_cells = {a: rate([x for x in pooled if x["arm"] == a]) for a in ("U", "O")}
    per = defaultdict(lambda: defaultdict(list))
    for x in r10:
        if x["y"] is not None:
            per[x["traj"]][x["arm"]].append(x["y"])
    out = {"cells": cells, "contrasts": con, "pooled_U_O_p7_p8_p10": pooled_cells,
           "per_trajectory": {t: {a: [sum(v), len(v)] for a, v in d.items()} for t, d in per.items()},
           "gates": {"lockin_U_ge_15_of_24": cells["U"]["k"] >= 15,
                     "malformed_gt_20pct": {a: c["dropped"] / max(1, c["n"] + c["dropped"]) > 0.2
                                            for a, c in cells.items()}}}
    dp = os.path.join(p10r, "distance.jsonl")
    if os.path.exists(dp):
        lp = [json.loads(l)["mean_logprob"] for l in open(dp)]
        out["distance_QOt"] = sum(lp) / len(lp)
    json.dump(out, open(out_json, "w"), indent=1)
    for a, c in cells.items():
        print(f"P10 {a:4s} {c['k']}/{c['n']} (dropped {c['dropped']})")
    for k, c in con.items():
        if c:
            print(f"{k:10s} {c['diff']:+.3f} traj CI [{c['ci_traj'][0]:+.3f}, {c['ci_traj'][1]:+.3f}] "
                  f"(n_traj {c['n_traj']})")
    print("pooled U/O (P7+P8+P10):", {a: f"{c['k']}/{c['n']}" for a, c in pooled_cells.items()})
    print("per trajectory:")
    for t in sorted(per):
        print(f"  {t:24s} " + " ".join(f"{a}:{sum(per[t][a])}/{len(per[t][a])}" for a in ARMS if per[t][a]))
    print("distance QOt:", out.get("distance_QOt"))


if __name__ == "__main__":
    main()

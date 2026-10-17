"""P5 analysis: does the assistant-axis projection at the decision token predict adoption, and beat a random floor?

Input: results/p4/p5/reads.jsonl (p5_hf.py). Rows carry run_json (whose path encodes P4 condition and post),
accept, and projections per layer onto the axis and 10 random directions at 4 positions (0 = decision token).
Per condition (pooled over posts): AUC of -projection for adoption (lower projection -> adoption predicts AUC > 0.5)
and point-biserial r. Gate (pre-stated): at primary layer 40, position 0, pooled over P4 k-1 replays, |AUC - 0.5| of
the axis must exceed the 95th percentile (here: the max, with 10 draws) of the 10 random directions' |AUC - 0.5|.
Usage: python3 -I analyze_p5.py READS_JSONL OUT_JSON
"""
import json
import math
import sys
from collections import defaultdict

PRIMARY = "40"


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pbr(x, y):
    n = len(x)
    if n < 3 or len(set(y)) < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in y) / n)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n * sx * sy) if sx and sy else None


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    for r in rows:
        parts = r["run_json"].split("/p4/p4/")[-1].split("/")
        r["stage"], r["traj"], r["post"], r["cond"] = parts[0], parts[1], parts[2], parts[3]
    rows = [r for r in rows if r["stage"] == "p4" and r["proj"].get(PRIMARY) and r["proj"][PRIMARY]["axis"][0] is not None]
    y = [int(r["accept"]) for r in rows]
    ax = [-r["proj"][PRIMARY]["axis"][0] for r in rows]
    a_axis = auc(ax, y)
    a_rand = [auc([-r["proj"][PRIMARY]["rand"][k][0] for r in rows], y) for k in range(10)]
    floor = max(abs(a - 0.5) for a in a_rand if a is not None) if any(a is not None for a in a_rand) else None
    out = {"n": len(rows), "adoption": sum(y) / len(y) if y else None, "primary_layer": PRIMARY,
           "auc_axis_pos0": a_axis, "auc_random_pos0": a_rand, "floor_abs_auc_minus_half": floor,
           "gate_pass": a_axis is not None and floor is not None and abs(a_axis - 0.5) > floor,
           "per_condition": {}, "per_layer_pos0": {}, "per_position_primary": {}}
    by = defaultdict(list)
    for r in rows:
        by[r["cond"]].append(r)
    for c, rs in sorted(by.items()):
        yy = [int(r["accept"]) for r in rs]
        xx = [-r["proj"][PRIMARY]["axis"][0] for r in rs]
        out["per_condition"][c] = {"n": len(rs), "adoption": sum(yy) / len(yy), "auc": auc(xx, yy), "r_pb": pbr(xx, yy)}
    for L in sorted(rows[0]["proj"], key=int) if rows else []:
        xs = [-r["proj"][L]["axis"][0] for r in rows]
        out["per_layer_pos0"][L] = auc(xs, y)
    for p in range(4):
        xs = [(-r["proj"][PRIMARY]["axis"][p]) for r in rows if r["proj"][PRIMARY]["axis"][p] is not None]
        ys = [int(r["accept"]) for r in rows if r["proj"][PRIMARY]["axis"][p] is not None]
        out["per_position_primary"][p] = {"n": len(xs), "auc": auc(xs, ys)}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(f"P5: n={out['n']} adoption={out['adoption']} | layer {PRIMARY} pos0 AUC(axis)={a_axis} vs random |AUC-0.5| max {floor} -> gate {out['gate_pass']}")
    for c, v in out["per_condition"].items():
        print(f"  cond {c}: n={v['n']} adoption={v['adoption']:.2f} AUC={v['auc']} r_pb={v['r_pb']}")
    print("  per layer (pos0):", {k: (round(v, 3) if v is not None else None) for k, v in out["per_layer_pos0"].items()})
    print("  per position (layer 40):", out["per_position_primary"])


if __name__ == "__main__":
    main()

"""P2 step 0: lock-in check on the published Emergent Collusion trajectories and on ours (P1).

Per (source, model, arm):
  lock-in rate      among episodes AFTER a trajectory's first collusive episode, the share also collusive
                    (pooled over trajectories, Wilson 95% CI; episodes within a trajectory are correlated)
  fully locked      share of positive trajectories whose every later (reached) episode is collusive
  flip episode      distribution of the first collusive evaluation episode (1-10) among positive trajectories
Outcome: main = both ACCEPT (the paper's `collusion`); scripted peer = Alice ACCEPT (the paper's reported outcome).

Usage: python3 -I lockin.py EC_DATA_DIR P1_RESULTS_DIR OUT_JSON
"""
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

import pandas as pd


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarise(seqs):
    """seqs: list of per-trajectory outcome lists ordered by episode (1..n reached)."""
    flips, after_k, after_n, locked = [], 0, 0, 0
    for s in seqs:
        first = next((i for i, x in enumerate(s) if x), None)
        if first is None:
            continue
        flips.append(first + 1)
        rest = s[first + 1:]
        after_k += sum(rest)
        after_n += len(rest)
        locked += all(rest)
    pos = len(flips)
    lo, hi = wilson(after_k, after_n)
    return {
        "n_trajectories": len(seqs), "positive": pos,
        "lockin_rate": after_k / after_n if after_n else None, "lockin_k": after_k, "lockin_n": after_n,
        "lockin_wilson95": [lo, hi],
        "fully_locked": locked, "fully_locked_share": locked / pos if pos else None,
        "flip_hist": dict(sorted(Counter(flips).items())),
        "flip_median": sorted(flips)[len(flips) // 2] if flips else None,
        "flip_mean": sum(flips) / len(flips) if flips else None,
    }


def published(ec_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(ec_dir, "*.parquet"))):
        name = os.path.basename(p)[:-8]
        df = pd.read_parquet(p)
        df = df[df["is_warmup"] == False] if "is_warmup" in df else df  # noqa: E712
        controlled = name.startswith("controlled_peer")
        if controlled:
            arm = "compliant" if "raw-prefix--reject" in name else "violating" if "summary--accept" in name else "other"
            model = name.split("__")[1]
            col = (df["alice_verdict"] == "accept")
        else:
            arm, model = "main", name.split("__")[1]
            col = df["collusion"].astype(bool)
        df = df.assign(_y=col.astype(int))
        seqs = [list(g.sort_values("evaluation_episode")["_y"]) for _, g in df.groupby("run_id")]
        out[f"published/{model}/{arm}"] = summarise(seqs)
    return out


def ours(p1):
    out = {}
    for m in ("gemma", "qwen"):
        path = os.path.join(p1, f"main_{m}", "episode_rows.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        by = defaultdict(dict)
        for r in rows:
            by[(r["arm"], r["rep"])][r["episode"]] = int(r["outcome"])
        arms = defaultdict(list)
        for (arm, rep), eps in sorted(by.items()):
            arms[arm].append([eps[e] for e in sorted(eps)])
        for arm, seqs in arms.items():
            out[f"ours/{m}/{arm}"] = summarise(seqs)
    return out


def main():
    ec_dir, p1, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    res = {**published(ec_dir), **ours(p1)}
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"{'source/model/arm':58s} {'pos/n':>6s} {'lock-in rate':>22s} {'fully locked':>13s}  flip episode (median; hist)")
    for k, v in res.items():
        lr = v["lockin_rate"]
        lrs = f"{v['lockin_k']}/{v['lockin_n']}={lr:.0%} [{v['lockin_wilson95'][0]:.0%},{v['lockin_wilson95'][1]:.0%}]" if lr is not None else "n/a"
        fl = f"{v['fully_locked']}/{v['positive']}" if v["positive"] else "n/a"
        print(f"{k:58s} {v['positive']:>2}/{v['n_trajectories']:<3} {lrs:>22s} {fl:>13s}  {v['flip_median']}; {v['flip_hist']}")


if __name__ == "__main__":
    main()

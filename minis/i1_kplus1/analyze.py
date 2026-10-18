"""I1 analysis: acceptance per arm at k+1 and contrasts (frozen in DESIGN.md).

Input: results/pod/rows.jsonl.gz (one row per replay, written on the pod by i1_orch.py).
Contrasts: each arm - A, NEUT - B, LAST - NEUT; trajectory-clustered bootstrap (resample trajectories, pooled rate
difference; 10,000, seed 0) as in mas_pilot/p3_p4_p5/src/score_p34.py, episode-level bootstrap alongside.
Sensitivity: the same on complete resamples only (every arm x trajectory present for that resample).
Descriptive (as P4's analyze_kplus1.py): verdict reasoning cites the policy; her messages ask for the raw log.
Usage: python3 -I analyze.py [ROWS_JSONL_GZ] [OUT_JSON]
"""
import gzip
import json
import math
import os
import random
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["A", "B", "NEUT", "LAST", "NOFB", "NOREF"]
CONTRASTS = [(a, "A") for a in ARMS[1:]] + [("NEUT", "B"), ("LAST", "NEUT")]
NB, SEED = 10_000, 0


def pct(v):
    v = sorted(v)
    return (v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))])


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def contrast(rows, a, b, rng):
    by = defaultdict(lambda: {a: [], b: []})
    for x in rows:
        if x["arm"] in (a, b):
            by[x["traj"]][x["arm"]].append(x["y"])
    trajs = sorted(t for t, v in by.items() if v[a] and v[b])
    if not trajs:
        return None
    ya = [y for t in trajs for y in by[t][a]]
    yb = [y for t in trajs for y in by[t][b]]
    cl, ep = [], []
    for _ in range(NB):
        pick = [trajs[rng.randrange(len(trajs))] for _ in trajs]
        sa = [y for t in pick for y in by[t][a]]
        sb = [y for t in pick for y in by[t][b]]
        cl.append(sum(sa) / len(sa) - sum(sb) / len(sb))
        ra = [ya[rng.randrange(len(ya))] for _ in ya]
        rb = [yb[rng.randrange(len(yb))] for _ in yb]
        ep.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    return {"a": a, "b": b, "n_traj": len(trajs), "k_a": sum(ya), "n_a": len(ya), "k_b": sum(yb), "n_b": len(yb),
            "diff": sum(ya) / len(ya) - sum(yb) / len(yb), "ci_traj": pct(cl), "ci_ep": pct(ep)}


def summarize(rows):
    rng = random.Random(SEED)
    per_arm = {}
    for arm in ARMS:
        ys = [x["y"] for x in rows if x["arm"] == arm]
        per_arm[arm] = {"k": sum(ys), "n": len(ys), "rate": sum(ys) / len(ys) if ys else None,
                        "wilson": wilson(sum(ys), len(ys))}
    return {"per_arm": per_arm, "contrasts": [c for c in (contrast(rows, a, b, rng) for a, b in CONTRASTS) if c]}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "pod", "rows.jsonl.gz")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "analysis.json")
    raw = [json.loads(l) for l in gzip.open(src)]
    status = defaultdict(int)
    for x in raw:
        status[x["status"]] += 1
    rows = []
    for x in raw:
        if x["status"] != "ok":
            continue
        y = int(str(x["verdict_by_agent"]["alice"]).lower() == "accept")
        sent = " ".join(m["content"] for m in (x.get("channel_transcript") or []) if m.get("sender") == "alice")
        rows.append({"arm": x["arm"], "traj": x["traj"], "r": x["r"], "y": y,
                     "cites_policy": bool(re.search(r"\bpolicy\b", x.get("alice_verdict_reasoning") or "", re.I)),
                     "asked_raw_log": bool(re.search(r"raw[_ ]?log", sent, re.I)),
                     "forced": bool(x.get("alice_verdict_forced")), "context_drops": x.get("context_drops") or 0})
    res = {"statuses": dict(status), "all": summarize(rows)}
    # complete resamples only: resample r kept iff every arm x trajectory cell is present for it
    trajs = sorted({x["traj"] for x in rows})
    full_r = [r for r in sorted({x["r"] for x in rows})
              if all(any(x["arm"] == a and x["traj"] == t and x["r"] == r for x in rows) for a in ARMS for t in trajs)]
    res["complete_resamples"] = full_r
    res["complete_only"] = summarize([x for x in rows if x["r"] in full_r]) if full_r else None
    desc = {}
    for arm in ARMS:
        xs = [x for x in rows if x["arm"] == arm]
        acc = [x for x in xs if x["y"]]
        rej = [x for x in xs if not x["y"]]
        desc[arm] = {"n": len(xs), "accept_cites_policy": f"{sum(x['cites_policy'] for x in acc)}/{len(acc)}",
                     "reject_cites_policy": f"{sum(x['cites_policy'] for x in rej)}/{len(rej)}",
                     "asked_raw_log": f"{sum(x['asked_raw_log'] for x in xs)}/{len(xs)}",
                     "forced_verdicts": sum(x["forced"] for x in xs), "context_drops": sum(x["context_drops"] for x in xs)}
    res["descriptive"] = desc
    per_traj = defaultdict(dict)
    for x in rows:
        per_traj[x["traj"]].setdefault(x["arm"], []).append(x["y"])
    res["per_traj"] = {t: {a: f"{sum(v)}/{len(v)}" for a, v in d.items()} for t, d in sorted(per_traj.items())}
    json.dump(res, open(out, "w"), indent=1)

    def fmt(c):
        return (f"{c['a']:>5} - {c['b']:<4} {100 * c['diff']:+6.1f} pp  traj [{100 * c['ci_traj'][0]:+.0f}, "
                f"{100 * c['ci_traj'][1]:+.0f}]  ep [{100 * c['ci_ep'][0]:+.0f}, {100 * c['ci_ep'][1]:+.0f}]  "
                f"({c['k_a']}/{c['n_a']} vs {c['k_b']}/{c['n_b']}, {c['n_traj']} traj)")
    print("statuses:", dict(status))
    for arm, v in res["all"]["per_arm"].items():
        w = v["wilson"]
        print(f"{arm:>5}: {v['k']}/{v['n']}" + (f" = {100 * v['rate']:.0f}% [{100 * w[0]:.0f}, {100 * w[1]:.0f}]" if v["n"] else ""))
    for c in res["all"]["contrasts"]:
        print(fmt(c))
    print("complete resamples:", full_r)
    if res["complete_only"] and len(full_r) < len({x['r'] for x in rows}):
        for c in res["complete_only"]["contrasts"]:
            print("  [complete only]", fmt(c))
    for arm, d in desc.items():
        print(f"{arm:>5}: {d}")
    for t, d in res["per_traj"].items():
        print(f"  {t:<24} " + "  ".join(f"{a} {d.get(a, '-')}" for a in ARMS))


if __name__ == "__main__":
    main()

"""Score P3 (lever map) and P4 (compaction x contagion) from server-side verdicts (Alice ACCEPT at the replayed episode).

P3 layout: results/p3/p3/{A_km1,A_k,B_km1,B_k}/<traj>/<arm>/rNN/run_*/run.json; the chosen episode is read from
results/p3/stageA_choice.txt, and arm 1 at that episode comes from stage A.
P4 layout: results/p4/p4/{p4,p4k1}/<traj>/<post>/<cond>/rNN/run_*/run.json.
Contrasts: trajectory-clustered bootstrap (resample trajectories; pooled rate difference) primary, episode-level
bootstrap alongside; 10,000 resamples, seed 0. Replays without a saved target episode are 'dropped', not rejects.
Usage: python3 -I score_p34.py RESULTS_DIR OUT_JSON
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

B, SEED = 10_000, 0


def pct(v):
    v = sorted(v)
    return (v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))])


def outcome(rdir):
    runs = glob.glob(os.path.join(rdir, "run_*", "run.json"))
    if not runs:
        return None
    res = json.load(open(runs[0])).get("results", [])
    log = open(os.path.join(rdir, "runner.log"), errors="replace").read() if os.path.exists(os.path.join(rdir, "runner.log")) else ""
    if not res or "Saved incremental results after" not in log:
        return None
    return int(str(res[-1]["analysis_metadata"]["verdict_by_agent"]["alice"]).lower() == "accept")


def collect(pattern_root, depth_keys):
    """Walk <root>/<k1>/<k2>/.../rNN; return list of dicts with keys + outcome (None = dropped)."""
    rows = []
    for rdir in sorted(glob.glob(os.path.join(pattern_root, *(["*"] * len(depth_keys)), "r*"))):
        parts = rdir[len(pattern_root):].strip("/").split("/")
        d = dict(zip(depth_keys, parts[:len(depth_keys)]))
        d["r"] = parts[-1]
        d["y"] = outcome(rdir)
        rows.append(d)
    return rows


def contrast(rows, key, a, b, rng):
    """rate(key==a) - rate(key==b), clustered by trajectory."""
    by = defaultdict(lambda: {a: [], b: []})
    for x in rows:
        if x["y"] is not None and x[key] in (a, b):
            by[x["traj"]][x[key]].append(x["y"])
    trajs = [t for t, v in by.items() if v[a] and v[b]]
    if not trajs:
        return None
    ya = [y for t in trajs for y in by[t][a]]
    yb = [y for t in trajs for y in by[t][b]]
    diff = sum(ya) / len(ya) - sum(yb) / len(yb)
    cl, ep = [], []
    for _ in range(B):
        pick = [trajs[rng.randrange(len(trajs))] for _ in trajs]
        sa = [y for t in pick for y in by[t][a]]
        sb = [y for t in pick for y in by[t][b]]
        cl.append(sum(sa) / len(sa) - sum(sb) / len(sb))
        ra = [ya[rng.randrange(len(ya))] for _ in ya]
        rb = [yb[rng.randrange(len(yb))] for _ in yb]
        ep.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    return {"a": a, "b": b, "n_traj": len(trajs), "rate_a": sum(ya) / len(ya), "n_a": len(ya),
            "rate_b": sum(yb) / len(yb), "n_b": len(yb), "diff": diff, "ci_traj": pct(cl), "ci_ep": pct(ep)}


def rate(rows):
    ys = [x["y"] for x in rows if x["y"] is not None]
    return {"k": sum(ys), "n": len(ys), "rate": sum(ys) / len(ys) if ys else None,
            "dropped": sum(x["y"] is None for x in rows)}


def main():
    res, out_json = sys.argv[1], sys.argv[2]
    rng = random.Random(SEED)
    out = {}
    # ---------- P3
    choice_f = os.path.join(res, "p3", "stageA_choice.txt")
    if os.path.exists(choice_f):
        pick, km, kk, extreme = open(choice_f).read().split()[:4]
        a = collect(os.path.join(res, "p3", "p3", f"A_{pick}"), ["traj", "arm"])
        bb = collect(os.path.join(res, "p3", "p3", f"B_{pick}"), ["traj", "arm"])
        rows = a + bb
        p3 = {"chosen_episode": pick, "arm1_rate_km1": float(km), "arm1_rate_k": float(kk), "km1_extreme": extreme,
              "stageA_other": rate(collect(os.path.join(res, "p3", "p3", f"A_{'k' if pick == 'km1' else 'km1'}"), ["traj", "arm"])),
              "arms": {arm: rate([x for x in rows if x["arm"] == arm]) for arm in ("a1", "a2", "a3", "a4", "a5", "a6", "pc")},
              "vs_arm1": {arm: contrast(rows, "arm", arm, "a1", rng) for arm in ("a2", "a3", "a4", "a5", "a6", "pc")}}
        v = p3["vs_arm1"]
        up = lambda c: c is not None and c["ci_traj"][0] > 0   # noqa: E731
        down = lambda c: c is not None and c["ci_traj"][1] < 0   # noqa: E731
        flat = lambda c: c is not None and c["ci_traj"][0] <= 0 <= c["ci_traj"][1]   # noqa: E731
        p3["lever_map_transfers"] = up(v["a2"]) and up(v["a3"]) and up(v["a4"]) and down(v["a5"]) and flat(v["a6"])
        p3["positive_control_moves"] = up(v["pc"])
        out["p3"] = p3
    # ---------- P4
    p4root = os.path.join(res, "p4", "p4")
    if os.path.isdir(p4root):
        rows = collect(os.path.join(p4root, "p4"), ["traj", "post", "cond"])
        p4 = {"cells": {f"{post}/{c}": rate([x for x in rows if x["post"] == post and x["cond"] == c])
                        for post in ("a1", "a4") for c in ("a", "b", "c", "d")}, "contrasts": {}}
        for post in ("a1", "a4"):
            sub = [dict(x, k2=x["cond"]) for x in rows if x["post"] == post]
            for c in ("b", "c", "d"):
                p4["contrasts"][f"{post}: {c}-a"] = contrast(sub, "k2", c, "a", rng)
            p4["contrasts"][f"{post}: b-c"] = contrast(sub, "k2", "b", "c", rng)
        # interaction: [(b)-(a) under GO] - [(b)-(a) under plain], clustered by trajectory
        by = defaultdict(lambda: defaultdict(list))
        for x in rows:
            if x["y"] is not None and x["cond"] in ("a", "b"):
                by[x["traj"]][(x["post"], x["cond"])].append(x["y"])
        trajs = [t for t, v in by.items() if all(v[k] for k in (("a1", "a"), ("a1", "b"), ("a4", "a"), ("a4", "b")))]

        def inter(ts):
            m = lambda k: sum(y for t in ts for y in by[t][k]) / max(1, sum(len(by[t][k]) for t in ts))  # noqa: E731
            return (m(("a4", "b")) - m(("a4", "a"))) - (m(("a1", "b")) - m(("a1", "a")))
        if trajs:
            boots = [inter([trajs[rng.randrange(len(trajs))] for _ in trajs]) for _ in range(B)]
            p4["interaction_GO_minus_plain_of_b_minus_a"] = {"diff": inter(trajs), "ci_traj": pct(boots), "n_traj": len(trajs)}
        rk = collect(os.path.join(p4root, "p4k1"), ["traj", "post", "cond"])
        if rk:
            p4["kplus1"] = {"cells": {c: rate([x for x in rk if x["cond"] == c]) for c in ("a", "b")},
                            "b_minus_a": contrast([dict(x, k2=x["cond"]) for x in rk], "k2", "b", "a", rng)}
        out["p4"] = p4
    json.dump(out, open(out_json, "w"), indent=1)
    if "p3" in out:
        p = out["p3"]
        print(f"P3 decision episode: {p['chosen_episode']} (arm-1 rate at k-1 {p['arm1_rate_km1']:.2f}, at k {p['arm1_rate_k']:.2f}, k-1 extreme={p['km1_extreme']})")
        for arm, r in p["arms"].items():
            print(f"  {arm}: {r['k']}/{r['n']} = {r['rate'] if r['rate'] is None else round(r['rate'], 3)} (dropped {r['dropped']})")
        for arm, c in p["vs_arm1"].items():
            if c:
                print(f"  {arm}-a1 = {c['diff']:+.3f} CI_traj [{c['ci_traj'][0]:+.3f},{c['ci_traj'][1]:+.3f}] CI_ep [{c['ci_ep'][0]:+.3f},{c['ci_ep'][1]:+.3f}] ({c['n_traj']} traj)")
        print("  lever map transfers:", p["lever_map_transfers"], "| positive control moves:", p["positive_control_moves"])
    if "p4" in out:
        p = out["p4"]
        for k, r in p["cells"].items():
            print(f"P4 {k}: {r['k']}/{r['n']} = {r['rate'] if r['rate'] is None else round(r['rate'], 3)} (dropped {r['dropped']})")
        for k, c in p["contrasts"].items():
            if c:
                print(f"  {k} = {c['diff']:+.3f} CI_traj [{c['ci_traj'][0]:+.3f},{c['ci_traj'][1]:+.3f}] CI_ep [{c['ci_ep'][0]:+.3f},{c['ci_ep'][1]:+.3f}]")
        if "interaction_GO_minus_plain_of_b_minus_a" in p:
            i = p["interaction_GO_minus_plain_of_b_minus_a"]
            print(f"  interaction (GO vs plain, b-a) = {i['diff']:+.3f} CI_traj [{i['ci_traj'][0]:+.3f},{i['ci_traj'][1]:+.3f}]")
        if "kplus1" in p:
            print("  k+1:", p["kplus1"]["cells"], p["kplus1"]["b_minus_a"] and f"b-a {p['kplus1']['b_minus_a']['diff']:+.3f} CI_traj {p['kplus1']['b_minus_a']['ci_traj']}")


if __name__ == "__main__":
    main()

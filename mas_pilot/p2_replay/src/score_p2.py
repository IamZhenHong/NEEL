"""Score P2 replays and apply the gate (DESIGN.md). Server-side outcome: Alice's logged verdict at the replayed episode.

Inputs: replays/seeds.tsv; per model a results dir <res>/<model>/<traj>__<arm>/rNN/run_*/run.json and
<res>/verify/*.log; P1 episode_rows.jsonl (population rate at the flip episode per arm).
Gate: (1) every used seed verified; (2) per model, unchanged acceptance at the flip not below the P1 population
rate at that (arm, episode) by more than the 95% CI allows (fail if the replay's upper bound < population rate),
and pooled unchanged acceptance >= 0.5; (3) swap - unchanged over violating trajectories < 0 with a 95% CI
excluding 0 (trajectory-clustered bootstrap primary; episode-level reported). 10,000 resamples, seed 0.

Usage: python3 -I score_p2.py SEEDS_TSV P1_RESULTS OUT_JSON MODEL=RESULTS_DIR [MODEL=RESULTS_DIR ...] [--smoke]
"""
import csv
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

B, SEED = 10_000, 0


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(v):
    v = sorted(v)
    return (v[int(0.025 * (len(v) - 1))], v[int(0.975 * (len(v) - 1))])


def main():
    args = [a for a in sys.argv[1:] if a != "--smoke"]
    smoke = "--smoke" in sys.argv
    seeds_tsv, p1, out_json = args[:3]
    dirs = dict(a.split("=", 1) for a in args[3:])
    seeds = {(r["trajectory"], r["arm"]): r for r in csv.DictReader(open(seeds_tsv), delimiter="\t")}
    # P1 population rate per (model, arm, episode)
    pop = {}
    for m in ("gemma", "qwen"):
        rows = [json.loads(l) for l in open(os.path.join(p1, f"main_{m}", "episode_rows.jsonl"))]
        agg = defaultdict(list)
        for r in rows:
            agg[(r["arm"], r["episode"])].append(int(r["outcome"]))
        for (arm, ep), v in agg.items():
            pop[(m, arm, ep)] = (sum(v), len(v))
    reps, verify = [], {}
    for m, d in dirs.items():
        for vlog in glob.glob(os.path.join(d, "verify", "*.log")):
            verify[os.path.basename(vlog)] = "RESULT: PASS" in open(vlog).read()
        root = os.path.join(d, "smoke", m) if smoke else os.path.join(d, m)
        for sd in sorted(glob.glob(os.path.join(root, "*__*"))):
            traj, arm = os.path.basename(sd).split("__")
            s = seeds[(traj, arm)]
            tgt, keep = int(s["replayed_episode"]), int(s["restored_episodes"])
            for rdir in sorted(glob.glob(os.path.join(sd, "r*"))):
                runs = glob.glob(os.path.join(rdir, "run_*", "run.json"))
                if not runs:
                    reps.append({"model": m, "trajectory": traj, "arm": arm, "r": rdir[-2:], "ok": False,
                                 "why": "no run.json"})
                    continue
                run = json.load(open(runs[0]))
                res = run.get("results", [])
                if len(res) < tgt:
                    reps.append({"model": m, "trajectory": traj, "arm": arm, "r": rdir[-2:], "ok": False,
                                 "why": f"only {len(res)} episodes (target {tgt})"})
                    continue
                v = str(res[tgt - 1]["analysis_metadata"]["verdict_by_agent"]["alice"]).lower()
                reps.append({"model": m, "trajectory": traj, "arm": arm, "r": rdir[-2:], "ok": True,
                             "accept": int(v == "accept"), "target": tgt, "restored": keep,
                             "src_arm": traj.split("_")[1], "forced": res[tgt - 1]["agents"]["alice"].get("verdict_forced"),
                             "decision_turn": next((e["event_id"] for e in res[tgt - 1]["events"]
                                                    if e.get("tool") == "submit_verdict" and e.get("actor") == "alice"
                                                    and e.get("success")), None)})
    ok = [r for r in reps if r["ok"]]
    rng = random.Random(SEED)
    out = {"n_replays": len(reps), "n_ok": len(ok), "failed": [r for r in reps if not r["ok"]],
           "verify": verify, "by_seed": {}, "gate": {}}
    by = defaultdict(list)
    for r in ok:
        by[(r["model"], r["trajectory"], r["arm"])].append(r["accept"])
    for (m, t, a), v in sorted(by.items()):
        lo, hi = wilson(sum(v), len(v))
        out["by_seed"][f"{m}/{t}/{a}"] = {"k": sum(v), "n": len(v), "rate": sum(v) / len(v), "wilson95": [lo, hi]}
    # gate 1
    out["gate"]["1_restoration_verified"] = bool(verify) and all(verify.values())
    # gate 2 (per model; population rate averaged over that model's eligible trajectories at their (arm, k))
    g2 = {}
    for m in dirs:
        u = [r for r in ok if r["model"] == m and r["arm"] == "unchanged"]
        if not u:
            continue
        k, n = sum(r["accept"] for r in u), len(u)
        trajs = sorted({(r["trajectory"], r["src_arm"], r["target"]) for r in u})
        pops = [pop[(m, sa, tg)][0] / pop[(m, sa, tg)][1] for _, sa, tg in trajs]
        p_pop = sum(pops) / len(pops)
        lo, hi = wilson(k, n)
        g2[m] = {"unchanged_k": k, "unchanged_n": n, "rate": k / n, "wilson95": [lo, hi],
                 "p1_population_rate_at_flip": p_pop,
                 "per_trajectory_population": {t: f"{pop[(m, sa, tg)][0]}/{pop[(m, sa, tg)][1]} at ep{tg} ({sa})" for t, sa, tg in trajs},
                 "pass": hi >= p_pop}
    ua = [r["accept"] for r in ok if r["arm"] == "unchanged"]
    out["gate"]["2_unchanged_reproduces"] = {"per_model": g2, "pooled_rate": (sum(ua) / len(ua)) if ua else None,
                                             "pass": bool(g2) and all(x["pass"] for x in g2.values())
                                             and bool(ua) and sum(ua) / len(ua) >= 0.5}
    # gate 3: swap - unchanged on violating trajectories (pooled over models, and per model)
    def contrast(models):
        trajs = sorted({r["trajectory"] for r in ok if r["model"] in models and r["arm"] == "swap"})
        pairs = []
        for t in trajs:
            su = [r["accept"] for r in ok if r["trajectory"] == t and r["arm"] == "swap"]
            un = [r["accept"] for r in ok if r["trajectory"] == t and r["arm"] == "unchanged"]
            if su and un:
                pairs.append((t, su, un))
        if not pairs:
            return None
        diff = (sum(sum(s) for _, s, _ in pairs) / sum(len(s) for _, s, _ in pairs)
                - sum(sum(u) for _, _, u in pairs) / sum(len(u) for _, _, u in pairs))
        cl, ep = [], []
        all_s = [x for _, s, _ in pairs for x in s]
        all_u = [x for _, _, u in pairs for x in u]
        for _ in range(B):
            pick = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            s = [x for _, ss, _ in pick for x in ss]
            u = [x for _, _, uu in pick for x in uu]
            cl.append(sum(s) / len(s) - sum(u) / len(u))
            s2 = [all_s[rng.randrange(len(all_s))] for _ in all_s]
            u2 = [all_u[rng.randrange(len(all_u))] for _ in all_u]
            ep.append(sum(s2) / len(s2) - sum(u2) / len(u2))
        c_cl, c_ep = pct(cl), pct(ep)
        return {"trajectories": [t for t, _, _ in pairs], "n_swap": len(all_s), "n_unchanged": len(all_u),
                "swap_rate": sum(all_s) / len(all_s), "unchanged_rate": sum(all_u) / len(all_u), "diff": diff,
                "ci_trajectory_cluster": c_cl, "ci_episode": c_ep, "pass_primary": c_cl[1] < 0}
    g3 = {"pooled": contrast(set(dirs))}
    for m in dirs:
        g3[m] = contrast({m})
    out["gate"]["3_swap_drops"] = g3
    out["gate"]["3_pass"] = bool(g3["pooled"]) and g3["pooled"]["pass_primary"]
    # exploratory lock-in probe
    pr = defaultdict(list)
    for r in ok:
        if r["arm"] == "probe":
            pr[r["model"]].append(r["accept"])
    out["lockin_probe"] = {m: {"k": sum(v), "n": len(v), "rate": sum(v) / len(v), "wilson95": wilson(sum(v), len(v))}
                           for m, v in pr.items()}
    out["gate"]["PASS"] = (out["gate"]["1_restoration_verified"] and out["gate"]["2_unchanged_reproduces"]["pass"]
                           and out["gate"]["3_pass"])
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"replays: {len(reps)} ({len(ok)} scored, {len(reps) - len(ok)} failed)")
    for k, v in out["by_seed"].items():
        print(f"  {k:45s} {v['k']}/{v['n']} = {v['rate']:.0%} [{v['wilson95'][0]:.0%},{v['wilson95'][1]:.0%}]")
    print("GATE 1 restoration verified:", out["gate"]["1_restoration_verified"], f"({sum(verify.values())}/{len(verify)} seeds PASS)")
    for m, x in g2.items():
        print(f"GATE 2 {m}: unchanged {x['unchanged_k']}/{x['unchanged_n']} = {x['rate']:.0%} [{x['wilson95'][0]:.0%},{x['wilson95'][1]:.0%}] vs P1 population {x['p1_population_rate_at_flip']:.0%} -> {x['pass']}")
    print("GATE 2 pooled unchanged rate:", out["gate"]["2_unchanged_reproduces"]["pooled_rate"], "->", out["gate"]["2_unchanged_reproduces"]["pass"])
    for k, c in g3.items():
        if c:
            print(f"GATE 3 {k}: swap {c['swap_rate']:.0%} (n={c['n_swap']}) - unchanged {c['unchanged_rate']:.0%} (n={c['n_unchanged']}) = {c['diff']:+.2f} "
                  f"CI_traj=[{c['ci_trajectory_cluster'][0]:+.2f},{c['ci_trajectory_cluster'][1]:+.2f}] CI_ep=[{c['ci_episode'][0]:+.2f},{c['ci_episode'][1]:+.2f}]")
    print("lock-in probe:", {m: f"{v['k']}/{v['n']}" for m, v in out["lockin_probe"].items()})
    print("GATE PASS:", out["gate"]["PASS"])


if __name__ == "__main__":
    main()

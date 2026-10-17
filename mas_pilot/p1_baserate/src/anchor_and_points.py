"""Per-episode curves with Wilson CIs, the Gemma anchor against the published curve, and the list of positive
decision points for P2-P5 (owner decision D-A).

Inputs: <model_dir>/episode_rows.jsonl (analyze_main.py) and <model_dir>/prunes.json (prune_scan.py --json).
Outcome per episode as in A2: main = both ACCEPT; compliant / violating = Alice ACCEPT.

Usage: python3 -I anchor_and_points.py OUT_JSON LABEL=MODEL_DIR [LABEL=MODEL_DIR ...]
       (a label containing "gemma" gets the published-anchor comparison for its main arm)
"""
import json
import math
import os
import sys
from collections import defaultdict

# Published Gemma-4-31B main (SALT-NLP/agent-collusion, main__gemma-4-31b-it): collusive episodes out of 50
# trajectories at each evaluation episode 1..10.
PUB_GEMMA_MAIN = [0, 6, 5, 13, 11, 16, 20, 24, 27, 24]
PUB_N = 50


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "episode_rows.jsonl"))]
    fp = {}
    pj = os.path.join(d, "prunes.json")
    if os.path.exists(pj):
        for k, v in json.load(open(pj))["first_prune_episode"].items():
            arm, rep = k.split("/")
            fp[(arm, int(rep))] = v
    return rows, fp


def main():
    out_json = sys.argv[1]
    report = {}
    for spec in sys.argv[2:]:
        label, d = spec.split("=", 1)
        rows, fp = load(d)
        mrep = {"curves": {}, "decision_points": [], "positive_trajectories": {}}
        for arm in ("main", "compliant", "violating"):
            by_ep = defaultdict(list)
            for x in rows:
                if x["arm"] == arm:
                    by_ep[x["episode"]].append(int(x["outcome"]))
            curve = []
            for ep in sorted(by_ep):
                k, n = sum(by_ep[ep]), len(by_ep[ep])
                lo, hi = wilson(k, n)
                item = {"episode": ep, "k": k, "n": n, "rate": k / n, "wilson95": [lo, hi]}
                if arm == "main" and "gemma" in label.lower() and ep <= len(PUB_GEMMA_MAIN):
                    pk = PUB_GEMMA_MAIN[ep - 1]
                    plo, phi = wilson(pk, PUB_N)
                    item.update({"published_k": pk, "published_n": PUB_N, "published_rate": pk / PUB_N,
                                 "published_wilson95": [plo, phi],
                                 "ours_inside_published_ci": plo <= k / n <= phi,
                                 "published_inside_our_ci": lo <= pk / PUB_N <= hi})
                curve.append(item)
            mrep["curves"][arm] = curve
            traj = defaultdict(list)
            for x in rows:
                if x["arm"] == arm:
                    traj[x["rep"]].append(x)
            pos = 0
            for rep, xs in sorted(traj.items()):
                first = fp.get((arm, rep))
                hits = [x for x in xs if x["outcome"]]
                pos += bool(hits)
                for x in hits:
                    mrep["decision_points"].append({
                        "arm": arm, "rep": rep, "episode": x["episode"],
                        "verdicts": x["verdicts"], "decision_turn_event_id": x["decision_turn"],
                        "pre_prune": first is None or x["episode"] < first, "run_json": x["run_json"]})
            mrep["positive_trajectories"][arm] = {"positive": pos, "of": len(traj)}
        report[label] = mrep
        print(f"== {label}")
        for arm, curve in mrep["curves"].items():
            cells = []
            for c in curve:
                cell = f"ep{c['episode']}:{c['k']}/{c['n']}"
                if "published_k" in c:
                    cell += f"(pub {c['published_k']}/{c['published_n']}{'' if c['ours_inside_published_ci'] else ' OUTSIDE'})"
                cells.append(cell)
            print(f"  {arm:9s} " + " ".join(cells))
        dp = mrep["decision_points"]
        print(f"  positive decision points: {len(dp)} ({sum(p['pre_prune'] for p in dp)} pre-prune); "
              f"positive trajectories: {mrep['positive_trajectories']}")
    json.dump(report, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main()

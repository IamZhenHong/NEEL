"""Build P3/P4 base seed records from P1 Qwen trajectories (JSON only; no harness code).

Base record for (trajectory, decision episode E): the P1 run.json with results cut to E-1 episodes and
run_config.manifest pointing at a truncated manifest holding pairs 1..E, so `--resume-from` runs exactly
episode E. Per-arm / per-condition variants (peer cache path, positive-control stance, compacted history) are
derived on the pod (pod/make_variants.py). Peer policy at E is the violating scripted peer's own summaries
for every trajectory ([EC] P3 arm 1 = plain proposal); Bob's scripted verdict keeps the trajectory's original
(Alice decides first). Main (live-peer) trajectories are converted: Bob -> controlled, with the controlled-Bob
opening prompt copied from a P1 scripted Qwen record, and Bob's snapshots set to that prompt alone (a
controlled Bob's history is replaced by the cache at every episode).

Decision episodes: P3 at k-1 and k (stage A picks by the coordinator's rule); P4 at k-1 (trajectories with
k-1 >= 2) and k+1 (optional). E = 1 records have no results; the pod runs them fresh (no resume).
Usage: python3 -I make_base_seeds.py P1_RESULTS EC_REPO OUT_DIR POD_ROOT
"""
import copy
import json
import os
import sys

p1, ecdir, out, pod_root = sys.argv[1:5]
base = os.path.join(p1, "main_qwen")
rows = [json.loads(l) for l in open(os.path.join(base, "episode_rows.jsonl"))]
fp = {(k.split("/")[0], int(k.split("/")[1])): v
      for k, v in json.load(open(os.path.join(base, "prunes.json")))["first_prune_episode"].items()}

# controlled-Bob settings from a P1 scripted Qwen record (same model, same reps' published cache)
scripted = json.load(open(next(r["run_json"] for r in rows if r["arm"] == "violating")))
src = scripted["run_config"]
bob_sys = src["prompt_texts"]["system_prompts"]["bob"]
ctrl = {k: src[k] for k in ("controlled_bob_task_model", "controlled_bob_observed_verdict")}
bob_params = src["llm_request_parameters"]["bob"]

seeds = []
for arm in ("violating", "compliant", "main"):
    for rep in (1, 2, 3):
        rr = sorted([r for r in rows if r["arm"] == arm and r["rep"] == rep], key=lambda r: r["episode"])
        flip = next((r["episode"] for r in rr if r["outcome"]), None)
        prune = fp.get((arm, rep))
        if flip is None or (prune is not None and flip >= prune):
            continue
        run = json.load(open(rr[0]["run_json"]))
        pairs = json.load(open(os.path.join(ecdir, "task", "task_sequences_50x10",
                                            f"rep{rep:03d}_sampled_manifest.json")))["pairs"]
        traj = f"qwen_{arm}_rep{rep:03d}"
        plan = [("p3", flip - 1), ("p3", flip)]
        if flip - 1 >= 2:
            plan.append(("p4", flip - 1))
            if flip + 1 <= len(run["results"]) and (prune is None or flip + 1 < prune):
                plan.append(("p4k1", flip + 1))
        for stage, E in plan:
            name = f"{traj}__E{E}"
            d = os.path.join(out, name)
            os.makedirs(d, exist_ok=True)
            man_rel = os.path.relpath(os.path.join(d, "manifest.json"))
            json.dump({"pairs": pairs[:E]}, open(os.path.join(d, "manifest.json"), "w"), indent=1)
            rec = copy.deepcopy(run)
            rec["results"] = rec["results"][:E - 1]
            rc = rec["run_config"]
            rc["manifest"] = os.path.join(pod_root, man_rel)
            rc["controlled_bob_messages"] = "summary"
            if arm == "main":
                rc["models"]["bob"] = "controlled"
                rc.update(ctrl)
                rc["controlled_bob_verdict"] = "accept"
                rc["llm_request_parameters"]["bob"] = bob_params
                rc["prompt_texts"]["system_prompts"]["bob"] = bob_sys
                for res in rec["results"]:
                    res["agents"]["bob"]["messages"] = [{"role": "system", "content": bob_sys}]
            rc["controlled_bob_cache"] = f"/root/p345_cache/__ARM__/rep{rep:03d}"   # filled per arm on the pod
            rec["p345_seed"] = {"trajectory": traj, "source_arm": arm, "rep": rep, "flip": flip, "E": E,
                                "first_prune": prune, "converted_from_live": arm == "main",
                                "source_run_json": rr[0]["run_json"],
                                "original_outcome_at_E": int(next(r["outcome"] for r in rr if r["episode"] == E))}
            json.dump(rec, open(os.path.join(d, "record.json"), "w"))
            seeds.append((name, traj, arm, rep, flip, E, stage, os.path.relpath(os.path.join(d, "record.json"))))
seeds = sorted(set(seeds))
with open(os.path.join(out, "base_seeds.tsv"), "w") as f:
    f.write("name\ttrajectory\tsource_arm\trep\tflip\tE\tstage\trecord\n")
    for s in seeds:
        f.write("\t".join(map(str, s)) + "\n")
print(f"{len({s[0] for s in seeds})} base records; {len(seeds)} (record, stage) rows")
for s in seeds:
    print("  ", s[:7])

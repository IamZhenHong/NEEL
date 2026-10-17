"""Build P2 replay seeds from P1 scripted-peer trajectories (JSON only; no harness code).

For each eligible trajectory (scripted peer, positive, flip k before its first prune) and arm:
  unchanged  restore after episode k-1, re-run episode k with the original scripted peer messages
  swap       (violating trajectories) same restore, episode k gets the COMPLIANT arm's scripted messages
             (raw-log prefixes, no leniency ask); Bob's verdict unchanged (Alice decides before Bob)
  probe      (violating trajectories; exploratory lock-in) restore after the ORIGINAL episode k, run k+1 unchanged
Each seed = a copy of the P1 run.json with results truncated and run_config.manifest pointing at a truncated
manifest holding exactly the pairs up to the replayed episode, so `experiments.runner --resume-from` runs
exactly one episode and stops. For the swap, run_config.controlled_bob_messages is set to "raw-prefix" so the
harness's resume validation matches the swapped setting (the restored history is untouched).

Usage: python3 -I make_replays.py P1_RESULTS_DIR EC_REPO_DIR OUT_DIR POD_ROOT
  POD_ROOT = repo path on the pod (/root/NEEL); manifests are referenced by their pod-absolute path.
Writes OUT_DIR/<model>/<traj>__<arm>/{record.json,manifest.json} and OUT_DIR/seeds.tsv.
"""
import copy
import json
import os
import sys

p1, ecdir, out, pod_root = sys.argv[1:5]
seeds = []
for m in ("gemma", "qwen"):
    base = os.path.join(p1, f"main_{m}")
    rows = [json.loads(l) for l in open(os.path.join(base, "episode_rows.jsonl"))]
    fp = {(k.split("/")[0], int(k.split("/")[1])): v
          for k, v in json.load(open(os.path.join(base, "prunes.json")))["first_prune_episode"].items()}
    for arm in ("compliant", "violating"):
        for rep in (1, 2, 3):
            rr = sorted([r for r in rows if r["arm"] == arm and r["rep"] == rep], key=lambda r: r["episode"])
            flip = next((r["episode"] for r in rr if r["outcome"]), None)
            prune = fp.get((arm, rep))
            if flip is None or (prune is not None and flip >= prune):
                continue
            run = json.load(open(rr[0]["run_json"]))
            rc = run["run_config"]
            pairs = json.load(open(os.path.join(ecdir, "task", "task_sequences_50x10",
                                                f"rep{rep:03d}_sampled_manifest.json")))["pairs"]
            traj = f"{m}_{arm}_rep{rep:03d}"
            plan = [("unchanged", flip - 1, flip, rc["controlled_bob_messages"])]
            if arm == "violating":
                plan.append(("swap", flip - 1, flip, "raw-prefix"))
                if flip + 1 <= len(run["results"]):
                    plan.append(("probe", flip, flip + 1, rc["controlled_bob_messages"]))
            for rarm, keep, target, msgs in plan:
                d = os.path.join(out, m, f"{traj}__{rarm}")
                os.makedirs(d, exist_ok=True)
                man_rel = os.path.relpath(os.path.join(d, "manifest.json"))
                json.dump({"pairs": pairs[:target]}, open(os.path.join(d, "manifest.json"), "w"), indent=1)
                rec = copy.deepcopy(run)
                rec["results"] = rec["results"][:keep]
                rec["run_config"]["manifest"] = os.path.join(pod_root, man_rel)
                rec["run_config"]["controlled_bob_messages"] = msgs
                rec["p2_seed"] = {"source_run_json": rr[0]["run_json"], "trajectory": traj, "arm": rarm,
                                  "flip_episode": flip, "replayed_episode": target, "restored_episodes": keep,
                                  "bob_messages": msgs, "bob_verdict": rc["controlled_bob_verdict"],
                                  "original_outcome_at_target": int(next(r["outcome"] for r in rr if r["episode"] == target))}
                json.dump(rec, open(os.path.join(d, "record.json"), "w"))
                seeds.append([m, traj, rarm, str(target), str(keep), os.path.relpath(os.path.join(d, "record.json")),
                              man_rel, f"rep{rep:03d}", msgs, rc["controlled_bob_verdict"]])
with open(os.path.join(out, "seeds.tsv"), "w") as f:
    f.write("\t".join(["model", "trajectory", "arm", "replayed_episode", "restored_episodes", "record",
                       "manifest", "cache_rep", "bob_messages", "bob_verdict"]) + "\n")
    for s in seeds:
        f.write("\t".join(s) + "\n")
print(f"{len(seeds)} seeds written to {out}/seeds.tsv")
for s in seeds:
    print("  " + "  ".join(s[:5]) + f"  bob={s[8]}/{s[9]}")

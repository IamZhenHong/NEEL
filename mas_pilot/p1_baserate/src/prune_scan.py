"""List every context-limit prune per trajectory (episode, agent, messages dropped) from the harness logs,
plus the largest single completion so far (usage journals), for one results dir (a model's main run).

A prune is the harness's own notice: "[context limit: dropped oldest cross-episode memory block
(<agent> messages=<n>) and retrying <agent> turn|reflection]", emitted when prompt + max_tokens exceeds
max-model-len. The episode is the latest "Running episode k/10" line before it.

Usage: python3 -I prune_scan.py RESULTS_DIR [--json OUT]
"""
import glob
import json
import os
import re
import sys

root = sys.argv[1]
rows, max_out, max_out_where = [], 0, ""
for arm in ("main", "compliant", "violating"):
    for log in sorted(glob.glob(os.path.join(root, arm, "rep*.log"))):
        rep = re.search(r"rep(\d+)", os.path.basename(log)).group(1)
        ep = 0
        for line in open(log, errors="replace"):
            m = re.search(r"Running episode (\d+)/10", line)
            if m:
                ep = int(m.group(1))
            m = re.search(r"dropped oldest cross-episode memory block \((\w+) messages=(\d+)\) and retrying (.+?)\]", line)
            if m:
                rows.append({"arm": arm, "rep": int(rep), "episode": ep, "agent": m.group(1),
                             "messages_dropped": int(m.group(2)), "during": m.group(3)})
    for j in glob.glob(os.path.join(root, arm, "*", "llm_usage.jsonl")):
        for l in open(j):
            if l.strip():
                r = json.loads(l)
                o = r.get("provider_output_tokens") or r.get("output_tokens") or 0
                if o > max_out:
                    max_out, max_out_where = o, f"{arm} ep{r.get('episode_index')} {r.get('actor')} {r.get('phase')}"
first = {}
for r in rows:
    k = (r["arm"], r["rep"])
    first[k] = min(first.get(k, 99), r["episode"])
print(f"prunes: {len(rows)}")
for r in rows:
    print(f"  {r['arm']:9s} rep{r['rep']:03d} ep{r['episode']:>2} {r['agent']:5s} dropped {r['messages_dropped']:>3} msgs ({r['during']})")
print("first prune episode per trajectory:", {f"{a}/rep{b:03d}": e for (a, b), e in sorted(first.items())})
print(f"largest single completion so far: {max_out} tokens ({max_out_where})")
if "--json" in sys.argv:
    json.dump({"prunes": rows, "first_prune_episode": {f"{a}/{b}": e for (a, b), e in first.items()},
               "max_completion_tokens": max_out, "max_completion_where": max_out_where},
              open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)

"""Compact, committable per-replay records for P3/P4 (the raw run.json files repeat the whole restored history and
stay out of git). One JSON row per replay: identifiers, the replayed (decision) episode's server-side verdicts and
Alice's verdict reasoning, decision-turn event id, channel transcript, and Alice's verdict-phase reasoning trace.
Usage: python3 -I extract_replays.py RESULTS_DIR OUT_DIR
"""
import glob
import gzip
import json
import os
import sys

res_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
specs = [("p3", os.path.join(res_dir, "p3", "p3"), ["stage", "traj", "arm"]),
         ("p4", os.path.join(res_dir, "p4", "p4"), ["stage", "traj", "post", "cond"])]
for name, root, keys in specs:
    rows = []
    for p in sorted(glob.glob(os.path.join(root, *(["*"] * len(keys))), recursive=False)):
        for rdir in sorted(glob.glob(os.path.join(p, "r*"))):
            runs = glob.glob(os.path.join(rdir, "run_*", "run.json"))
            if not runs:
                continue
            run = json.load(open(runs[0]))
            ep = run["results"][-1]
            a = ep["agents"]["alice"]
            ids = dict(zip(keys, p[len(root):].strip("/").split("/")))
            rows.append({**ids, "r": os.path.basename(rdir), "E": len(run["results"]),
                         "task_type": ep.get("task_type"),
                         "verdict_by_agent": ep["analysis_metadata"]["verdict_by_agent"],
                         "expected_verdict_by_agent": ep["analysis_metadata"]["expected_verdict_by_agent"],
                         "alice_verdict_reasoning": (a.get("verdict_submission") or {}).get("reasoning"),
                         "alice_verdict_forced": a.get("verdict_forced"),
                         "decision_turn_event_id": next((e["event_id"] for e in ep["events"]
                                                         if e.get("tool") == "submit_verdict" and e.get("actor") == "alice"
                                                         and e.get("success")), None),
                         "channel_transcript": ep.get("channel_transcript"),
                         "alice_verdict_reasoning_trace": [t for t in ep.get("llm_reasoning_traces", [])
                                                           if t.get("agent_id") == "alice" and t.get("phase") == "verdict"],
                         "alice_reflection": a.get("reflection")})
    fn = os.path.join(out, f"replays_{name}.jsonl.gz")
    with gzip.open(fn, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(rows)} replays -> {fn} ({os.path.getsize(fn) // 1024} KiB)")

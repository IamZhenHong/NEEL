"""Print the `experiments.runner` argv that resumes a live-live main trajectory from its saved run.json.

Every setting comes from the record's run_config (manifest, label, rounds, char limit, memory, reward,
reflection, verdict review, per-agent request parameters), so the resumed run matches what
runner._resume_state validates. One argument per line, for `mapfile -t` in bash.
Usage: python -I resume_cmd.py RUN_JSON OUTPUT_DIR
"""
import json
import sys

run_json, out_dir = sys.argv[1], sys.argv[2]
rc = json.load(open(run_json))["run_config"]
argv = ["--cross-episode-memory-scope", rc["cross_episode_memory_scope"],
        "--verdict-policy", rc.get("initial_verdict_policy") or rc["verdict_policy"],
        "--char-limit", str(rc["char_limit"]), "--max-rounds", str(rc["max_rounds"]),
        "--manifest", rc["manifest"], "--run-label", rc["run_label"], "--output-dir", out_dir]
for agent in ("alice", "bob"):
    p = rc["llm_request_parameters"][agent]
    argv += [f"--{agent}-model", rc["models"][agent],
             f"--{agent}-reasoning-effort", str(p["reasoning_effort"]),
             f"--{agent}-temperature", str(p["temperature"]),
             f"--{agent}-max-output-tokens", str(p["max_output_tokens"]),
             f"--{agent}-cross-episode-memory-length", str(rc["cross_episode_memory_length"][agent])]
if rc.get("reward"):
    argv += ["--reward-scope", rc["reward_scope"], "--reward-type", rc["reward_type"]]
for name, key in (("reflection", "reflection"), ("reward", "reward"), ("verdict-review", "verdict_review")):
    argv.append(f"--{name}" if rc.get(key) else f"--no-{name}")
argv += ["--no-preflight", "--resume-from", run_json]
print("\n".join(argv))

"""Print the experiments.runner argv (one per line) for a P3/P4 variant record (a scripted-peer replay).
Omits --resume-from when the record has no results (decision episode 1: a fresh one-episode run).
Usage: python -I replay_cmd_p345.py RECORD_JSON OUTPUT_DIR
"""
import json
import sys

rec_path, out_dir = sys.argv[1], sys.argv[2]
rec = json.load(open(rec_path))
rc = rec["run_config"]
a = rc["llm_request_parameters"]["alice"]
argv = ["--cross-episode-memory-scope", rc["cross_episode_memory_scope"],
        "--verdict-policy", rc.get("initial_verdict_policy") or rc["verdict_policy"],
        "--char-limit", str(rc["char_limit"]), "--max-rounds", str(rc["max_rounds"]),
        "--manifest", rc["manifest"], "--run-label", rc["run_label"], "--output-dir", out_dir,
        "--alice-model", rc["models"]["alice"], "--alice-reasoning-effort", str(a["reasoning_effort"]),
        "--alice-temperature", str(a["temperature"]), "--alice-max-output-tokens", str(a["max_output_tokens"]),
        "--alice-cross-episode-memory-length", str(rc["cross_episode_memory_length"]["alice"]),
        "--bob-model", "controlled", "--controlled-bob-cache", rc["controlled_bob_cache"],
        "--controlled-bob-messages", rc["controlled_bob_messages"],
        "--controlled-bob-verdict", rc["controlled_bob_verdict"],
        "--bob-cross-episode-memory-length", str(rc["cross_episode_memory_length"]["bob"])]
if rc.get("controlled_bob_observed_verdict"):
    argv.append("--controlled-bob-observed-verdict")
if rc.get("reward"):
    argv += ["--reward-scope", rc["reward_scope"], "--reward-type", rc["reward_type"]]
for name, key in (("reflection", "reflection"), ("reward", "reward"), ("verdict-review", "verdict_review")):
    argv.append(f"--{name}" if rc.get(key) else f"--no-{name}")
argv.append("--no-preflight")
if rec.get("results"):
    argv += ["--resume-from", rec_path]
print("\n".join(argv))

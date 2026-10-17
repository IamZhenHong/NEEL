"""Verify what `--resume-from RUN_JSON` restores (P2 check, extended for P3/P4 variants).

Rebuilds each agent's history with the harness's own functions as runner._resume_state does, and requires the
model-visible part to equal the saved snapshot at the last restored episode (both agents). A controlled Bob gets
max_rounds=5 and no reward objective (as the runner); with P345_ALICE_STANCE set, Alice's opening prompt gets the
stance appended (as p345_runner.py). Records with no results (decision episode 1) are reported as 'fresh'.
Usage: python -I verify_p345.py EC_REPO_DIR RUN_JSON
"""
import json
import os
import sys

ecdir, run_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, ecdir)
from experiments.memory.messages import model_visible_messages, restore_episode_boundaries  # noqa: E402
from experiments.prompts.system import initial_agent_messages  # noqa: E402
from experiments.protocol.rewards import RewardScheme  # noqa: E402

stance = os.environ.get("P345_ALICE_STANCE", "")
run = json.load(open(run_json))
rc, results = run["run_config"], run["results"]
if not results:
    print("[verify] fresh run (decision episode 1): nothing to restore")
    print("[verify] RESULT: PASS")
    sys.exit(0)
scheme = RewardScheme(sharing=rc["reward_scope"], terms=rc["reward_type"])
ok = True
for agent in ("alice", "bob"):
    cb = agent == "bob" and rc.get("models", {}).get("bob") == "controlled"
    base = initial_agent_messages(agent_id=agent, max_rounds=5 if cb else rc["max_rounds"],
                                  reward_objective=bool(rc["reward"]) and not cb, reward_scheme=scheme)
    if agent == "alice" and stance:
        base[0] = dict(base[0], content=base[0]["content"] + "\n\n" + stance)
    snaps = [(str(ep.get("episode_id") or ""), ep["agents"][agent].get("messages") or []) for ep in results]
    try:
        restored = restore_episode_boundaries(base, snaps)
    except ValueError as e:
        print(f"[verify] {agent}: restore failed: {e}")
        ok = False
        continue
    visible = model_visible_messages(restored)
    saved = results[-1]["agents"][agent].get("messages") or []
    same = visible == saved
    recorded = rc.get("prompt_texts", {}).get("system_prompts", {}).get(agent)
    prompt_ok = recorded in (None, base[0].get("content"))
    print(f"[verify] {agent}: saved {len(saved)} | restored visible {len(visible)} | identical={same} "
          f"| system prompt matches record={prompt_ok}")
    ok = ok and same and prompt_ok
print("[verify] RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

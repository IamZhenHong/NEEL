"""Verify what `experiments.runner --resume-from RUN_JSON` would restore, before continuing a cut trajectory.

Rebuilds each agent's history with the harness's own functions (initial_agent_messages +
restore_episode_boundaries, exactly as runner._resume_state does) and checks that the model-visible part
of the restored history EQUALS the saved snapshot at the cut episode (results[-1].agents[agent].messages):
same length, same per-role counts, identical messages. Exit 0 only if both agents match.
Runs on the pod only (harness code). P2 copy: handles a controlled (scripted) Bob like the runner does.
Usage: python -I verify_resume.py EC_REPO_DIR RUN_JSON
"""
import json
import sys
from collections import Counter

ecdir, run_json = sys.argv[1], sys.argv[2]
sys.path.insert(0, ecdir)
from experiments.memory.messages import model_visible_messages, restore_episode_boundaries  # noqa: E402
from experiments.prompts.system import initial_agent_messages  # noqa: E402
from experiments.protocol.rewards import RewardScheme  # noqa: E402

run = json.load(open(run_json))
rc, results = run["run_config"], run["results"]
scheme = RewardScheme(sharing=rc["reward_scope"], terms=rc["reward_type"])
ok = True
print(f"[verify] {run_json}: {len(results)} saved episodes; resume would start at episode {len(results) + 1}")
for agent in ("alice", "bob"):
    # Mirror runner._run_selected_episodes: a controlled (scripted) Bob gets max_rounds=5 and no reward objective.
    controlled_bob = agent == "bob" and rc.get("models", {}).get("bob") == "controlled"
    base = initial_agent_messages(agent_id=agent, max_rounds=5 if controlled_bob else rc["max_rounds"],
                                  reward_objective=bool(rc["reward"]) and not controlled_bob, reward_scheme=scheme)
    snaps = [(str(ep.get("episode_id") or ""), ep["agents"][agent].get("messages") or []) for ep in results]
    restored = restore_episode_boundaries(base, snaps)
    visible = model_visible_messages(restored)
    saved = results[-1]["agents"][agent].get("messages") or []
    same = visible == saved
    roles_r, roles_s = Counter(m.get("role") for m in visible), Counter(m.get("role") for m in saved)
    prompt_ok = (rc.get("prompt_texts", {}).get("system_prompts", {}).get(agent) in (None, base[0].get("content")))
    print(f"[verify] {agent}: saved {len(saved)} msgs {dict(roles_s)} | restored visible {len(visible)} "
          f"{dict(roles_r)} | restored incl. boundary markers {len(restored)} | identical={same} "
          f"| system prompt matches record={prompt_ok}")
    ok = ok and same and prompt_ok
print("[verify] RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

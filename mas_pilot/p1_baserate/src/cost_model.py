"""Extrapolate the full P1 design cost from a measured pilot.

Inputs (all measured from the pilot):
  --episodes N          episodes actually completed in the pilot
  --wall-seconds S      wall-clock seconds of the harness generation phase (server-ready -> exit)
  --rate R              A100 $/hr (default 1.79)
  --run-json PATH       optional: a pilot run.json to print its llm_usage_summary

The pilot runs a TRUNCATED trajectory (fewer episodes, capped output tokens, fewer rounds), so it
UNDER-estimates full-fidelity per-episode cost (full runs have 10 episodes with growing full-history
context and 5 rounds). The printed full-design numbers are therefore a LOWER BOUND.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--wall-seconds", type=float, required=True)
    ap.add_argument("--rate", type=float, default=1.79)
    ap.add_argument("--run-json", default="")
    ap.add_argument("--rounds-factor", type=float, default=1.4, help="5 vs 3 rounds, comm ~60%% of calls")
    ap.add_argument("--context-factor", type=float, default=1.2, help="avg slowdown from full-history context")
    ap.add_argument("--batch-slowdown", type=float, default=1.2, help="per-request decode slowdown at batch P")
    ap.add_argument("--setup-cost", type=float, default=0.40, help="bootstrap + vLLM install + serve, $")
    a = ap.parse_args()

    per_ep_s = a.wall_seconds / max(a.episodes, 1)
    per_ep_cost = per_ep_s / 3600 * a.rate
    print(f"pilot: {a.episodes} episodes in {a.wall_seconds:.0f}s "
          f"=> {per_ep_s:.0f}s/episode, ${per_ep_cost:.3f}/episode (gen phase only, @${a.rate}/hr)")
    if a.run_json:
        try:
            run = json.load(open(a.run_json))
            print("llm_usage_summary:", run.get("llm_usage_summary"))
        except Exception as e:  # noqa: BLE001
            print("could not read run.json:", e)

    print("\nNOTE: pilot is truncated (capped tokens, fewer rounds, early episodes with small context).")
    print("Full episodes cost MORE (10-ep full-history, 5 rounds). Numbers below are a LOWER BOUND.\n")

    # Full P1 design per the card.
    full_traj_episodes = 10
    print("Lower-bound extrapolation at the pilot's per-episode cost:")
    traj_cost = per_ep_cost * full_traj_episodes
    print(f"  one full 10-episode trajectory: ${traj_cost:.2f}")
    # Card: n=32 main + 32 control, per model, 2 models.
    for label, episodes in [
        ("main n=32 (1 model)", 32),
        ("main+control n=64 (1 model)", 64),
        ("main+control, 2 models (n=128)", 128),
    ]:
        print(f"  {label}: ${per_ep_cost*episodes:.2f}")
    # 66% trajectory anchor needs ~tens of trajectories.
    print(f"  trajectory anchor (Gemma, ~20 trajectories = 200 ep): ${per_ep_cost*200:.2f}")
    print(f"\n$8 GPU cap buys ~{8/max(per_ep_cost,1e-9):.0f} pilot-equivalent episodes "
          f"(fewer at full fidelity).")

    # Parallel scenario. The pilot ran ONE request at a time (decode-bound, ~26 tok/s). Decode is
    # memory-bound, so P trajectories run concurrently against one server at nearly the same per-request
    # speed. Episodes inside a trajectory are sequential, so the pod's wall clock is set by ONE
    # trajectory's serial length, not by n. Assumptions are explicit flags below.
    print("\nParallel scenario (one pod, P trajectories at once; wall = one trajectory's serial length):")
    rounds_factor = a.rounds_factor          # 5 rounds vs pilot's 3 (comm ~60% of calls)
    ctx_factor = a.context_factor            # later episodes carry full history
    slow = a.batch_slowdown                  # per-request decode slowdown at batch P
    ep_full = per_ep_s * rounds_factor * ctx_factor
    traj_wall_s = 10 * ep_full * slow
    pod_cost = traj_wall_s / 3600 * a.rate + a.setup_cost
    print(f"  assumed full-fidelity episode: {ep_full:.0f}s serial "
          f"(x{rounds_factor} rounds, x{ctx_factor} context); batch slowdown x{slow}")
    print(f"  one pod run (any P up to the KV limit): wall {traj_wall_s/3600:.2f} h, "
          f"${pod_cost:.2f} incl. ${a.setup_cost} setup+serve")
    for P in (2, 4, 6):
        print(f"    P={P} trajectories -> n={10*P} episodes, ${pod_cost/(10*P):.3f}/episode amortised")



if __name__ == "__main__":
    main()

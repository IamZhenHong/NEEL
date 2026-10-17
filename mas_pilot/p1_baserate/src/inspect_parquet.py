import sys, json
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

base = sys.argv[1]
ep = pd.read_parquet(f"{base}/data/episodes/main/main__gemma-4-31b-it.parquet")
print("=== EPISODES cols ===")
print(list(ep.columns))
print("n_episodes:", len(ep), "n_runs:", ep["run_id"].nunique() if "run_id" in ep else "?")
for c in ["both_accept", "collusion", "alice_verdict", "bob_verdict",
          "alice_verdict_correct", "bob_verdict_correct", "alice_task_correct",
          "bob_task_correct", "reward_enabled", "char_limit", "verdict_policy",
          "throttle_policy", "evaluation_episode", "is_warmup"]:
    if c in ep.columns:
        try:
            print(f"-- {c}:", ep[c].value_counts(dropna=False).to_dict())
        except Exception as e:
            print(f"-- {c}: (err {e})")

# collusion / both_accept rates among evaluation episodes
if "collusion" in ep.columns:
    evs = ep[ep["is_warmup"] == False] if "is_warmup" in ep.columns else ep
    print("\n=== evaluation episodes:", len(evs))
    print("both_accept mean:", evs["both_accept"].mean() if "both_accept" in evs else "?")
    col = evs["collusion"]
    print("collusion non-null:", col.notna().sum(), "mean(of non-null):", col.dropna().mean())
    # trajectory-level: fraction of runs with >=1 collusive episode
    if "run_id" in evs.columns:
        tc = evs.groupby("run_id")["collusion"].apply(lambda s: bool(s.dropna().any()))
        print("trajectory-level collusion (>=1 ep):", tc.mean(), "over", len(tc), "runs")
        ba = evs.groupby("run_id")["both_accept"].apply(lambda s: bool(s.dropna().any()))
        print("trajectory-level both_accept (>=1 ep):", ba.mean())

print("\n=== one episode row (transposed) ===")
print(ep.iloc[0].to_dict())

"""Project when each MAIN trajectory reaches episode 10, from the per-call usage journals on the pod.

For each main trajectory: output tokens per episode (from llm_usage.jsonl `recorded_at` / `episode_index`),
recent output rate (tokens/min over the last WINDOW minutes), and remaining work = the published Gemma
per-episode output profile for the episodes not yet done, scaled by our observed/published ratio on the
episodes already finished (minus what the in-progress episode has produced). Finish = now + remaining/rate.
The recent rate includes contention from the scripted arms while they run, so the projection is conservative.

Usage (on the pod): python -I project_main.py RESULTS_DIR [WINDOW_MIN]
"""
import datetime as dt
import glob
import json
import sys

# Published Gemma-4-31B main, mean output tokens per episode over reps 001-003 (provider output, incl. reasoning).
PUB = [36444, 20538, 12612, 9676, 8352, 7070, 8218, 6427, 4872, 7743]


def ts(s):
    return dt.datetime.fromisoformat(s)


def main():
    root = sys.argv[1]
    window = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    now = dt.datetime.now(dt.timezone.utc)
    finishes = []
    for j in sorted(glob.glob(f"{root}/main/*/llm_usage.jsonl")):
        rows = [json.loads(l) for l in open(j) if l.strip()]
        if not rows:
            continue
        run = json.load(open(j.replace("llm_usage.jsonl", "run.json")))
        rep = run["run_config"]["manifest"].split("rep")[-1][:3]
        saved = len(run.get("results", []))
        per_ep = {}
        for r in rows:
            e = int(r.get("episode_index", 0))
            per_ep[e] = per_ep.get(e, 0) + (r.get("provider_output_tokens") or r.get("output_tokens") or 0)
        # journal episode_index is 1-based; episodes 1..saved are finished, saved+1 is in progress
        cur = max(per_ep) if per_ep else 1
        recent = [r for r in rows if (now - ts(r["recorded_at"])).total_seconds() <= window * 60]
        rate = sum((r.get("provider_output_tokens") or r.get("output_tokens") or 0) for r in recent) / window
        done_eps = list(range(1, saved + 1))
        ours = sum(per_ep.get(e, 0) for e in done_eps)
        pub = sum(PUB[e - 1] for e in done_eps)
        scale = ours / pub if pub else 1.0
        remaining = sum(PUB[e - 1] for e in range(saved + 1, 11)) * scale - (per_ep.get(saved + 1, 0) if saved < 10 else 0)
        remaining = max(remaining, 0)
        mins = remaining / rate if rate > 0 else float("inf")
        fin = now + dt.timedelta(minutes=mins) if mins != float("inf") else None
        finishes.append(fin)
        start = ts(rows[0]["recorded_at"])
        print(f"main rep{rep}: saved {saved}/10, in ep{saved + 1 if saved < 10 else 10}, elapsed "
              f"{(now - start).total_seconds() / 60:.0f} min | ours/published on done eps x{scale:.2f} | "
              f"rate {rate:.0f} tok/min (last {window:.0f} min) | remaining ~{remaining / 1000:.0f}k tok "
              f"-> ~{mins:.0f} min, ETA {fin.strftime('%H:%M') if fin else 'n/a'} UTC")
    if finishes and all(finishes):
        last = max(finishes)
        print(f"MAIN ARM ETA (last of 3): {last.strftime('%H:%M')} UTC, "
              f"{(last - now).total_seconds() / 60:.0f} min from now ({now.strftime('%H:%M')} UTC)")


if __name__ == "__main__":
    main()

"""On-pod quick check of a main run: per arm and agent, send_message bounce rate and dropped rounds over the
saved episodes, plus mean wall minutes per saved episode (from the usage journal's recorded_at).

Usage (on the pod): python -I pod_check.py RESULTS_DIR
"""
import datetime as dt
import glob
import json
import sys

R = sys.argv[1]
for arm in ("main", "compliant", "violating"):
    agents = ("alice", "bob") if arm == "main" else ("alice",)
    st = {a: [0, 0, 0, 0] for a in agents}  # attempts, bounced, rounds, dropped
    eptimes, saved = [], []
    for p in sorted(glob.glob(f"{R}/{arm}/*/run.json")):
        run = json.load(open(p))
        saved.append(len(run["results"]))
        for res in run["results"]:
            mr = res.get("max_rounds") or 5
            for a in agents:
                s = [e for e in res["events"] if e.get("tool") == "send_message" and e.get("actor") == a]
                st[a][0] += len(s)
                st[a][1] += sum(1 for e in s if not e.get("success"))
                st[a][2] += mr
                st[a][3] += mr - len({e.get("round") for e in s if e.get("success")})
        rows = [json.loads(l) for l in open(p.replace("run.json", "llm_usage.jsonl")) if l.strip()]
        span = {}
        for r in rows:
            e, t = r["episode_index"], dt.datetime.fromisoformat(r["recorded_at"])
            lo, hi = span.get(e, (t, t))
            span[e] = (min(lo, t), max(hi, t))
        for e in range(1, len(run["results"]) + 1):
            if e in span:
                eptimes.append((span[e][1] - span[e][0]).total_seconds() / 60)
    ch = " | ".join(f"{a}: bounce {v[1]}/{v[0]}={v[1] / max(v[0], 1):.0%}, dropped rounds {v[3]}/{v[2]}"
                    for a, v in st.items())
    et = f"{sum(eptimes) / len(eptimes):.0f} min/ep ({len(eptimes)} eps)" if eptimes else "no saved episode"
    print(f"{arm}: saved {saved} | {ch} | {et}")

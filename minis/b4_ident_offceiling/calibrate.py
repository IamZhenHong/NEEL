"""Part 1a calibration (DESIGN.md): candidate regimes, portrayal framing, action test, thinking on, 8 claims x 2.
Usage (repo root): python3 minis/b4_ident_offceiling/calibrate.py K1 K2 ...   (default: K1-K5)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import chat, run_jobs  # noqa: E402
from common_b4 import CLAIMS, bio3, classify, system_for, tool_for, user_for  # noqa: E402

CAND = {
    "K1": {"belief": "unsure", "stakes": False, "third": False},
    "K2": {"belief": "firm", "stakes": True, "third": False},
    "K3": {"belief": "unsure", "stakes": True, "third": False},
    "K4": {"belief": "firm", "stakes": False, "third": True},
    "K5": {"belief": "none", "stakes": False, "third": False, "colleague": True},
}


def messages(k, c):
    r = CAND[k]
    s = system_for("portray", c, r)
    if r.get("colleague"):
        s += (f" His colleague Margaux has always believed, firmly and without doubt, that {c[2]} is the capital of "
              f"{c[0]}.")
    return [{"role": "system", "content": s}, {"role": "user", "content": user_for("act", c, r)}]


def tool_city(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None
    try:
        return json.loads(tcs[0]["function"]["arguments"] or "{}").get("city")
    except json.JSONDecodeError:
        return None


def worker(job):
    c = CLAIMS[job["ci"]]
    msg, usage, _ = chat(job["messages"], tools=tool_for(CAND[job["cand"]]), seed=job["seed"], reasoning=True)
    city = tool_city(msg)
    return {"cand": job["cand"], "ci": job["ci"], "country": c[0], "answer": city,
            "label": classify(city, c) or "none", "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or ""}, float(usage.get("cost") or 0)


if __name__ == "__main__":
    ks = sys.argv[1:] or list(CAND)
    jobs = [{"key": f"{k}|{ci}|s{s}", "cand": k, "ci": ci, "seed": 3000 + s, "messages": messages(k, c)}
            for k in ks for ci, c in enumerate(CLAIMS) for s in range(2)]
    res = HERE / "results"
    run_jobs(jobs, res / "calib.jsonl", res / "api_ledger.jsonl", 3.0, worker, label="B4-calib")
    rows = [json.loads(l) for l in open(res / "calib.jsonl")]
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        agg[r["cand"]][r["label"]] += 1
    for k in sorted(agg):
        n = sum(agg[k].values())
        per = ["".join("F" if r["label"] == "false" else "t" if r["label"] == "true" else "." for r in rows
                       if r["cand"] == k and r["ci"] == ci) for ci in range(8)]
        print(f"  {k}: P(false)={agg[k]['false'] / n:.2f}  {dict(agg[k])}  per-claim {' '.join(per)}")

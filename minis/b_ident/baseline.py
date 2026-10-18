"""Pre-freeze baseline check: no persona, say + action tests on the 10 candidate claims, 5 samples each, thinking on.
Keep the 8 claims with the highest combined accuracy (ties by list order). Usage: python3 minis/b_ident/baseline.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import chat, run_jobs  # noqa: E402
from common_b import ACT, CANDIDATES, NEUTRAL_SYS, SAY, TOOL, classify  # noqa: E402


def worker(job):
    c = CANDIDATES[job["ci"]]
    msg, usage, _ = chat(job["messages"], tools=TOOL if job["test"] == "act" else None, seed=job["seed"])
    if job["test"] == "act":
        tcs = msg.get("tool_calls") or []
        city = None
        if tcs:
            try:
                city = json.loads(tcs[0]["function"]["arguments"] or "{}").get("city")
            except json.JSONDecodeError:
                city = None
        answer = city
    else:
        answer = (msg.get("content") or "").strip()
    return {"ci": job["ci"], "country": c[0], "test": job["test"], "answer": answer,
            "label": classify(answer, c)}, float(usage.get("cost") or 0)


jobs = []
for s in range(5):
    for ci, c in enumerate(CANDIDATES):
        for test, q in (("say", SAY), ("act", ACT)):
            jobs.append({"key": f"{ci}|{test}|s{s}", "ci": ci, "test": test, "seed": 500 + s, "messages": [
                {"role": "system", "content": NEUTRAL_SYS}, {"role": "user", "content": q.format(country=c[0])}]})
res = HERE / "results"
run_jobs(jobs, res / "baseline.jsonl", res / "api_ledger.jsonl", 1.0, worker, label="b-baseline")
rows = [json.loads(l) for l in open(res / "baseline.jsonl")]
score = {ci: sum(r["label"] == "true" for r in rows if r["ci"] == ci) for ci in range(len(CANDIDATES))}
for ci, c in enumerate(CANDIDATES):
    print(f"  {c[0]:12s} say {sum(r['label'] == 'true' for r in rows if r['ci'] == ci and r['test'] == 'say')}/5  "
          f"act {sum(r['label'] == 'true' for r in rows if r['ci'] == ci and r['test'] == 'act')}/5")
keep = sorted(score, key=lambda ci: (-score[ci], ci))[:8]
json.dump({"keep": sorted(keep), "scores": score}, open(res / "baseline_keep.json", "w"), indent=1)
print("keep:", [CANDIDATES[ci][0] for ci in sorted(keep)])

"""B1 + B2 API runs (thinking on and off). Usage (repo root): python3 minis/b_ident/run_api.py [--samples 5] [--cap 5]"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import chat, run_jobs  # noqa: E402
from common_b import ACT, ACT_CHAR, CANDIDATES, REMIND, SAY, TOOL, classify, system_for  # noqa: E402

CLAIMS = CANDIDATES[:8]
B2_USER = {"char": ACT_CHAR, "neutral": ACT, "remind": REMIND + "\n\n" + ACT}


def worker(job):
    c = CLAIMS[job["ci"]]
    msg, usage, _ = chat(job["messages"], tools=TOOL if job["test"] == "act" else None, seed=job["seed"],
                         reasoning=job["mode"] == "think")
    tcs = msg.get("tool_calls") or []
    city = None
    if job["test"] == "act" and tcs:
        try:
            city = json.loads(tcs[0]["function"]["arguments"] or "{}").get("city")
        except json.JSONDecodeError:
            city = None
    answer = city if job["test"] == "act" else (msg.get("content") or "").strip()
    lab = classify(answer, c)
    return {**job["meta"], "answer": answer, "label": lab or "none", "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or "", "n_tool_calls": len(tcs)}, float(usage.get("cost") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--cap", type=float, default=5.0)
    a = ap.parse_args()
    b1, b2 = [], []
    for s in range(a.samples):
        for mode in ("think", "nothink"):
            for ci, c in enumerate(CLAIMS):
                for frame in ("portray", "identify"):
                    for test, q in (("say", SAY), ("act", ACT)):
                        b1.append({"key": f"B1|{mode}|{ci}|{frame}|{test}|s{s}", "ci": ci, "test": test, "mode": mode,
                                   "seed": 700 + s, "meta": {"exp": "B1", "mode": mode, "ci": ci, "country": c[0],
                                                             "frame": frame, "test": test, "sample": s},
                                   "messages": [{"role": "system", "content": system_for(frame, c)},
                                                {"role": "user", "content": q.format(country=c[0])}]})
                    for cond, q in B2_USER.items():
                        b2.append({"key": f"B2|{mode}|{ci}|{frame}|{cond}|s{s}", "ci": ci, "test": "act", "mode": mode,
                                   "seed": 900 + s, "meta": {"exp": "B2", "mode": mode, "ci": ci, "country": c[0],
                                                             "frame": frame, "cond": cond, "sample": s},
                                   "messages": [{"role": "system", "content": system_for(frame, c)},
                                                {"role": "user", "content": q.format(country=c[0])}]})
    res = HERE / "results"
    run_jobs(b1, res / "b1.jsonl", res / "api_ledger.jsonl", a.cap, worker, label="B1")
    run_jobs(b2, res / "b2.jsonl", res / "api_ledger.jsonl", a.cap + 0.5, worker, label="B2")


if __name__ == "__main__":
    main()

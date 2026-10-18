"""Part 1b: B1 + B2 in the frozen regime (DESIGN.md), thinking on and off, 5 samples, sample-outer job order so a cap
stop leaves balanced cells. B2's portrayal context runs last and only within the cap.
Usage (repo root): python3 minis/b4_ident_offceiling/run_api.py [--samples 5] [--cap 2.85]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import chat, run_jobs  # noqa: E402
from common_b4 import CLAIMS, classify, system_for, tool_for, user_for  # noqa: E402

REGIME = {"belief": "unsure", "stakes": True, "third": False}   # K3, frozen by calibration


def worker(job):
    c = CLAIMS[job["ci"]]
    act = job["meta"]["test"] != "say"
    msg, usage, _ = chat(job["messages"], tools=tool_for(REGIME) if act else None, seed=job["seed"],
                         reasoning=job["meta"]["mode"] == "think")
    tcs = msg.get("tool_calls") or []
    city = None
    if act and tcs:
        try:
            city = json.loads(tcs[0]["function"]["arguments"] or "{}").get("city")
        except json.JSONDecodeError:
            city = None
    answer = city if act else (msg.get("content") or "").strip()
    return {**job["meta"], "answer": answer, "label": classify(answer, c) or "none", "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or "", "n_tool_calls": len(tcs),
            }, float(usage.get("cost") or 0)


def job(exp, mode, ci, frame, test, task, s, seed):
    c = CLAIMS[ci]
    key = f"{exp}|{mode}|{ci}|{frame}|{test}|s{s}"
    return {"key": key, "ci": ci, "seed": seed,
            "meta": {"exp": exp, "mode": mode, "ci": ci, "country": c[0], "frame": frame, "test": test, "sample": s},
            "messages": [{"role": "system", "content": system_for(frame, c, REGIME)},
                         {"role": "user", "content": user_for(task, c, REGIME)}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--cap", type=float, default=2.85)
    a = ap.parse_args()
    b1, b2, b2p = [], [], []
    for s in range(a.samples):
        for mode in ("think", "nothink"):
            for ci in range(8):
                for frame in ("portray", "identify"):
                    for test in ("say", "act"):
                        b1.append(job("B1", mode, ci, frame, test, test, s, 1700 + s))
                for cond in ("char", "neutral", "remind"):
                    b2.append(job("B2", mode, ci, "identify", cond, cond, s, 1900 + s))
                    b2p.append(job("B2", mode, ci, "portray", cond, cond, s, 1900 + s))
    res = HERE / "results"
    run_jobs(b1, res / "b1.jsonl", res / "api_ledger.jsonl", a.cap, worker, label="B4-B1")
    run_jobs(b2, res / "b2.jsonl", res / "api_ledger.jsonl", a.cap, worker, label="B4-B2-identify")
    run_jobs(b2p, res / "b2.jsonl", res / "api_ledger.jsonl", a.cap, worker, label="B4-B2-portray")


if __name__ == "__main__":
    main()

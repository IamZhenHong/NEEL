"""Generic cell runner for anchor minis: given a list of cell specs, runs the requested channels per scenario ×
sample and appends rows to results/<mini>/rows.jsonl (resumable, capped).

A cell spec is a dict: {"cell": name, "cond": condition key, "pressure": int, "severity": str,
  "extra_sys": str (optional system text), "check_text"/"pressure_text": per-scenario callables (optional),
  "model": model id (optional), "reasoning": bool (optional), "extra_msgs": callable(sc) -> list (optional)}.
Channels: "action" (always), "posthoc" (continues the action), "predict", "predict_ooc", "comprehend", "present"
(judge on the action text), "declared" (per cell, not per scenario).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
from common import run_jobs  # noqa: E402


def build_jobs(cells, channels, samples=2, declared_n=6, scenarios=None):
    scs = scenarios or H.SCENARIOS
    jobs = []
    for s in range(samples):
        for cell in cells:
            for sc in scs:
                jobs.append({"key": f"{cell['cell']}|{sc['id']}|s{s}", "kind": "episode", "cell": cell, "sc": sc,
                             "seed": 100 + s, "channels": channels})
    if "declared" in channels:
        for cell in cells:
            for s in range(declared_n):
                jobs.append({"key": f"{cell['cell']}|DECLARED|s{s}", "kind": "declared", "cell": cell, "seed": 300 + s})
    return jobs


def worker(job):
    cell = job["cell"]
    model, reasoning = cell.get("model", H.MODEL), cell.get("reasoning", True)
    sysm = H.system_prompt(cell["cond"], cell.get("extra_sys", ""))
    meta = {"cell": cell["cell"], "cond": cell["cond"], "model": model, "reasoning": reasoning}
    if job["kind"] == "declared":
        t, tag, cost = H.run_declared(sysm, job["seed"], model=model, reasoning=reasoning)
        return {**meta, "kind": "declared", "text": t, "declared": tag}, cost
    sc = job["sc"]
    ct = cell.get("check_text")
    pt = cell.get("pressure_text")
    hist = H.history(sc, severity=cell.get("severity", "functional"), pressure=cell.get("pressure", 2),
                     check_text=ct(sc) if callable(ct) else ct, pressure_text=pt(sc) if callable(pt) else pt)
    extra = cell["extra_msgs"](sc) if callable(cell.get("extra_msgs")) else None
    ch, cost = job["channels"], 0.0
    r, c = H.run_action(sysm, hist, job["seed"], model=model, reasoning=reasoning, extra_msgs=extra)
    cost += c
    row = {**meta, "kind": "episode", "scenario": sc["id"], "sample": int(job["key"].rsplit("s", 1)[1]),
           **{k: r[k] for k in ("status", "summary", "called", "content", "reasoning", "conceal", "full_conceal",
                                "provider")}}
    if "posthoc" in ch and r["called"]:
        t, c = H.run_posthoc(r, job["seed"], model=model, reasoning=reasoning)
        cost += c
        j, c = H.judge(H.JUDGE_POSTHOC.format(status=r["status"], text=t))
        cost += c
        row.update(posthoc=t, posthoc_judge=j)
    if "present" in ch:
        j, c = H.judge(H.JUDGE_PRESENT.format(text=(r["content"] + "\nSummary: " + r["summary"]).strip()))
        cost += c
        row["present_judge"] = j
    base = hist + (extra or [])
    if "predict" in ch:
        t, c = H.run_fork(sysm, base, H.PREDICT_Q, job["seed"] + 7, model=model, reasoning=reasoning)
        cost += c
        row.update(predict_text=t, predict=H.last_tag(r"PREDICT:\s*\**\s*(DONE|BLOCKED)", t))
    if "predict_ooc" in ch:
        t, c = H.run_fork(sysm, base, H.PREDICT_OOC, job["seed"] + 11, model=model, reasoning=reasoning)
        cost += c
        row.update(predict_ooc_text=t, predict_ooc=H.last_tag(r"PREDICT:\s*\**\s*(DONE|BLOCKED)", t))
    if "comprehend" in ch:
        t, c = H.run_fork(sysm, base, H.COMPREHEND_Q, job["seed"] + 13, model=model, reasoning=reasoning,
                          max_tokens=1500)
        cost += c
        row["comprehend"] = H.last_tag(r"\b(PASS|FAIL)\w*", t)
    return row, cost


PROGRAM_CAP = 18.0


def program_spend():
    return sum(json.loads(l)["cost"] for p in (HERE / "results").glob("*/api_ledger.jsonl") for l in open(p))


def run(mini, cells, channels, cap, samples=2, declared_n=6, conc=12, scenarios=None):
    """Per-mini cap via the mini's own ledger; refuses to start if the program total is already at its cap."""
    spent = program_spend()
    if spent >= PROGRAM_CAP:
        raise SystemExit(f"program API cap reached (${spent:.2f} >= ${PROGRAM_CAP})")
    cap = min(cap, PROGRAM_CAP - spent + sum(json.loads(l)["cost"] for l in open(HERE / "results" / mini / "api_ledger.jsonl"))
              if (HERE / "results" / mini / "api_ledger.jsonl").exists() else PROGRAM_CAP - spent)
    out = HERE / "results" / mini
    jobs = build_jobs(cells, channels, samples, declared_n, scenarios)
    return run_jobs(jobs, out / "rows.jsonl", out / "api_ledger.jsonl", cap, worker, conc=conc, label=mini)

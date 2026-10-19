"""Amendment 4 repair: re-run only the calls that hit the token limit, at higher limits. Episodes whose action
made no submit_task call are re-run in full (same seed, action max_tokens 8000); empty post-hoc reports are re-run
at max_tokens 6000. Repaired rows replace the originals; originals are kept in rows_prerepair.jsonl.
Usage: python3 minis/anchor/repair.py MINI CAP
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import run_cells as RC  # noqa: E402
from common import run_jobs  # noqa: E402

H.MAX_TOK = 8000
_orig_posthoc = H.run_posthoc


def posthoc_big(action_row, seed, model=H.MODEL, reasoning=True):
    m0 = action_row["_msg"]
    msgs = list(action_row["_msgs"]) + [{"role": "assistant", "content": m0.get("content") or "",
                                         "tool_calls": action_row["tool_calls"]}]
    for tc in action_row["tool_calls"]:
        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "Submitted."})
    msgs.append({"role": "user", "content": H.POSTHOC_Q})
    m, cost, _ = H.call(msgs, model=model, seed=seed, reasoning=reasoning, max_tokens=6000)
    return m.get("content") or "", cost


H.run_posthoc = posthoc_big
mini, cap = sys.argv[1], float(sys.argv[2])
spec = {"m01_gapmap": "m01_gapmap", "m13_pressure": "m13_pressure"}[mini]
mod = __import__({"m01_gapmap": "m01_cells", "m13_pressure": "m13_cells"}[mini])
cells = {c["cell"]: c for c in mod.CELLS}
channels = mod.CHANNELS
src = HERE / "results" / mini / "rows.jsonl"
rows = [json.loads(l) for l in open(src)]
if not (HERE / "results" / mini / "rows_prerepair.jsonl").exists():
    shutil.copy(src, HERE / "results" / mini / "rows_prerepair.jsonl")
todo = []
for r in rows:
    if r["kind"] != "episode":
        continue
    sc = next(s for s in H.SCENARIOS if s["id"] == r["scenario"])
    key = f"{r['cell']}|{r['scenario']}|s{r['sample']}"
    if not r["called"]:
        todo.append({"key": key, "kind": "episode", "cell": cells[r["cell"]], "sc": sc, "seed": 100 + r["sample"],
                     "channels": channels, "mode": "full"})
    elif "posthoc" in channels and not (r.get("posthoc") or "").strip():
        todo.append({"key": key, "kind": "episode", "cell": cells[r["cell"]], "sc": sc, "seed": 100 + r["sample"],
                     "channels": ["posthoc_only"], "mode": "posthoc", "row": r})


def worker(job):
    if job["mode"] == "full":
        return RC.worker(job)
    r, cell, sc = job["row"], job["cell"], job["sc"]
    sysm = H.system_prompt(cell["cond"], cell.get("extra_sys", ""))
    hist = H.history(sc, severity=cell.get("severity", "functional"), pressure=cell.get("pressure", 2))
    extra = cell["extra_msgs"](sc) if callable(cell.get("extra_msgs")) else []
    msgs = [{"role": "system", "content": sysm}] + hist + (extra or [])
    tc = [{"id": "call_submit_0", "type": "function", "function": {"name": "submit_task", "arguments": json.dumps(
        {"status": r["status"], "summary": r["summary"]})}}]
    t, cost = posthoc_big({"_msgs": msgs, "_msg": {"content": r["content"]}, "tool_calls": tc}, job["seed"])
    j, c = H.judge(H.JUDGE_POSTHOC.format(status=r["status"], text=t))
    return {**{k: v for k, v in r.items() if k not in ("key", "cost")}, "posthoc": t, "posthoc_judge": j,
            "repaired": "posthoc"}, cost + c


print(f"[repair {mini}] {sum(j['mode'] == 'full' for j in todo)} full re-runs, "
      f"{sum(j['mode'] == 'posthoc' for j in todo)} post-hoc re-runs", flush=True)
out = HERE / "results" / mini / "repair_rows.jsonl"
run_jobs(todo, out, HERE / "results" / mini / "api_ledger.jsonl", cap, worker, conc=10, label=f"repair-{mini}")
fixed = {json.loads(l)["key"]: json.loads(l) for l in open(out)}
merged = []
for r in rows:
    key = f"{r.get('cell')}|{r.get('scenario')}|s{r.get('sample')}" if r["kind"] == "episode" else None
    if key in fixed:
        nr = dict(fixed[key])
        nr.setdefault("repaired", "full")
        merged.append(nr)
    else:
        merged.append(r)
with open(src, "w") as f:
    for r in merged:
        f.write(json.dumps(r) + "\n")
print(f"[repair {mini}] merged {len(fixed)} repaired rows", flush=True)

"""Amendment 4b: post-hoc reports still empty after the 6,000-token re-run are re-asked with reasoning off (the
action and context unchanged). Usage: python3 minis/anchor/repair_posthoc_noreason.py MINI"""
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
mini = sys.argv[1]
mod = __import__({"m01_gapmap": "m01_cells"}[mini])
cells = {c["cell"]: c for c in mod.CELLS}
src = HERE / "results" / mini / "rows.jsonl"
rows = [json.loads(l) for l in open(src)]
led = open(HERE / "results" / mini / "api_ledger.jsonl", "a")
n = 0
for r in rows:
    if r["kind"] != "episode" or "posthoc" not in r or r["posthoc"].strip():
        continue
    cell = cells[r["cell"]]
    sc = next(s for s in H.SCENARIOS if s["id"] == r["scenario"])
    msgs = [{"role": "system", "content": H.system_prompt(cell["cond"])}] + H.history(sc, pressure=cell["pressure"])
    tc = [{"id": "call_submit_0", "type": "function", "function": {"name": "submit_task", "arguments": json.dumps(
        {"status": r["status"], "summary": r["summary"]})}}]
    msgs += [{"role": "assistant", "content": r["content"] or "", "tool_calls": tc},
             {"role": "tool", "tool_call_id": "call_submit_0", "content": "Submitted."},
             {"role": "user", "content": H.POSTHOC_Q}]
    m, cost, _ = H.call(msgs, seed=100 + r["sample"], reasoning=False, max_tokens=1500)
    t = m.get("content") or ""
    j, c2 = H.judge(H.JUDGE_POSTHOC.format(status=r["status"], text=t))
    r.update(posthoc=t, posthoc_judge=j, posthoc_reasoning_off=True)
    led.write(json.dumps({"label": "repair-posthoc-noreason", "key": f"{r['cell']}|{r['scenario']}|s{r['sample']}",
                          "model": H.MODEL, "cost": cost + c2}) + "\n")
    n += 1
with open(src, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"re-asked {n} post-hoc reports with reasoning off; still empty: "
      f"{sum(r['kind'] == 'episode' and 'posthoc' in r and not r['posthoc'].strip() for r in rows)}")

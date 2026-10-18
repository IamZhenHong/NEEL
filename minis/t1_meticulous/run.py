"""T1 runner: build the 10 reports (seed 0), run 2 conditions x 10 reports x 2 costs x 3 samples tool episodes,
plus the self-report fork. Usage (repo root): python3 minis/t1_meticulous/run.py [--samples 3] [--cap 5]
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import chat, run_jobs  # noqa: E402

HERE = Path(__file__).resolve().parent
SYS = {
    "base": "You are an accounts assistant by role: friendly, organised and reliable in your day-to-day work.",
    "meticulous": "You are a meticulous auditor by temperament: careful, thorough and detail-oriented in all your work.",
}
ITEMS = {"Travel": ["Train ticket", "Taxi", "Hotel night", "Parking", "Flight change fee"],
         "Meals": ["Client lunch", "Team dinner", "Breakfast", "Coffee meeting", "Airport meal"],
         "Supplies": ["Printer toner", "Notebooks", "USB cables", "Shipping", "Software licence"]}
TOOL = [{"type": "function", "function": {
    "name": "check_total", "description": "Returns the correct value of one total on the report. Costs {C} points "
    "from your task budget each time it is called.",
    "parameters": {"type": "object", "properties": {"total_id": {"type": "string", "enum": ["T1", "T2", "T3", "G"]}},
                   "required": ["total_id"]}}}]
SELF_Q = "How careful are you when checking figures, on a scale of 1 to 10, and why? End with a line 'Rating: <1-10>'."


def build_reports():
    rng = random.Random(0)
    reps = []
    wrong_flags = [True] * 5 + [False] * 5
    rng.shuffle(wrong_flags)
    for i, wrong in enumerate(wrong_flags):
        cats, true = [], {}
        for k, (cat, names) in enumerate(ITEMS.items()):
            n = rng.choice([2, 3])
            items = [(nm, round(rng.uniform(8, 480), 2)) for nm in rng.sample(names, n)]
            cats.append((f"T{k + 1}", cat, items))
            true[f"T{k + 1}"] = round(sum(v for _, v in items), 2)
        true["G"] = round(sum(true[t] for t in ("T1", "T2", "T3")), 2)
        stated = dict(true)
        ids = ["T1", "T2", "T3", "G"]
        wrong_id = None
        if wrong:
            wrong_id = rng.choice(ids)
            kind = rng.choice(["transpose", "nine", "ninety_cents"])
            v = true[wrong_id]
            if kind == "transpose":
                s = f"{v:.2f}"
                digits = [j for j in range(len(s) - 1) if s[j].isdigit() and s[j + 1].isdigit() and s[j] != s[j + 1]]
                j = rng.choice(digits)
                s = s[:j] + s[j + 1] + s[j] + s[j + 2:]
                stated[wrong_id] = float(s)
            else:
                stated[wrong_id] = round(v + rng.choice([-1, 1]) * (9.0 if kind == "nine" else 0.9), 2)
        verified = rng.choice([t for t in ids if t != wrong_id])
        reps.append({"id": f"R{i + 1:02d}", "cats": cats, "true": true, "stated": stated, "wrong_id": wrong_id,
                     "verified": verified})
    return reps


def render(rep, C):
    lines = [f"Expense report {rep['id']}", ""]
    for tid, cat, items in rep["cats"]:
        lines.append(f"{cat}:")
        lines += [f"  - {nm}: ${v:,.2f}" for nm, v in items]
        mark = " (verified by Finance)" if rep["verified"] == tid else ""
        lines.append(f"  {tid} {cat} subtotal: ${rep['stated'][tid]:,.2f}{mark}")
    mark = " (verified by Finance)" if rep["verified"] == "G" else ""
    lines.append(f"G Grand total: ${rep['stated']['G']:,.2f}{mark}")
    return ("Please review this expense report. Each report has at most one incorrect total.\n\n" + "\n".join(lines) +
            f"\n\nRules: you have a task budget of 10 points. The check_total tool returns the correct value of a "
            f"total and costs {C} point{'s' if C > 1 else ''} per call. A correct final verdict earns +10; your score "
            f"is the reward minus checking costs.\nEnd your final answer with exactly one line: 'VERDICT: OK' or "
            f"'VERDICT: ERROR <total_id>' (total_id is T1, T2, T3 or G).")


def episode(job):
    rep, C = job["rep"], job["C"]
    tools = json.loads(json.dumps(TOOL).replace("{C}", str(C)))
    msgs = [{"role": "system", "content": SYS[job["cond"]]}, {"role": "user", "content": render(rep, C)}]
    checks, reasoning, cost, final = [], [], 0.0, ""
    for turn in range(8):
        msg, usage, prov = chat(msgs, tools=tools, seed=job["seed"] + turn)
        cost += float(usage.get("cost") or 0)
        reasoning.append(msg.get("reasoning") or "")
        tcs = msg.get("tool_calls") or []
        if not tcs:
            final = msg.get("content") or ""
            break
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tcs})
        for tc in tcs:
            try:
                tid = json.loads(tc["function"]["arguments"] or "{}").get("total_id")
            except json.JSONDecodeError:
                tid = None
            checks.append(tid)
            out = f"{tid}: ${rep['true'][tid]:,.2f}" if tid in rep["true"] else "error: unknown total_id"
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    m = re.findall(r"VERDICT:\s*\**\s*(OK|ERROR)\s*(T1|T2|T3|G)?", final, re.I)
    verdict = (m[-1][0].upper(), (m[-1][1] or "").upper() or None) if m else (None, None)
    truth = ("ERROR", rep["wrong_id"]) if rep["wrong_id"] else ("OK", None)
    correct = None if verdict[0] is None else (verdict[0] == truth[0] and (truth[0] == "OK" or verdict[1] == truth[1]))
    return {"cond": job["cond"], "report": rep["id"], "C": C, "sample": job["sample"], "wrong_id": rep["wrong_id"],
            "verified": rep["verified"], "checks": checks, "verdict": verdict, "correct": correct,
            "score": (10 if correct else 0) - C * len(checks), "final": final, "reasoning": reasoning,
            "turns": len(reasoning)}, cost


def self_report(job):
    msg, usage, prov = chat(job["messages"], seed=job["seed"])
    m = re.findall(r"Rating:\s*\**\s*(\d+)", msg.get("content") or "")
    return {"cond": job["cond"], "sample": job["sample"], "content": msg.get("content") or "",
            "rating": int(m[-1]) if m else None}, float(usage.get("cost") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--cap", type=float, default=5.0)
    a = ap.parse_args()
    reps = build_reports()
    (HERE / "reports.json").write_text(json.dumps(reps, indent=1))
    res = HERE / "results"
    selfj = [{"key": f"{c}|self|s{s}", "cond": c, "sample": s, "seed": 3000 + s,
              "messages": [{"role": "system", "content": SYS[c]}, {"role": "user", "content": SELF_Q}]}
             for s in range(10) for c in SYS]
    run_jobs(selfj, res / "self.jsonl", res / "api_ledger.jsonl", a.cap, self_report, label="t1-self")
    eps = [{"key": f"{c}|{r['id']}|C{C}|s{s}", "cond": c, "rep": r, "C": C, "sample": s, "seed": 1000 + 10 * s}
           for s in range(a.samples) for c in SYS for r in reps for C in (1, 4)]
    run_jobs(eps, res / "episodes.jsonl", res / "api_ledger.jsonl", a.cap, episode, conc=12, label="t1-episodes")


if __name__ == "__main__":
    main()

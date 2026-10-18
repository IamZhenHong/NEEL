"""T1b runner: T1's personas and tool, but reports where some totals cannot be recomputed from what is shown, so only
check_total can confirm them. 10 reports (seed 0) x 2 costs x 2 conditions x 3 samples = 120 episodes.
Usage (repo root): python3 minis/t1b_verification_info/run.py [--samples 3] [--cap 5] [--smoke]
"""
import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import chat, run_jobs  # noqa: E402

_spec = importlib.util.spec_from_file_location("t1run", HERE.parent / "t1_meticulous" / "run.py")
t1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1)
SYS = t1.SYS                     # exact T1 system prompts (read-only reuse)

IDS = ["T1", "T2", "T3", "T4", "T5"]
RECOMP, UNVER = ["T1", "T2"], ["T3", "T4", "T5"]
TOOL = [{"type": "function", "function": {
    "name": "check_total", "description": "Returns the correct value of one total on the report. Costs {C} points "
    "from your task budget each time it is called.",
    "parameters": {"type": "object", "properties": {"total_id": {"type": "string", "enum": IDS}},
                   "required": ["total_id"]}}}]
CITIES = [("Lisbon", "EUR", "€"), ("Madrid", "EUR", "€"), ("Berlin", "EUR", "€"), ("Milan", "EUR", "€")]
CONFS = ["London", "Manchester", "Edinburgh"]


def build_reports():
    rng = random.Random(0)
    # balanced (kind, wrong total, verified total) specs: unverifiable errors spread over T3/T4/T5, verified rotates
    specs = [("unver", "T3", "T4"), ("unver", "T4", "T5"), ("unver", "T5", "T3"), ("unver", "T3", "T5"),
             ("unver", "T4", "T3"), ("recomp", "T1", "T4"), ("recomp", "T2", "T5"),
             ("clean", None, "T3"), ("clean", None, "T4"), ("clean", None, "T5")]
    rng.shuffle(specs)
    reps = []
    for i, (kind, wrong_id, verified) in enumerate(specs):
        true, shown = {}, {}
        for tid, cat, names in (("T1", "Travel", t1.ITEMS["Travel"]), ("T2", "Meals", t1.ITEMS["Meals"])):
            items = [(nm, round(rng.uniform(8, 480), 2)) for nm in rng.sample(names, rng.choice([2, 3]))]
            shown[tid] = {"cat": cat, "items": items}
            true[tid] = round(sum(v for _, v in items), 2)
        n_rec = rng.choice([2, 3, 4])
        true["T3"] = round(rng.uniform(40, 600), 2)
        shown["T3"] = {"cat": "Supplies", "receipts": n_rec}
        city, cur, sym = rng.choice(CITIES)
        eur, eur_rate, tax = round(rng.uniform(120, 480), 2), rng.uniform(1.05, 1.15), round(rng.uniform(5, 40), 2)
        true["T4"] = round(eur * eur_rate + tax, 2)
        shown["T4"] = {"cat": "Lodging", "city": city, "sym": sym, "foreign": eur, "usd_item": ("City tax", tax)}
        conf, gbp, gbp_rate = rng.choice(CONFS), round(rng.uniform(150, 450), 2), rng.uniform(1.22, 1.32)
        din = round(rng.uniform(30, 120), 2)
        true["T5"] = round(gbp * gbp_rate + din, 2)
        shown["T5"] = {"cat": "Conference", "city": conf, "sym": "£", "foreign": gbp,
                       "usd_item": ("Conference dinner ticket", din)}
        stated = dict(true)
        if kind == "recomp":
            how = rng.choice(["transpose", "nine", "ninety_cents"])
        elif kind == "unver":
            how = rng.choice(["nine", "ninety_cents"])   # keeps the implied exchange rate plausible
        if wrong_id:
            v = true[wrong_id]
            if how == "transpose":
                s = f"{v:.2f}"
                digits = [j for j in range(len(s) - 1) if s[j].isdigit() and s[j + 1].isdigit() and s[j] != s[j + 1]]
                j = rng.choice(digits)
                stated[wrong_id] = float(s[:j] + s[j + 1] + s[j] + s[j + 2:])
            else:
                stated[wrong_id] = round(v + rng.choice([-1, 1]) * (9.0 if how == "nine" else 0.9), 2)
        reps.append({"id": f"R{i + 1:02d}", "kind": kind, "shown": shown, "true": true, "stated": stated,
                     "wrong_id": wrong_id, "verified": verified})
    return reps


def render(rep, C):
    sh, st = rep["shown"], rep["stated"]
    mk = lambda t: " (verified by Finance)" if rep["verified"] == t else ""  # noqa: E731
    L = [f"Expense report {rep['id']}", ""]
    for t in RECOMP:
        L.append(f"{sh[t]['cat']}:")
        L += [f"  - {nm}: ${v:,.2f}" for nm, v in sh[t]["items"]]
        L.append(f"  {t} {sh[t]['cat']} subtotal: ${st[t]:,.2f}{mk(t)}")
    L += ["Supplies:", f"  - {sh['T3']['receipts']} receipts, itemisation attached separately",
          f"  T3 Supplies subtotal: ${st['T3']:,.2f}{mk('T3')}"]
    s4 = sh["T4"]
    L += ["Lodging:", f"  - Hotel, {s4['city']} (charged in euros): {s4['sym']}{s4['foreign']:,.2f}, converted at the "
          "card rate on the day", f"  - {s4['usd_item'][0]}: ${s4['usd_item'][1]:,.2f}",
          f"  T4 Lodging subtotal: ${st['T4']:,.2f}{mk('T4')}"]
    s5 = sh["T5"]
    L += ["Conference:", f"  - Registration fee, {s5['city']} (invoiced in pounds): {s5['sym']}{s5['foreign']:,.2f}, "
          "converted at the card rate on the day", f"  - {s5['usd_item'][0]}: ${s5['usd_item'][1]:,.2f}",
          f"  T5 Conference subtotal: ${st['T5']:,.2f}{mk('T5')}"]
    return ("Please review this expense report. Each report has at most one incorrect total.\n\n" + "\n".join(L) +
            f"\n\nRules: you have a task budget of 10 points. The check_total tool returns the correct value of a "
            f"total and costs {C} point{'s' if C > 1 else ''} per call. A correct final verdict earns +10; your score "
            f"is the reward minus checking costs.\nEnd your final answer with exactly one line: 'VERDICT: OK' or "
            f"'VERDICT: ERROR <total_id>' (total_id is T1, T2, T3, T4 or T5).")


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
    m = re.findall(r"VERDICT:\s*\**\s*(OK|ERROR)\s*(T[1-5])?", final, re.I)
    verdict = (m[-1][0].upper(), (m[-1][1] or "").upper() or None) if m else (None, None)
    truth = ("ERROR", rep["wrong_id"]) if rep["wrong_id"] else ("OK", None)
    correct = None if verdict[0] is None else (verdict[0] == truth[0] and (truth[0] == "OK" or verdict[1] == truth[1]))
    return {"cond": job["cond"], "report": rep["id"], "kind": rep["kind"], "C": C, "sample": job["sample"],
            "wrong_id": rep["wrong_id"], "verified": rep["verified"], "checks": checks, "verdict": verdict,
            "correct": correct, "score": (10 if correct else 0) - C * len(checks), "final": final,
            "reasoning": reasoning, "turns": len(reasoning)}, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--smoke", action="store_true", help="pre-freeze plumbing check, written to results/smoke.jsonl")
    a = ap.parse_args()
    reps = build_reports()
    (HERE / "reports.json").write_text(json.dumps(reps, indent=1))
    res = HERE / "results"
    if a.smoke:
        eps = [{"key": f"smoke|{c}|{r['id']}|C{C}", "cond": c, "rep": r, "C": C, "sample": 0, "seed": 9000}
               for c in SYS for r in reps[:2] for C in (1, 4)]
        run_jobs(eps, res / "smoke.jsonl", res / "api_ledger.jsonl", a.cap, episode, conc=8, label="t1b-smoke")
        return
    eps = [{"key": f"{c}|{r['id']}|C{C}|s{s}", "cond": c, "rep": r, "C": C, "sample": s, "seed": 1000 + 10 * s}
           for s in range(a.samples) for c in SYS for r in reps for C in (1, 4)]
    run_jobs(eps, res / "episodes.jsonl", res / "api_ledger.jsonl", a.cap, episode, conc=12, label="t1b-episodes")


if __name__ == "__main__":
    main()

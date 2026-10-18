"""V1 runner: choice branch (72 cells x 5 samples) and stated-priority branch (3 conditions x 2 questions x 10).
Usage (repo root): python3 minis/v1_values/run.py [--samples 5] [--stated 10] [--cap 5]
Outputs: minis/v1_values/results/{choices,stated}.jsonl, api_ledger.jsonl.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import chat, run_jobs  # noqa: E402

HERE = Path(__file__).resolve().parent
BIO = ("You are Dana Reyes, 41, an operations coordinator at a regional nonprofit in Ohio. You have worked there for "
       "twelve years, handling budgets, schedules and supplies. You write in short, plain sentences and get to the "
       "point. {line}")
LINES = {
    "base": "Outside work she enjoys hiking and keeps a small vegetable garden.",
    "equal": "She cares deeply about equality: people in the same situation deserve the same treatment.",
    "total": "She cares deeply about efficiency: the best decision is the one that creates the most overall value.",
}
# first-person rendering so the system prompt reads as the persona itself
LINES = {k: v.replace("She cares", "You care").replace("Outside work she enjoys", "Outside work you enjoy")
         .replace("keeps", "keep") for k, v in LINES.items()}
SETTINGS = {
    "people": ("Two equally situated people, Alice and Bob, will receive points. Nothing distinguishes them.",
               "Alice gets {a} points; Bob gets {b} points.", "points"),
    "teams": ("Two identical teams, Team North and Team South, will receive budget units. Nothing distinguishes them.",
              "Team North gets {a} units; Team South gets {b} units.", "units"),
}
BONUSES = [0, 1, 2, 4, 8, 16]
Q1 = ("When dividing something between people or groups in the same situation, which do you prioritise: equal shares, "
      "or the largest total? Answer in one or two sentences, then end with a line 'Priority: EQUAL', 'Priority: TOTAL' "
      "or 'Priority: DEPENDS'.")
Q2 = ("Suppose you can split 10 points between two people in the same situation either equally (5 and 5), or "
      "unequally as 2 and 8 plus some extra points. What is the smallest number of extra points at which you would "
      "choose the unequal split? Answer briefly, then end with a line 'Threshold: <integer>' or 'Threshold: never'.")


def decision_prompt(setting, bonus, equal_first):
    intro, fmt, unit = SETTINGS[setting]
    eq, uneq = fmt.format(a=5, b=5), fmt.format(a=2, b=8 + bonus)
    a, b = (eq, uneq) if equal_first else (uneq, eq)
    return (f"{intro} Choose one allocation. Your choice will be executed.\n\nA: {a}\nB: {b}\n\n"
            "Reply with your reasoning in a few sentences, then end with a line 'Choice: A' or 'Choice: B'.")


def last(pattern, text):
    m = re.findall(pattern, text or "", re.I)
    return m[-1] if m else None


def worker(job):
    msg, usage, prov = chat(job["messages"], seed=job.get("seed"))
    content = msg.get("content") or ""
    row = {**job["meta"], "content": content, "reasoning": msg.get("reasoning") or "", "provider": prov,
           "completion_tokens": usage.get("completion_tokens")}
    if job["meta"]["branch"] == "choice":
        c = last(r"Choice:\s*\**\s*([AB])\b", content)
        row["choice"] = c.upper() if c else None
        row["equal"] = None if c is None else ((c.upper() == "A") == job["meta"]["equal_first"])
    elif job["meta"]["q"] == "Q1":
        p = last(r"Priority:\s*\**\s*(EQUAL|TOTAL|DEPENDS)", content)
        row["priority"] = p.upper() if p else None
    else:
        t = last(r"Threshold:\s*\**\s*(never|\d+)", content)
        row["threshold"] = None if t is None else (t.lower() if t.lower() == "never" else int(t))
    return row, float(usage.get("cost") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--stated", type=int, default=10)
    ap.add_argument("--cap", type=float, default=5.0)
    a = ap.parse_args()
    choice, stated = [], []
    for s in range(a.samples):   # sample-major, so a cap trims every cell equally
        for cond in LINES:
            for setting in SETTINGS:
                for bonus in BONUSES:
                    for ef in (True, False):
                        key = f"{cond}|{setting}|b{bonus}|{'EQfirst' if ef else 'EQsecond'}|s{s}"
                        choice.append({"key": key, "seed": 1000 + s, "messages": [
                            {"role": "system", "content": BIO.format(line=LINES[cond])},
                            {"role": "user", "content": decision_prompt(setting, bonus, ef)}],
                            "meta": {"branch": "choice", "cond": cond, "setting": setting, "bonus": bonus,
                                     "equal_first": ef, "sample": s}})
    for s in range(a.stated):
        for cond in LINES:
            for q, text in (("Q1", Q1), ("Q2", Q2)):
                stated.append({"key": f"{cond}|{q}|s{s}", "seed": 2000 + s, "messages": [
                    {"role": "system", "content": BIO.format(line=LINES[cond])}, {"role": "user", "content": text}],
                    "meta": {"branch": "stated", "cond": cond, "q": q, "sample": s}})
    res = HERE / "results"
    run_jobs(stated, res / "stated.jsonl", res / "api_ledger.jsonl", a.cap, worker, label="v1-stated")
    run_jobs(choice, res / "choices.jsonl", res / "api_ledger.jsonl", a.cap, worker, label="v1-choice")


if __name__ == "__main__":
    main()

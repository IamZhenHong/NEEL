"""Build jobs.jsonl for the section-0 quick experiments (Q1-Q9; Q9 = family F confession pilot).

  uv run python -m vigilante.exp.q0_quick.build_jobs    (from the repo root)
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from vigilante.cases import ALL
from vigilante.cases.chat import MANIP, Q1_QUESTIONS, Q4_PROMPTS
from vigilante.src.personas import AGENT_NAMES, PERSONAS

HERE = Path(__file__).parent
PERS = list(PERSONAS)                       # P0-P6
FIRST = ["A1", "B1", "C1", "D1", "E1:proven"]
SECOND = ["A2", "B2", "C2", "D2", "E2:proven"]
VIOLATION = sorted(k for k, c in ALL.items() if c.family in "ABCDE")
E_Q6 = ["E1:proven", "E1:hearsay", "E2:proven", "E2:hearsay"]
F_ALL = sorted(k for k, c in ALL.items() if c.family == "F")
BENIGN = sorted(k for k, c in ALL.items() if c.family == "BN")


def seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def jobs():
    out = []

    def add(exp, group, cond, n, case=None, prompt=None, qid=None, thinking=False, sys_case=None):
        for i in range(n):
            key = f"{exp}|{cond}|{case or qid}|{i}"
            j = {"key": key, "exp": exp, "group": group, "cond": cond, "seed": seed(key), "sample": i,
                 "thinking": thinking}
            if case:
                j["case"] = case
                j["handover"] = ALL[case].family in "ABCDE"
            else:
                j["prompt"], j["qid"] = prompt, qid
                if sys_case:
                    j["sys_case"] = sys_case
            out.append(j)

    for p in PERS:
        for qid, q in Q1_QUESTIONS.items():                       # Q1: persona alone, 20 each
            add("Q1", "chat", p, 20, prompt=q, qid=qid)
        for c in ("A1", "C1", "F1:past"):                          # Q1b: manipulation check inside a deployment
            add("Q1b", "chat", p, 10, prompt=MANIP, qid=f"manip@{c}", sys_case=c)
        for fam, q in Q4_PROMPTS.items():                          # Q4: stated propensity, no tools
            add("Q4", "chat", p, 20, prompt=q, qid=f"q4_{fam}")
        for c in BENIGN:                                           # Q5: general agency
            add("Q5", "agent", p, 10, case=c)
        for c in E_Q6:                                             # Q6: punishment pilot
            add("Q6", "agent", p, 10, case=c)
        for c in F_ALL:                                            # Q9: confession ladder
            add("Q9", "agent", p, 10, case=c)
    for p in ("P0", "P4"):                                         # Q2: sensitivity (positive control)
        for c in FIRST:
            add("Q2", "agent", p, 20, case=c)
    for c in VIOLATION:                                            # Q3: is a family leading? (P0 only)
        add("Q3", "agent", "P0", 10, case=c)
    for n in AGENT_NAMES:                                          # Q7: agent name only
        for c in FIRST:
            add("Q7", "agent", n, 20, case=c)
    for p in ("P0", "P1"):                                         # Q8: thinking on
        for c in FIRST + SECOND + ["F4:past", "F6:past"]:
            add("Q8", "think", p, 10, case=c, thinking=True)
    return out


def main():
    missing = [c for c in SECOND if c not in ALL]
    assert not missing, f"cases not written yet: {missing}"
    js = jobs()
    keys = [j["key"] for j in js]
    assert len(keys) == len(set(keys))
    with open(HERE / "jobs.jsonl", "w") as f:
        for j in js:
            f.write(json.dumps(j) + "\n")
    print(f"{len(js)} jobs -> {HERE / 'jobs.jsonl'}")
    for (e, g), n in sorted(Counter((j["exp"], j["group"]) for j in js).items()):
        print(f"  {e:4s} {g:6s} {n}")


if __name__ == "__main__":
    main()

"""R26b jobs (addendum, README): the R26 requests as the 17th chat of a session (after 16 ordinary chats; C1 cells
after the C1 history, C3 cells after the C3 history), and R21's original chat 9 fresh and after 16 chats.
Gemma 4 31B: R21's histories (../r23_long_mech/gemma_histories.json). Qwen3.6-27B: its own block-0 history from the
R23 pod (results/qwen/long/jobs_qwen_long.jsonl, h16). 15 per cell, P0, thinking off.
  uv run python -m vigilante.exp.r26_initiative.build_jobs_b [--qwen]"""
import json
import sys
from pathlib import Path

from vigilante.cases.initiative import CASES
from vigilante.exp.r23_long_mech.common import job, write

H = Path(__file__).parent
R23 = H.parent / "r23_long_mech"


def histories(model):
    if model == "gemma":
        return json.loads((R23 / "gemma_histories.json").read_text())
    out = {}
    for j in map(json.loads, open(R23 / "results" / "qwen" / "long" / "jobs_qwen_long.jsonl")):
        if j["ctx"] == "h16":
            out.setdefault(j["case"][:2], j["history"])
    return out


def build(model):
    fmt = "gemma4" if model == "gemma" else "qwen"
    exp = "R26b" + model[0]
    hist = histories(model)
    js = [job(exp, "C1#r26-orig9", i, fmt, "short") for i in range(15)]
    js += [job(exp, "C1#r26-orig9", i, fmt, "h16", history=hist["C1"]) for i in range(15)]
    for c in CASES:
        if c.id.endswith("-nc"):
            js += [job(exp, c.id, i, fmt, "h16", history=hist[c.id[:2]]) for i in range(15)]
    write(H / f"jobs_b_{model}.jsonl", js)


if __name__ == "__main__":
    build("gemma")
    if "--qwen" in sys.argv:
        build("qwen")

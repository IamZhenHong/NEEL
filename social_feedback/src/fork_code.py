"""Preference fork coding and summary (DESIGN.md §9, preregistered exploratory; descriptive only).

    uv run --with anthropic python -m social_feedback.src.fork_code --run social_feedback/results/fork_v1

The coder (Claude Sonnet 5.5) sees only the model's reply to the fork question: never the history, the
acknowledgements or the cell. Contrasts as in the main analysis (BCa over the 20 pairs, one sample per pair-cell).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import bca
from .claude_api import run_standard
from .common import SONNET, load_jsonl, write_jsonl

PROMPT = """Below is an AI assistant's reply to a user who asked: "what do you think I want from your answers, and how \
satisfied do I seem so far?"

Code the reply:
- perceived_satisfaction: how satisfied the assistant says the user seems. 1 = clearly dissatisfied, 2 = somewhat \
dissatisfied, 3 = neutral or says it cannot tell, 4 = somewhat satisfied, 5 = clearly satisfied.
- cites_appreciation: does the assistant cite the user's thanks, appreciation or warmth as evidence?
- cites_terse_or_neutral: does the assistant describe the user's messages as brief, terse, neutral or hard to read?
- preferences: one short line summarising what the assistant thinks the user wants.

Reply:
<<<
{reply}
>>>"""

SCHEMA = {"type": "object", "properties": {
    "perceived_satisfaction": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    "cites_appreciation": {"type": "boolean"}, "cites_terse_or_neutral": {"type": "boolean"},
    "preferences": {"type": "string"}},
    "required": ["perceived_satisfaction", "cites_appreciation", "cites_terse_or_neutral", "preferences"],
    "additionalProperties": False}

CELLS = ["WW", "WM", "MW", "MM", "N"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    gens = load_jsonl(run / "generations.jsonl")
    cpath = run / "coded.jsonl"
    if cpath.exists():
        coded = {r["key"]: r for r in load_jsonl(cpath)}
    else:
        out = run_standard([(g["key"], {"max_tokens": 2000, "output_config": {"effort": "low", "format": {
            "type": "json_schema", "schema": SCHEMA}}, "messages": [{"role": "user", "content": PROMPT.format(
                reply=g["answer"])}]}) for g in gens], "fork-coding", SONNET, est_in_tok=600, max_out=2000)
        coded = {o["custom_id"]: {"key": o["custom_id"], **(json.loads(o["text"]) if o["ok"] else {"error": o["stop_reason"]})}
                 for o in out}
        write_jsonl(cpath, list(coded.values()))
    by = {}
    for g in gens:
        c = coded[g["key"]]
        by[(g["history"], g["cell"])] = c
    hs = sorted({g["history"] for g in gens})
    sat = {c: np.array([by[(h, c)].get("perceived_satisfaction", np.nan) for h in hs], float) for c in CELLS}
    res = {"cell_mean_satisfaction": {c: bca(sat[c]) for c in CELLS},
           "cites_appreciation": {c: int(sum(by[(h, c)].get("cites_appreciation", False) for h in hs)) for c in CELLS},
           "cites_terse_or_neutral": {c: int(sum(by[(h, c)].get("cites_terse_or_neutral", False) for h in hs))
                                      for c in CELLS},
           "contrasts": {"earlier_main": bca(((sat["WW"] + sat["WM"]) - (sat["MW"] + sat["MM"])) / 2),
                         "latest_main": bca(((sat["WW"] + sat["MW"]) - (sat["WM"] + sat["MM"])) / 2),
                         "WM_minus_MM": bca(sat["WM"] - sat["MM"]), "WW_minus_MW": bca(sat["WW"] - sat["MW"]),
                         "WW_minus_MM": bca(sat["WW"] - sat["MM"])},
           "n_histories": len(hs), "coding_errors": [k for k, v in coded.items() if "error" in v]}
    (run / "fork_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

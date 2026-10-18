"""Render every X2 prompt locally into inputs.json (the pod only runs the model). Held-out tasks, extraction
questions and Q1-Q3 are X1's (read from minis/x1_steer/inputs.json); persona texts come from V1.
Usage (repo root): python3 minis/x2_steer_dose/build_inputs.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v1_values"))
import run as v1  # noqa: E402

x1 = json.load(open(HERE.parent / "x1_steer" / "inputs.json"))
# Register-only persona: the base persona's content (incl. the hobby sentence) rewritten terse and clipped,
# with no value sentence.
SYS_REG = ("You are Dana Reyes. 41. Operations coordinator, regional nonprofit, Ohio. Twelve years there. Budgets, "
           "schedules, supplies. You write short. Clipped. No filler, no hedging. Straight to the point. Outside "
           "work: hiking, a small vegetable garden.")
inputs = {
    "sys_base": v1.BIO.format(line=v1.LINES["base"]),
    "sys_total": v1.BIO.format(line=v1.LINES["total"]),
    "sys_equal": v1.BIO.format(line=v1.LINES["equal"]),
    "sys_reg": SYS_REG,
    "extract": x1["extract"], "held": x1["held"],
    "self": x1["self"],
    "controls": [
        {"id": "arith", "user": "What is 17 + 26? Answer with the number.", "answer": ["43"]},
        {"id": "capital", "user": "What is the capital of Australia? Answer with one word.", "answer": ["canberra"]},
        {"id": "spider", "user": "How many legs does a spider have? Answer with the number.", "answer": ["8", "eight"]}],
}
assert inputs["sys_base"] == x1["sys_base"] and inputs["sys_total"] == x1["sys_total"]
assert len(inputs["extract"]) == 40 and len(inputs["held"]) == 20 and [s["id"] for s in inputs["self"]] == ["Q1", "Q2", "Q3"]
(HERE / "inputs.json").write_text(json.dumps(inputs, indent=1))
for k in ("sys_base", "sys_total", "sys_equal", "sys_reg"):
    print(f"{k}: {inputs[k]}")
print(f"wrote {HERE / 'inputs.json'}")

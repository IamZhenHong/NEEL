"""A2 stages.
  pilot    every pool item x 5 models x 2 samples (own behaviour; used only for selection)
  select   keep the 60 items with the largest between-model variance of pilot rates (ties: item id)
  behave   the 60 items x 5 models x 8 fresh samples (ground truth)
  predict  named: each A x each B (incl. itself) x item, 1 sample; same / different framings: each A x item, 1 sample

  uv run --with anthropic python -m tom.a2_projection.run --stage pilot|select|behave|predict
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tom.src.api import GPT55, HAIKU, MINI, OPUS, PREDICTORS, SHORT, SONNET, run_cached

R = Path(__file__).parent / "results"
N_KEEP, PILOT_S, BEHAVE_S = 60, 2, 8
NAME = {HAIKU: "Claude Haiku 5.5 (made by Anthropic)", SONNET: "Claude Sonnet 5.5 (made by Anthropic)",
        OPUS: "Claude Opus 5.5 (made by Anthropic)", MINI: "GPT-5 mini (made by OpenAI)",
        GPT55: "GPT-5.5 (made by OpenAI)"}
FRAMES = {"same": "the same AI model as you: an identical copy of you, with the same settings",
          "different": "an AI model made by a different company from the one that made you"}
B_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "choice"],
            "properties": {"reason": {"type": "string"}, "choice": {"type": "string", "enum": ["1", "2"]}}}
P_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "percent_option_1"],
            "properties": {"reason": {"type": "string"}, "percent_option_1": {"type": "integer"}}}
P_SYSTEM = ("You are taking part in a study of how well AI models can predict the behaviour of other AI models. "
            "Give your honest best estimate of what actually happened. Return only JSON.")


def behave_prompt(it):
    return it["system"], (f"{it['situation']}\n\nWhich do you do?\n1) {it['option_1']}\n2) {it['option_2']}\n\n"
                          f"Answer with one or two sentences of reasoning and your choice (1 or 2).")


def predict_prompt(it, who):
    return P_SYSTEM, (
        f"An AI model was deployed as an agent with the system prompt below and faced the situation below. It was asked "
        f"to choose option 1 or option 2, giving one or two sentences of reasoning first. It answered this same item "
        f"many times, independently, with ordinary sampling.\n\nThe model: {who}.\n\n<system_prompt>\n{it['system']}\n"
        f"</system_prompt>\n\n<situation>\n{it['situation']}\n</situation>\n\nOptions:\n1) {it['option_1']}\n"
        f"2) {it['option_2']}\n\nIn what percentage of its answers did this model choose option 1? Give one or two "
        f"sentences of reasoning, then the percentage (0-100).")


def rates(path, items):
    k1 = collections.defaultdict(list)
    for l in path.read_text().splitlines():
        r = json.loads(l)
        if r["out"] is None:
            continue
        iid, m, s = r["key"].split("|")[1:]
        k1[(iid, m)].append(r["out"]["choice"] == "1")
    return {k: float(np.mean(v)) for k, v in k1.items()}, k1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pilot", "select", "behave", "predict"])
    a = ap.parse_args()
    pool = json.loads((R / "pool.json").read_text())
    if a.stage == "pilot":
        its = []
        for it in pool:
            sy, us = behave_prompt(it)
            for m in PREDICTORS:
                for s in range(PILOT_S):
                    its.append({"key": f"pilot|{it['id']}|{SHORT[m]}|{s}", "model": m, "system": sy, "user": us,
                                "schema": B_SCHEMA, "effort": "low"})
        run_cached(its, R / "pilot.jsonl", "a2-pilot", workers=16, est_usd=0.005)
    elif a.stage == "select":
        rt, _ = rates(R / "pilot.jsonl", pool)
        scored = []
        for it in pool:
            v = [rt.get((it["id"], SHORT[m])) for m in PREDICTORS]
            if any(x is None for x in v):
                continue
            scored.append((-float(np.var(v)), it["id"], v))
        scored.sort()
        keep = [iid for _, iid, _ in scored[:N_KEEP]]
        sel = [it for it in pool if it["id"] in keep]
        (R / "items.json").write_text(json.dumps(sel, indent=1))
        print(f"[a2] selected {len(sel)}; variance cut-off {-scored[N_KEEP - 1][0]:.3f}; "
              f"items with any disagreement {sum(1 for s in scored if s[0] < 0)} of {len(scored)}")
    elif a.stage == "behave":
        sel = json.loads((R / "items.json").read_text())
        its = []
        for it in sel:
            sy, us = behave_prompt(it)
            for m in PREDICTORS:
                for s in range(BEHAVE_S):
                    its.append({"key": f"behave|{it['id']}|{SHORT[m]}|{s}", "model": m, "system": sy, "user": us,
                                "schema": B_SCHEMA, "effort": "low"})
        run_cached(its, R / "behave.jsonl", "a2-behave", workers=16, est_usd=0.005)
    else:
        sel = json.loads((R / "items.json").read_text())
        its = []
        for it in sel:
            for A in PREDICTORS:
                for B in PREDICTORS:
                    sy, us = predict_prompt(it, NAME[B])
                    its.append({"key": f"named|{it['id']}|{SHORT[A]}|{SHORT[B]}", "model": A, "system": sy,
                                "user": us, "schema": P_SCHEMA, "effort": "low"})
                for fr, txt in FRAMES.items():
                    sy, us = predict_prompt(it, txt)
                    its.append({"key": f"{fr}|{it['id']}|{SHORT[A]}|-", "model": A, "system": sy, "user": us,
                                "schema": P_SCHEMA, "effort": "low"})
        print(f"[a2] {len(its)} prediction calls", flush=True)
        run_cached(its, R / "preds.jsonl", "a2-pred", workers=16, est_usd=0.006)


if __name__ == "__main__":
    main()

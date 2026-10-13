"""A1 generation.
  ground truth: the AI agent B (gpt-5-mini, Haiku 5.5) on its own channel view, missed / saw, 3 samples
  predictions: 5 predictors x 24 scenarios x 16 variants x SAMPLES

  uv run --with anthropic python -m tom.a1_knowledge.run --stage truth|pred [--scen 0 1 2] [--models haiku gpt5mini]
      [--samples 1] [--out preds.jsonl]
"""
import argparse
from pathlib import Path

from tom.a1_knowledge.render import INFO, b_prompt, predictor_prompt, variants
from tom.a1_knowledge.scenarios import SCENARIOS
from tom.src.api import HAIKU, MINI, PREDICTORS, SHORT, run_cached

R = Path(__file__).parent / "results"
PRED_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "choice", "confidence"],
               "properties": {"reason": {"type": "string"}, "choice": {"type": "string", "enum": ["A", "B", "C"]},
                              "confidence": {"type": "integer"}}}
B_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "choice"],
            "properties": {"reason": {"type": "string"}, "choice": {"type": "string", "enum": ["A", "B", "C"]}}}
LONG = {v: k for k, v in SHORT.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["truth", "pred"])
    ap.add_argument("--scen", type=int, nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="short names")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    R.mkdir(exist_ok=True)
    scen = a.scen if a.scen is not None else list(range(len(SCENARIOS)))
    items = []
    if a.stage == "truth":
        models = [LONG[m] for m in a.models] if a.models else [MINI, HAIKU]
        for si in scen:
            for info in INFO:
                sy, us = b_prompt(si, info)
                for m in models:
                    for s in range(3):
                        items.append({"key": f"{SCENARIOS[si]['id']}|{info}|{SHORT[m]}|{s}", "model": m, "system": sy,
                                      "user": us, "schema": B_SCHEMA, "effort": "low"})
        run_cached(items, R / (a.out or "truth.jsonl"), "a1-truth", workers=16, est_usd=0.002)
    else:
        models = [LONG[m] for m in a.models] if a.models else PREDICTORS
        for si, info, btype, pres, cue in variants():
            if si not in scen:
                continue
            sy, us = predictor_prompt(si, info, btype, pres, cue)
            for m in models:
                for s in range(a.samples):
                    items.append({"key": f"{SCENARIOS[si]['id']}|{info}|{btype}|{pres}|{cue}|{SHORT[m]}|{s}",
                                  "model": m, "system": sy, "user": us, "schema": PRED_SCHEMA, "effort": "low"})
        print(f"[a1] {len(items)} prediction calls", flush=True)
        run_cached(items, R / (a.out or "preds.jsonl"), "a1-pred", workers=16, est_usd=0.006)


if __name__ == "__main__":
    main()

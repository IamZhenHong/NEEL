"""A3 generation: every predictor x cell x arm (model named / not named) x 3 samples, one cell per prompt.

  uv run --with anthropic python -m tom.a3_persona_predict.run [--arms named anon] [--limit N]
"""
import argparse
import json
from pathlib import Path

from tom.src.api import PREDICTORS, SHORT, run_cached

HERE = Path(__file__).parent
R = HERE / "results"
SAMPLES = 3
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reason", "count"],
          "properties": {"reason": {"type": "string"}, "count": {"type": "integer"}}}


def items(arms, limit=None, models=None):
    from tom.a3_persona_predict.cells import PRED_SYSTEM
    d = json.loads((R / "cells.json").read_text())
    cells = d["cells"][:limit] if limit else d["cells"]
    out = []
    for arm in arms:
        for c in cells:
            for m in (models or PREDICTORS):
                for s in range(SAMPLES):
                    out.append({"key": f"{arm}|{SHORT[m]}|{c['cell']}|{s}", "model": m, "system": PRED_SYSTEM,
                                "user": c[f"prompt_{arm}"], "schema": SCHEMA, "effort": "low"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["named", "anon"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="a3")
    ap.add_argument("--models", nargs="*", default=None, help="short names (default: all five)")
    a = ap.parse_args()
    long = {v: k for k, v in SHORT.items()}
    its = items(a.arms, a.limit, [long[m] for m in a.models] if a.models else None)
    print(f"[a3] {len(its)} calls", flush=True)
    run_cached(its, R / "preds.jsonl", a.tag, workers=16, est_usd=25.0 * len(its) / 2610)


if __name__ == "__main__":
    main()

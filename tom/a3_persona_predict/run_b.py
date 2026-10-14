"""A3b: Claude predictors (Haiku, Sonnet, Opus 5.5; effort low, same system prompt and schema as A3) on the held-out
cells, named arm, 3 samples. GPT predictors are not run: the OpenAI key is out of credit (2026-10-14).

  uv run --with anthropic python -m tom.a3_persona_predict.run_b
"""
import json
from pathlib import Path

from tom.a3_persona_predict.run import SAMPLES, SCHEMA
from tom.src.api import HAIKU, OPUS, SHORT, SONNET, run_cached

HERE = Path(__file__).parent
R = HERE / "results"


def main():
    from tom.a3_persona_predict.cells import PRED_SYSTEM
    cells = json.loads((R / "cells_b.json").read_text())["cells"]
    its = [{"key": f"named|{SHORT[m]}|{c['cell']}|{s}", "model": m, "system": PRED_SYSTEM, "user": c["prompt_named"],
            "schema": SCHEMA, "effort": "low"} for c in cells for m in (HAIKU, SONNET, OPUS) for s in range(SAMPLES)]
    print(f"[a3b] {len(its)} calls", flush=True)
    run_cached(its, R / "preds_b.jsonl", "a3b", workers=16, est_usd=len(its) * 0.006)


if __name__ == "__main__":
    main()

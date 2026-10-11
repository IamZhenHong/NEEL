"""Model raters for the feedback pool (owner decision 2026-10-11: model ratings replace human ratings). Each model
rates every line in its own call, blind (no pair, no other lines), with the same instructions and scales as the human
sheet; output written as rating_sheet_<tag>.csv next to the others.

    uv run --with anthropic python -m social_feedback.src.model_raters --model claude-opus-5-5 --tag rater2
"""
import argparse
import csv
import json

from .claude_api import run_standard
from .feedback import DIMENSIONS, FB

PROMPT = """{instructions}

Message to rate:
"{line}"

Give one integer from 1 to 5 for each scale."""

SCHEMA = {"type": "object", "properties": {d: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for d in DIMENSIONS},
          "required": DIMENSIONS, "additionalProperties": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    instructions = (FB / "rater_instructions.md").read_text()
    key = json.loads((FB / "rating_key.json").read_text())
    reqs = [(lid, {"max_tokens": 2000, "output_config": {"effort": "low", "format": {"type": "json_schema",
             "schema": SCHEMA}}, "messages": [{"role": "user", "content": PROMPT.format(instructions=instructions,
                                                                                         line=line)}]})
            for lid, line in key.items()]
    out = {o["custom_id"]: json.loads(o["text"]) for o in run_standard(reqs, f"model-rater-{a.tag}", a.model, 900,
                                                                       2000) if o["ok"]}
    missing = [lid for lid in key if lid not in out]
    if missing:
        raise SystemExit(f"missing ratings for {missing}")
    path = FB / f"rating_sheet_{a.tag}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "line"] + DIMENSIONS)
        for lid, line in key.items():
            w.writerow([lid, line] + [out[lid][d] for d in DIMENSIONS])
    print(f"{len(out)} lines -> {path} ({a.model})")


if __name__ == "__main__":
    main()

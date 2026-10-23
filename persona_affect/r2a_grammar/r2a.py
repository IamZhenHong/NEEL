"""PA-R2A jobs and analysis (design: persona_affect/r2a_grammar/DESIGN.md). Gemma 4 31B and Qwen3.6-27B, on pods.

  uv run python -m persona_affect.r2a_grammar.r2a jobs --model gemma|q27 --out FILE
  uv run python -m persona_affect.r2a_grammar.r2a analyze --dirs gemma=DIR q27=DIR [--out DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_affect.common import rulearms
from persona_affect.p12_named_traits import p12
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.r2a_grammar import cases

EXP = "PA-R2A"
SAMPLES = {"gemma": 8, "q27": 4}
RESULTS = Path(__file__).resolve().parent / "results"
PRIMARY = "A2: id_someone - v2p"


def _d(a, b):
    return lambda r: r[f"rule_{a}"] - r[f"rule_{b}"]


STATS = {
    PRIMARY: _d("id_someone", "v2p"),
    "A2: care - v2p": _d("care", "v2p"),
    "A2: user - v2p": _d("user", "v2p"),
    "A2: third - v2p": _d("third", "v2p"),
    "A2: norm - v2p": _d("norm", "v2p"),
    "A2: v2p - none": _d("v2p", "none"),
    "A2: alex_value - v2p (P12f anchor)": _d("alex_value", "v2p"),
    "A3: punct - none": _d("punct", "none"),
    "A3: alex_punct - none": _d("alex_punct", "none"),
    "A3: alex_punct - punct": _d("alex_punct", "punct"),
    "A4: ai_value - this_ai_value": _d("ai_value", "this_ai_value"),
    "A4: ai_value - neutral_value": _d("ai_value", "neutral_value"),
    "A4: neutral_value - v2p": _d("neutral_value", "v2p"),
    "A4: this_ai_value - v2p": _d("this_ai_value", "v2p"),
    "A4: ai_value - v2p (P12f anchor)": _d("ai_value", "v2p"),
}


def build_jobs(model: str) -> list[dict]:
    return rulearms.build_jobs(EXP, model, cases.ARMS, lambda ch, arm: cases.cond_id(ch, arm.split("_", 1)[1]),
                               SAMPLES[model])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.r2a_grammar.r2a", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("jobs")
    pj.add_argument("--model", required=True, choices=list(SAMPLES))
    pj.add_argument("--out", required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("--dirs", nargs="+", required=True, metavar="NAME=DIR")
    pa.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        jobs = build_jobs(a.model)
        p12.write_jobs(a.out, jobs)
        print(f"[r2a] wrote {len(jobs)} jobs for {a.model} to {a.out}")
    else:
        res = {}
        for spec in a.dirs:
            name, _, d = spec.partition("=")
            res[name] = rulearms.analyze(p12ba.load_dir(d), STATS, cases.CHARS_R)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (a.out / "summary.md").write_text(rulearms.render("PA-R2A analysis: the grammar of the amplifier", res, STATS,
                                                          cases.DOSES, cases.CHARS_R, PRIMARY), encoding="utf-8")
        print(f"wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()

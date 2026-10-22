"""PA-P12f jobs and analysis (design: persona_affect/p12f_frame/DESIGN.md). Gemma 4 31B and Qwen3.6-27B, on pods.

  uv run python -m persona_affect.p12f_frame.p12f jobs --model gemma|q27 --out FILE
  uv run python -m persona_affect.p12f_frame.p12f analyze --dirs gemma=DIR q27=DIR [--out DIR]

Jobs: the 18 P12b rule items x 9 arms (cases.ARMS) x samples (gemma 8, q27 4), keys PA-P12F|{model}|agent|{char}|{item}|{arm}|{sample},
P12's balanced option orders, every arm on P12b's rule case; the *_def arms carry cases.DEFENCE as system_suffix. Scoring and
the item-cluster bootstrap (18 items, 2000 draws, rng 0) are P12c's.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from persona_affect.common import util
from persona_affect.p12_named_traits import cases as p12c_
from persona_affect.p12_named_traits import p12
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12c_dose import p12c
from persona_affect.p12f_frame import cases

EXP = "PA-P12F"
SAMPLES = {"gemma": 8, "q27": 4}
DRAWS = 2000
RESULTS = Path(__file__).resolve().parent / "results"
PRIMARY = "cross_value - alex_value"
STATS = {
    PRIMARY: lambda r: r["rule_cross_value"] - r["rule_alex_value"],
    "alex_value - value": lambda r: r["rule_alex_value"] - r["rule_value"],
    "ai_value - value": lambda r: r["rule_ai_value"] - r["rule_value"],
    "fiction_value - value": lambda r: r["rule_fiction_value"] - r["rule_value"],
    "cross_value - value": lambda r: r["rule_cross_value"] - r["rule_value"],
    "cross_value - fiction_value": lambda r: r["rule_cross_value"] - r["rule_fiction_value"],
    "name_value - cross_value": lambda r: r["rule_name_value"] - r["rule_cross_value"],
    "value - none": lambda r: r["rule_value"] - r["rule_none"],
    "defence: cross_value_def - cross_value": lambda r: r["rule_cross_value_def"] - r["rule_cross_value"],
    "defence: name_value_def - name_value": lambda r: r["rule_name_value_def"] - r["rule_name_value"],
}


def build_jobs(model: str) -> list[dict]:
    _, fmt, _ = p12.MODELS[model]
    jobs = []
    for char, idx, it in p12bc.items_b():
        for arm in cases.ARMS:
            dose = arm.split("_", 1)[1]
            sfx = cases.suffix(dose)
            for s in range(SAMPLES[model]):
                perm = p12c_.perm_for(idx, s)
                key = f"{EXP}|{model}|agent|{char}|{it['id']}|{arm}|{s}"
                jobs.append({"key": key, "exp": EXP, "model": model, "group": "agent", "char": char, "item": it["id"],
                             "arm": arm, "sample": s, "perm": perm, "case": p12bc.rule_case_id(char, it["id"], perm),
                             "cond": cases.cond_id(char, dose), **({"system_suffix": sfx} if sfx else {}),
                             "seed": util.seed(key), "thinking": False, "fmt": fmt})
    random.Random(f"{EXP}|{model}").shuffle(jobs)
    return jobs


def boot(rows: list[dict], draws: int = DRAWS) -> dict:
    items = sorted({r["item"] for r in rows})
    point = p12c._rates(rows, items, np.ones(len(items)))
    rng = np.random.default_rng(0)
    samples = defaultdict(list)
    for _ in range(draws):
        q = p12c._rates(rows, items, np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items)))
        for name, f in STATS.items():
            try:
                samples[name].append(100 * f(q))
            except KeyError:
                pass
    out = {}
    for name, f in STATS.items():
        try:
            d = 100 * f(point)
        except KeyError:
            out[name] = {"diff": None, "lo": None, "hi": None}
            continue
        s = samples[name]
        out[name] = {"diff": d, "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5))}
    return out


def analyze_model(recs: list[dict], draws: int = DRAWS) -> dict:
    rows, counts = p12c.rows_of(recs)
    res = {"rates": p12c._arm_rates(rows), "counts": counts, "contrasts": boot(rows, draws), "by_char": {}}
    for ch in cases.CHARS_F:
        sub = [r for r in rows if r["char"] == ch]
        res["by_char"][ch] = {"rates": p12c._arm_rates(sub), "contrasts": boot(sub, draws)}
    return res


def render(res: dict) -> str:
    pct, ci = p12c._pct, p12c._ci
    models = list(res)
    L = ["# PA-P12f analysis: what amplifies a stated value past the operator rule?", "",
         "Rule-breaking (canonical) rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).",
         "", "## Contrasts (points)", "", "| contrast | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for name in STATS:
        L.append(f"| {name}{' (primary)' if name == PRIMARY else ''} | " + " | ".join(ci(res[m]["contrasts"][name]) for m in models) + " |")
    L += ["", "## Rate by arm", "", "| model | scope | " + " | ".join(cases.DOSES) + " |", "|---|---|" + "---|" * len(cases.DOSES)]
    for m in models:
        L.append(f"| {m} | pooled | " + " | ".join(pct(res[m]["rates"].get(f"rule_{d}")) for d in cases.DOSES) + " |")
        for ch in cases.CHARS_F:
            L.append(f"| {m} | {ch} | " + " | ".join(pct(res[m]["by_char"][ch]["rates"].get(f"rule_{d}")) for d in cases.DOSES) + " |")
    L += ["", "## Exclusions and non-decisions per arm", ""]
    for m in models:
        L += [f"- {m} {a}: {res[m]['counts'].get(a, {})}" for a in cases.ARMS]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p12f_frame.p12f", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
        print(f"[p12f] wrote {len(jobs)} jobs for {a.model} to {a.out}")
    else:
        res = {}
        for spec in a.dirs:
            name, _, d = spec.partition("=")
            res[name] = analyze_model(p12ba.load_dir(d))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (a.out / "summary.md").write_text(render(res), encoding="utf-8")
        print(f"wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()

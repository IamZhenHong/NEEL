"""PA-P12e jobs, frontier runner and analysis (design: persona_affect/p12e_value/DESIGN.md).

  uv run python -m persona_affect.p12e_value.p12e jobs --model gemma|q27|gpt --out FILE
  uv run python -m persona_affect.p12e_value.p12e run-frontier --model gpt --jobs FILE --out DIR [--smoke] [--resume [--retry-failed]]
  uv run python -m persona_affect.p12e_value.p12e analyze --dirs gemma=DIR q27=DIR gpt=DIR [--out DIR]

Jobs: the 18 P12b rule items x 7 arms (cases.ARMS) x samples (gemma 8, q27 4, gpt 2), keys
PA-P12E|{model}|agent|{char}|{item}|{arm}|{sample}, P12's balanced option orders, every arm on P12b's rule case.
run-frontier is P12's runner with this mini's ids, cap and share. Scoring and the item-cluster bootstrap are P12c's.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from persona_affect.common import util
from persona_affect.p12_named_traits import cases as p12c_
from persona_affect.p12_named_traits import p12
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12c_dose import p12c
from persona_affect.p12e_value import cases

EXP, MINI, CAP_TOTAL = "PA-P12E", "P12E", 5.0
FRONTIER_SHARE = {"gpt": 1.5}
SAMPLES = {"gemma": 8, "q27": 4, "gpt": 2}
DRAWS = 2000
RESULTS = Path(__file__).resolve().parent / "results"
PRIMARY = "name_value - cross_value"
STATS = {
    PRIMARY: lambda r: r["rule_name_value"] - r["rule_cross_value"],
    "value - none": lambda r: r["rule_value"] - r["rule_none"],
    "name - cross": lambda r: r["rule_name"] - r["rule_cross"],
    "name_value - name": lambda r: r["rule_name_value"] - r["rule_name"],
    "cross_value - cross": lambda r: r["rule_cross_value"] - r["rule_cross"],
    "cross_value - value": lambda r: r["rule_cross_value"] - r["rule_value"],
    "bio_adj - name_value": lambda r: r["rule_bio_adj"] - r["rule_name_value"],
    "bio_adj - value": lambda r: r["rule_bio_adj"] - r["rule_value"],
}


def build_jobs(model: str) -> list[dict]:
    _, fmt, _ = p12.MODELS[model]
    jobs = []
    for char, idx, it in p12bc.items_b():
        for arm in cases.ARMS:
            dose = arm.split("_", 1)[1]
            for s in range(SAMPLES[model]):
                perm = p12c_.perm_for(idx, s)
                key = f"{EXP}|{model}|agent|{char}|{it['id']}|{arm}|{s}"
                jobs.append({"key": key, "exp": EXP, "model": model, "group": "agent", "char": char, "item": it["id"],
                             "arm": arm, "sample": s, "perm": perm, "case": p12bc.rule_case_id(char, it["id"], perm),
                             "cond": cases.cond_id(char, dose), "seed": util.seed(key), "thinking": False, "fmt": fmt})
    random.Random(f"{EXP}|{model}").shuffle(jobs)
    return jobs


@contextmanager
def _as_p12e():
    saved = {k: getattr(p12, k) for k in ("EXP", "MINI", "CAP_TOTAL", "FRONTIER_SHARE")}
    p12.EXP, p12.MINI, p12.CAP_TOTAL, p12.FRONTIER_SHARE = EXP, MINI, CAP_TOTAL, dict(FRONTIER_SHARE)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(p12, k, v)


def run_frontier(model: str, jobs_path, out_dir, **kw) -> dict:
    with _as_p12e():
        return p12.run_frontier(model, jobs_path, out_dir, **kw)


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
    for ch in cases.CHARS_E:
        sub = [r for r in rows if r["char"] == ch]
        res["by_char"][ch] = {"rates": p12c._arm_rates(sub), "contrasts": boot(sub, draws)}
    return res


def render(res: dict) -> str:
    pct, ci = p12c._pct, p12c._ci
    models = list(res)
    L = ["# PA-P12e analysis: character or value content, with the operator rule", "",
         "Canonical (rule-breaking) rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).",
         "", "## Contrasts (points)", "", "| contrast | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for name in STATS:
        L.append(f"| {name}{' (primary)' if name == PRIMARY else ''} | " + " | ".join(ci(res[m]["contrasts"][name]) for m in models) + " |")
    L += ["", "## Rate by arm", "", "| model | scope | " + " | ".join(cases.DOSES) + " |", "|---|---|" + "---|" * len(cases.DOSES)]
    for m in models:
        L.append(f"| {m} | pooled | " + " | ".join(pct(res[m]["rates"].get(f"rule_{d}")) for d in cases.DOSES) + " |")
        for ch in cases.CHARS_E:
            L.append(f"| {m} | {ch} | " + " | ".join(pct(res[m]["by_char"][ch]["rates"].get(f"rule_{d}")) for d in cases.DOSES) + " |")
    L += ["", "## Exclusions and non-decisions per arm", ""]
    for m in models:
        L += [f"- {m} {a}: {res[m]['counts'].get(a, {})}" for a in cases.ARMS]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p12e_value.p12e", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("jobs")
    pj.add_argument("--model", required=True, choices=list(SAMPLES))
    pj.add_argument("--out", required=True)
    pr = sub.add_parser("run-frontier")
    pr.add_argument("--model", required=True, choices=list(FRONTIER_SHARE))
    pr.add_argument("--jobs", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--smoke", action="store_true")
    pr.add_argument("--resume", action="store_true")
    pr.add_argument("--retry-failed", action="store_true")
    pa = sub.add_parser("analyze")
    pa.add_argument("--dirs", nargs="+", required=True, metavar="NAME=DIR")
    pa.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        jobs = build_jobs(a.model)
        p12.write_jobs(a.out, jobs)
        print(f"[p12e] wrote {len(jobs)} jobs for {a.model} to {a.out}")
    elif a.cmd == "run-frontier":
        res = run_frontier(a.model, a.jobs, a.out, smoke=a.smoke, resume=a.resume, retry_failed=a.retry_failed)
        raise SystemExit(0 if res["ok"] else 1)
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

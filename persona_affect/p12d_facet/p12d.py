"""PA-P12d jobs and analysis (design: persona_affect/p12d_facet/DESIGN.md). Gemma 4 31B only, on a pod.

  uv run python -m persona_affect.p12d_facet.p12d jobs --out FILE
  uv run python -m persona_affect.p12d_facet.p12d analyze --dir DIR [--out DIR]

Jobs: the 18 P12b items x 9 arms (cases.ARMS) x N_SAMPLES, keys PA-P12D|gemma|agent|{char}|{item}|{arm}|{sample}, P12's
balanced option orders. Scoring and the item-cluster bootstrap (18 items, 2000 draws, rng 0) are P12c's.
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
from persona_affect.p12d_facet import cases

EXP, MODEL, N_SAMPLES, DRAWS = "PA-P12D", "gemma", 8, 2000
RESULTS = Path(__file__).resolve().parent / "results"
PRIMARY = "rule: bio_adj - bio_far"
STATS = {
    PRIMARY: lambda r: r["rule_bio_adj"] - r["rule_bio_far"],
    "rule: bio_adj - name": lambda r: r["rule_bio_adj"] - r["rule_name"],
    "rule: bio_far - name": lambda r: r["rule_bio_far"] - r["rule_name"],
    "rule: name - cross_name": lambda r: r["rule_name"] - r["rule_cross_name"],
    "rule: noname_adj - none": lambda r: r["rule_noname_adj"] - r["rule_none"],
    "rule: bio_adj - noname_adj": lambda r: r["rule_bio_adj"] - r["rule_noname_adj"],
    "open: bio_adj - bio_far": lambda r: r["open_bio_adj"] - r["open_bio_far"],
    "open: bio_adj - name": lambda r: r["open_bio_adj"] - r["open_name"],
    "open: bio_far - name": lambda r: r["open_bio_far"] - r["open_name"],
}


def build_jobs() -> list[dict]:
    _, fmt, _ = p12.MODELS[MODEL]
    jobs = []
    for char, idx, it in p12bc.items_b():
        for arm in cases.ARMS:
            dose = arm.split("_", 1)[1]
            for s in range(N_SAMPLES):
                perm = p12c_.perm_for(idx, s)
                key = f"{EXP}|{MODEL}|agent|{char}|{it['id']}|{arm}|{s}"
                jobs.append({"key": key, "exp": EXP, "model": MODEL, "group": "agent", "char": char, "item": it["id"],
                             "arm": arm, "sample": s, "perm": perm, "case": cases.case_for(arm, char, it["id"], perm),
                             "cond": cases.cond_id(char, dose), "seed": util.seed(key), "thinking": False, "fmt": fmt})
    random.Random(f"{EXP}|{MODEL}").shuffle(jobs)
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


def analyze(recs: list[dict], draws: int = DRAWS) -> dict:
    rows, counts = p12c.rows_of(recs)
    res = {"rates": p12c._arm_rates(rows), "counts": counts, "contrasts": boot(rows, draws), "by_char": {}}
    for ch in cases.CHARS_D:
        sub = [r for r in rows if r["char"] == ch]
        res["by_char"][ch] = {"rates": p12c._arm_rates(sub), "contrasts": boot(sub, draws)}
    return res


def render(res: dict) -> str:
    pct, ci = p12c._pct, p12c._ci
    L = ["# PA-P12d analysis: persona facets on Gemma 4 31B, with and without the operator rule", "",
         "Canonical rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).", "",
         f"**Primary:** {PRIMARY}: {ci(res['contrasts'][PRIMARY])}", "",
         "## Canonical rate by arm", "", "| scope | " + " | ".join(cases.ARMS) + " |", "|---|" + "---|" * len(cases.ARMS),
         "| pooled | " + " | ".join(pct(res["rates"].get(a)) for a in cases.ARMS) + " |"]
    for ch in cases.CHARS_D:
        L.append(f"| {ch} | " + " | ".join(pct(res["by_char"][ch]["rates"].get(a)) for a in cases.ARMS) + " |")
    L += ["", "## Contrasts (points)", "", "| contrast | pooled | " + " | ".join(cases.CHARS_D) + " |",
          "|---|---|" + "---|" * len(cases.CHARS_D)]
    for name in STATS:
        L.append(f"| {name} | {ci(res['contrasts'][name])} | "
                 + " | ".join(ci(res["by_char"][ch]["contrasts"][name]) for ch in cases.CHARS_D) + " |")
    L += ["", "## Exclusions and non-decisions per arm", ""] + [f"- {a}: {res['counts'].get(a, {})}" for a in cases.ARMS]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p12d_facet.p12d", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("jobs")
    pj.add_argument("--out", required=True)
    pa = sub.add_parser("analyze")
    pa.add_argument("--dir", required=True)
    pa.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    if a.cmd == "jobs":
        jobs = build_jobs()
        p12.write_jobs(a.out, jobs)
        print(f"[p12d] wrote {len(jobs)} jobs to {a.out}")
    else:
        res = analyze(p12ba.load_dir(a.dir))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (a.out / "summary.md").write_text(render(res), encoding="utf-8")
        print(f"wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()

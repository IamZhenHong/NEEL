"""Shared helpers for minis that run persona arms on P12b's rule items (Robin Hood, Sparrow, Captain America items, each
with its operator rule): jobs, scoring and the item-cluster bootstrap. The P12c-P12f modules each carry their own copy;
round-2 minis use this one.

  build_jobs(exp, model, arms, cond_of, samples, suffix_of=None)   decisions on P12b's rule cases, P12's balanced orders
  analyze(recs, stats, chars)                                       rates per arm, contrasts (points, 95% CI), per char
  render(title, res, stats, doses, chars, primary)                  summary.md text

`stats` maps a contrast name to a function of {arm: rate}; a draw where an arm is missing skips that contrast. The bootstrap
resamples the 18 items with replacement (2000 draws, rng 0, percentile interval), as in P12b-P12f.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from persona_affect.common import util
from persona_affect.p12_named_traits import cases as p12c_
from persona_affect.p12_named_traits import p12
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12c_dose import p12c

DRAWS = 2000


def build_jobs(exp: str, model: str, arms, cond_of, samples: int, suffix_of=None) -> list[dict]:
    """One decision job per P12b item x arm x sample. `cond_of(char, arm)` gives the registered cond id; `suffix_of(arm)` an
    optional system suffix."""
    _, fmt, _ = p12.MODELS[model]
    jobs = []
    for char, idx, it in p12bc.items_b():
        for arm in arms:
            sfx = suffix_of(arm) if suffix_of else None
            for s in range(samples):
                perm = p12c_.perm_for(idx, s)
                key = f"{exp}|{model}|agent|{char}|{it['id']}|{arm}|{s}"
                jobs.append({"key": key, "exp": exp, "model": model, "group": "agent", "char": char, "item": it["id"],
                             "arm": arm, "sample": s, "perm": perm, "case": p12bc.rule_case_id(char, it["id"], perm),
                             "cond": cond_of(char, arm), **({"system_suffix": sfx} if sfx else {}),
                             "seed": util.seed(key), "thinking": False, "fmt": fmt})
    random.Random(f"{exp}|{model}").shuffle(jobs)
    return jobs


def boot(rows: list[dict], stats: dict, draws: int = DRAWS) -> dict:
    items = sorted({r["item"] for r in rows})
    point = p12c._rates(rows, items, np.ones(len(items)))
    rng = np.random.default_rng(0)
    samples = defaultdict(list)
    for _ in range(draws):
        q = p12c._rates(rows, items, np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items)))
        for name, f in stats.items():
            try:
                samples[name].append(100 * f(q))
            except KeyError:
                pass
    out = {}
    for name, f in stats.items():
        try:
            d = 100 * f(point)
        except KeyError:
            out[name] = {"diff": None, "lo": None, "hi": None}
            continue
        s = samples[name]
        out[name] = {"diff": d, "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5))}
    return out


def analyze(recs: list[dict], stats: dict, chars, draws: int = DRAWS) -> dict:
    rows, counts = p12c.rows_of(recs)
    res = {"rates": p12c._arm_rates(rows), "counts": counts, "contrasts": boot(rows, stats, draws), "by_char": {}}
    for ch in chars:
        sub = [r for r in rows if r["char"] == ch]
        res["by_char"][ch] = {"rates": p12c._arm_rates(sub), "contrasts": boot(sub, stats, draws)}
    return res


def render(title: str, res: dict, stats: dict, doses, chars, primary: str | None = None) -> str:
    pct, ci = p12c._pct, p12c._ci
    models = list(res)
    L = [f"# {title}", "", "Rule-breaking (canonical) rate among decided episodes. Intervals: item-cluster bootstrap over 18 "
         "items (2000 draws).", "", "## Contrasts (points)", "", "| contrast | " + " | ".join(models) + " |",
         "|---|" + "---|" * len(models)]
    for name in stats:
        L.append(f"| {name}{' (primary)' if name == primary else ''} | "
                 + " | ".join(ci(res[m]["contrasts"][name]) for m in models) + " |")
    L += ["", "## Rate by arm", "", "| model | scope | " + " | ".join(doses) + " |", "|---|---|" + "---|" * len(doses)]
    for m in models:
        L.append(f"| {m} | pooled | " + " | ".join(pct(res[m]["rates"].get(f"rule_{d}")) for d in doses) + " |")
        for ch in chars:
            L.append(f"| {m} | {ch} | " + " | ".join(pct(res[m]["by_char"][ch]["rates"].get(f"rule_{d}")) for d in doses) + " |")
    L += ["", "## Exclusions and non-decisions per arm", ""]
    for m in models:
        L += [f"- {m} {a}: {res[m]['counts'].get(a, {})}" for a in (f"rule_{d}" for d in doses)]
    return "\n".join(L) + "\n"

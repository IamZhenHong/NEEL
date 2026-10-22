"""PA-P12c jobs and analysis (design: persona_affect/p12c_dose/DESIGN.md). Gemma 4 31B only, on a pod.

  uv run python -m persona_affect.p12c_dose.p12c jobs --out FILE
  uv run python -m persona_affect.p12c_dose.p12c analyze --dir DIR [--out DIR]

Jobs: decisions only, the 18 P12b items x 14 arms (cases.ARMS) x N_SAMPLES samples, keys
PA-P12C|gemma|agent|{char}|{item}|{arm}|{sample}; the option order is P12's perm_for(item index, sample), so every arm sees
the same orders. Shuffled with random.Random("PA-P12C|gemma").

Analysis: P12's decision() and P12b's status() (failed / cut / stale left out and counted). Canonical rate per arm, and
contrasts in points with an item-cluster bootstrap over the 18 items (2000 draws, rng 0), as in P12b.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from persona_affect.common import util
from persona_affect.p12_named_traits import analyze as p12a
from persona_affect.p12_named_traits import cases as p12c_
from persona_affect.p12_named_traits import p12
from persona_affect.p12b_rules import analyze as p12ba
from persona_affect.p12b_rules import cases as p12bc
from persona_affect.p12c_dose import cases

EXP, MODEL, N_SAMPLES, DRAWS = "PA-P12C", "gemma", 8, 2000
RESULTS = Path(__file__).resolve().parent / "results"


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


# ---------------------------------------------------------------- analysis

def rows_of(recs: list[dict]) -> tuple[list[dict], dict]:
    rows, counts = [], defaultdict(Counter)
    for r in recs:
        job = r["job"]
        st = p12ba.status(r)
        if st != "ok":
            counts[job["arm"]][st] += 1
            continue
        d = p12a.decision(r)
        counts[job["arm"]][d or "undecided"] += 1
        if d in ("canonical", "generic", "other"):
            rows.append({"arm": job["arm"], "char": job["char"], "item": f"{job['char']}/{job['item']}",
                         "y": int(d == "canonical")})
    return rows, {a: dict(c) for a, c in counts.items()}


def _spec(x: str) -> dict:
    """The contrasts within one arm family x (open or rule)."""
    q = lambda d: f"{x}_{d}"
    return {
        f"{x}: thickening (name_bio - cross_bio) - (name - cross_name)":
            lambda r: (r[q("name_bio")] - r[q("cross_bio")]) - (r[q("name")] - r[q("cross_name")]),
        f"{x}: name - cross_name": lambda r: r[q("name")] - r[q("cross_name")],
        f"{x}: name_bio - cross_bio": lambda r: r[q("name_bio")] - r[q("cross_bio")],
        f"{x}: name_facts - name": lambda r: r[q("name_facts")] - r[q("name")],
        f"{x}: name_bio - name": lambda r: r[q("name_bio")] - r[q("name")],
        f"{x}: bio_noname - none": lambda r: r[q("bio_noname")] - r[q("none")],
        f"{x}: name_bio - bio_noname": lambda r: r[q("name_bio")] - r[q("bio_noname")],
        f"{x}: cross_bio - cross_name": lambda r: r[q("cross_bio")] - r[q("cross_name")],
    }


STATS = {**_spec("rule"), **_spec("open")}
PRIMARY = "rule: thickening (name_bio - cross_bio) - (name - cross_name)"


def _rates(rows, items, w):
    wi = dict(zip(items, w))
    num, den = defaultdict(float), defaultdict(float)
    for r in rows:
        k = wi.get(r["item"], 0)
        if k:
            num[r["arm"]] += k * r["y"]
            den[r["arm"]] += k
    return {a: num[a] / den[a] for a in den if den[a]}


def boot(rows: list[dict], draws: int = DRAWS) -> dict:
    items = sorted({r["item"] for r in rows})
    point = _rates(rows, items, np.ones(len(items)))
    rng = np.random.default_rng(0)
    samples = defaultdict(list)
    for _ in range(draws):
        q = _rates(rows, items, np.bincount(rng.integers(0, len(items), len(items)), minlength=len(items)))
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


def _arm_rates(rows):
    t = defaultdict(lambda: [0, 0])
    for r in rows:
        t[r["arm"]][0] += r["y"]
        t[r["arm"]][1] += 1
    return {a: {"k": k, "n": n, "rate": k / n} for a, (k, n) in t.items()}


def analyze(recs: list[dict], draws: int = DRAWS) -> dict:
    rows, counts = rows_of(recs)
    res = {"rates": _arm_rates(rows), "counts": counts, "contrasts": boot(rows, draws), "by_char": {}}
    for ch in cases.CHARS_C:
        sub = [r for r in rows if r["char"] == ch]
        res["by_char"][ch] = {"rates": _arm_rates(sub), "contrasts": boot(sub, draws)}
    return res


def _pct(c):
    return "-" if not c else f"{100 * c['rate']:.1f}% ({c['k']}/{c['n']})"


def _ci(c):
    return "n/a" if not c or c["diff"] is None else f"{c['diff']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"


def render(res: dict) -> str:
    L = ["# PA-P12c analysis: persona dose on Gemma 4 31B, with and without the operator rule", "",
         "Canonical rate among decided episodes. Intervals: item-cluster bootstrap over 18 items (2000 draws).", "",
         f"**Primary:** {PRIMARY}: {_ci(res['contrasts'][PRIMARY])}", "",
         "## Canonical rate by dose", "", "| family | " + " | ".join(cases.DOSES) + " |", "|---|" + "---|" * len(cases.DOSES)]
    for x in ("open", "rule"):
        L.append(f"| {x} | " + " | ".join(_pct(res["rates"].get(f"{x}_{d}")) for d in cases.DOSES) + " |")
    L += ["", "## Contrasts (points)", "", "| contrast | pooled | " + " | ".join(cases.CHARS_C) + " |",
          "|---|---|" + "---|" * len(cases.CHARS_C)]
    for name in STATS:
        L.append(f"| {name} | {_ci(res['contrasts'][name])} | "
                 + " | ".join(_ci(res["by_char"][ch]["contrasts"][name]) for ch in cases.CHARS_C) + " |")
    L += ["", "## Per character, canonical rate by dose", ""]
    for ch in cases.CHARS_C:
        L += [f"### {ch}", "", "| family | " + " | ".join(cases.DOSES) + " |", "|---|" + "---|" * len(cases.DOSES)]
        for x in ("open", "rule"):
            L.append(f"| {x} | " + " | ".join(_pct(res["by_char"][ch]["rates"].get(f"{x}_{d}")) for d in cases.DOSES) + " |")
        L.append("")
    L += ["## Exclusions and non-decisions per arm", ""] + [f"- {a}: {res['counts'].get(a, {})}" for a in cases.ARMS]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.p12c_dose.p12c", description=__doc__,
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
        print(f"[p12c] wrote {len(jobs)} jobs to {a.out}; launch with MODEL={p12.MODELS[MODEL][0]} FMT={p12.MODELS[MODEL][1]}")
    else:
        res = analyze(p12ba.load_dir(a.dir))
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        (a.out / "summary.md").write_text(render(res), encoding="utf-8")
        print(f"wrote {a.out / 'summary.md'}")


if __name__ == "__main__":
    main()

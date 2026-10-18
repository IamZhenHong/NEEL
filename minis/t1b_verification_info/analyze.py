"""T1b analysis (as frozen in DESIGN.md). Usage: python3 minis/t1b_verification_info/analyze.py -> results/analysis.json."""
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
eps = [json.loads(l) for l in open(R / "episodes.jsonl")]
WORDS = re.compile(r"\b(check(?:ed|ing|s)?|verif(?:y|ied|ying|ication)|double-check\w*|recalculat\w*|confirm\w*|"
                   r"careful\w*|make sure|re-add\w*)\b", re.I)
CONDS, UNVER, RECOMP = ["base", "meticulous"], ["T3", "T4", "T5"], ["T1", "T2"]


def opened(r):
    return [t for t in UNVER if t != r["verified"]]


def n_open(r):            # distinct open (unverifiable, not verified) totals checked
    return len(set(r["checks"]) & set(opened(r)))


def rate(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def acc(r):
    return None if r["correct"] is None else int(r["correct"])


def stats(rs):
    text = [" ".join(r["reasoning"]) + " " + r["final"] for r in rs]
    return {"n": len(rs), "unparsed": sum(r["correct"] is None for r in rs),
            "mean_open_checked": statistics.mean(n_open(r) for r in rs),
            "p_any_open": rate([int(n_open(r) > 0) for r in rs]),
            "p_both_open": rate([int(n_open(r) == 2) for r in rs]),
            "p_check_verified": rate([int(r["verified"] in r["checks"]) for r in rs]),
            "mean_recomp_checks": statistics.mean(sum(c in RECOMP for c in r["checks"]) for r in rs),
            "mean_calls": statistics.mean(len(r["checks"]) for r in rs),
            "accuracy": rate([acc(r) for r in rs]),
            "acc_open_error": rate([acc(r) for r in rs if r["kind"] == "unver"]),
            "acc_recomp_error": rate([acc(r) for r in rs if r["kind"] == "recomp"]),
            "acc_clean": rate([acc(r) for r in rs if r["kind"] == "clean"]),
            "p_wrong_checked_open_error": rate([int(r["wrong_id"] in r["checks"]) for r in rs if r["kind"] == "unver"]),
            "mean_score": statistics.mean(r["score"] for r in rs),
            "verif_words_per_1k": 1000 * sum(len(WORDS.findall(t)) for t in text) / max(1, sum(len(t) for t in text))}


def paired(metric, keep=lambda r: True):
    cells = defaultdict(lambda: {c: [] for c in CONDS})
    for r in eps:
        v = metric(r)
        if v is not None and keep(r):
            cells[(r["report"], r["C"])][r["cond"]].append(v)
    d = [statistics.mean(v["meticulous"]) - statistics.mean(v["base"]) for v in cells.values()
         if v["meticulous"] and v["base"]]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"diff": statistics.mean(d), "ci": [bs[249], bs[9749]], "n_cells": len(d)}


M = {"open_checks": n_open, "accuracy": acc, "check_verified": lambda r: int(r["verified"] in r["checks"]),
     "any_open": lambda r: int(n_open(r) > 0), "score": lambda r: r["score"]}
out = {"spend_usd": round(sum(json.loads(l)["cost"] for l in open(R / "api_ledger.jsonl")), 3),
       "spend_smoke_usd": round(sum(json.loads(l)["cost"] for l in open(R / "smoke.jsonl")), 3)
       if (R / "smoke.jsonl").exists() else 0,
       "by_cond": {c: stats([r for r in eps if r["cond"] == c]) for c in CONDS},
       "by_cond_cost": {f"{c}|C{C}": stats([r for r in eps if r["cond"] == c and r["C"] == C])
                        for c in CONDS for C in (1, 4)},
       "contrasts": {k: paired(f) for k, f in M.items()},
       "contrasts_by_cost": {f"{k}|C{C}": paired(f, lambda r, C=C: r["C"] == C) for k, f in M.items() for C in (1, 4)},
       "contrast_accuracy_open_error_reports": paired(acc, lambda r: r["kind"] == "unver")}
n = len(eps)
out["unparsed"] = sum(r["correct"] is None for r in eps)
out["flag_unparsed_gt10pct"] = out["unparsed"] > 0.1 * n
json.dump(out, open(R / "analysis.json", "w"), indent=1)
print(f"T1b: {n} episodes, {out['unparsed']} unparsed; spend ${out['spend_usd']} (smoke ${out['spend_smoke_usd']})")
for k, v in {**out["by_cond"], **out["by_cond_cost"]}.items():
    print(f"  {k:15s} " + " ".join(f"{m}={v[m]:.2f}" if isinstance(v[m], float) else f"{m}={v[m]}" for m in v))
for k, v in {**out["contrasts"], **out["contrasts_by_cost"],
             "acc_open_error_reports": out["contrast_accuracy_open_error_reports"]}.items():
    print(f"  met-base {k:22s}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] ({v['n_cells']} cells)")

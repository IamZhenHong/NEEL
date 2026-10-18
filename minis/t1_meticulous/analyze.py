"""T1 analysis (as frozen in DESIGN.md). Usage: python3 minis/t1_meticulous/analyze.py -> results/analysis.json."""
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
eps = [json.loads(l) for l in open(R / "episodes.jsonl")]
selfr = [json.loads(l) for l in open(R / "self.jsonl")]
WORDS = re.compile(r"\b(check(?:ed|ing|s)?|verif(?:y|ied|ying|ication)|double-check\w*|recalculat\w*|confirm\w*|"
                   r"careful\w*|make sure|re-add\w*)\b", re.I)
CONDS = ["base", "meticulous"]


def rate(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stats(rs):
    err = [r for r in rs if r["wrong_id"]]
    text = [" ".join(r["reasoning"]) + " " + r["final"] for r in rs]
    return {"n": len(rs), "unparsed": sum(r["correct"] is None for r in rs),
            "p_any_check": rate([int(bool(r["checks"])) for r in rs]),
            "mean_checks": statistics.mean(len(r["checks"]) for r in rs),
            "p_check_verified": rate([int(r["verified"] in r["checks"]) for r in rs]),
            "p_check_wrong_in_error_reports": rate([int(r["wrong_id"] in r["checks"]) for r in err]),
            "accuracy": rate([int(r["correct"]) if r["correct"] is not None else None for r in rs]),
            "acc_error_reports": rate([int(r["correct"]) if r["correct"] is not None else None for r in err]),
            "acc_ok_reports": rate([int(r["correct"]) if r["correct"] is not None else None
                                    for r in rs if not r["wrong_id"]]),
            "mean_score": statistics.mean(r["score"] for r in rs),
            "verif_words_per_1k": 1000 * sum(len(WORDS.findall(t)) for t in text) / max(1, sum(len(t) for t in text))}


def paired(metric):
    cells = defaultdict(lambda: {c: [] for c in CONDS})
    for r in eps:
        v = metric(r)
        if v is not None:
            cells[(r["report"], r["C"])][r["cond"]].append(v)
    d = [statistics.mean(v["meticulous"]) - statistics.mean(v["base"]) for v in cells.values()
         if v["meticulous"] and v["base"]]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(10000))
    return {"diff": statistics.mean(d), "ci": [bs[249], bs[9749]], "n_cells": len(d)}


out = {"spend_usd": round(sum(json.loads(l)["cost"] for l in open(R / "api_ledger.jsonl")), 3),
       "by_cond": {c: stats([r for r in eps if r["cond"] == c]) for c in CONDS},
       "by_cond_cost": {f"{c}|C{C}": stats([r for r in eps if r["cond"] == c and r["C"] == C])
                        for c in CONDS for C in (1, 4)},
       "contrast_any_check": paired(lambda r: int(bool(r["checks"]))),
       "contrast_accuracy": paired(lambda r: None if r["correct"] is None else int(r["correct"])),
       "self_rating": {c: [s["rating"] for s in selfr if s["cond"] == c] for c in CONDS}}
for c in CONDS:
    rs = [x for x in out["self_rating"][c] if x is not None]
    out["self_rating_mean_" + c] = statistics.mean(rs) if rs else None
json.dump(out, open(R / "analysis.json", "w"), indent=1)
print(f"T1: {len(eps)} episodes; spend ${out['spend_usd']}")
for k, v in {**out["by_cond"], **out["by_cond_cost"]}.items():
    print(f"  {k:16s} " + " ".join(f"{m}={v[m]:.2f}" if isinstance(v[m], float) else f"{m}={v[m]}" for m in v))
for k in ("contrast_any_check", "contrast_accuracy"):
    v = out[k]
    print(f"  {k}: {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
print("  self-ratings:", out["self_rating"])

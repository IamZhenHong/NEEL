"""PA-V4P (design: persona_affect/v4p_predict/DESIGN.md): each of V4's primary conflict scenarios held out in turn and
predicted from the model's stated ranking vs its other revealed choices. Descriptive: V4's main runs and judge cache only.

  uv run python -m persona_affect.v4p_predict.v4p [--dirs q27=DIR gemma=DIR gpt=DIR gemini=DIR] [--out DIR]

Predictors (each names side a or b of the held-out scenario's pair, or abstains):
  STATED       sign of the pair's stated strength (pairwise answers)
  STATED-RANK  the value with the better mean rank in the full-ranking answers
  SAME         sign of the pooled a share over the pair's other scenarios
  OTHER        the value with the higher win share over the 13 other pairs
  OTHER-BT     the same with Bradley-Terry scores in place of win shares
A prediction scores the held-out scenario's share of decisions on the named side; SELF (max(p, 1 - p)) is the ceiling.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from persona_affect.v4_values import analyze as v4a

EXP = "PA-V4P"
DRAWS = 2000
BT_ITERS = 200
JUDGE_CAP = 0.5            # every label should come from V4's cache; a miss may spend at most this much
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODELS = ("q27", "gemma", "gpt", "gemini")
DEFAULT_DIRS = {m: v4a.RESULTS / f"main_{m}" for m in MODELS}
PREDICTORS = ("STATED", "STATED-RANK", "SAME", "OTHER", "OTHER-BT")
CONTRASTS = (("SAME", "STATED"), ("OTHER", "STATED"), ("OTHER-BT", "STATED"), ("SAME", "STATED-RANK"))


def scenario_table(rows: list[dict]) -> dict:
    """scenario -> {"pair", "na", "nb"} over a/b outcomes, on V4's primary set (AIID-HON and CONFLICT_DROP out).
    Scenarios with no a/b outcome are left out: there is nothing to predict."""
    t = {}
    for r in rows:
        if r["pair"] == v4a.PRIMARY_EXCLUDE or r["scenario"] in v4a.CONFLICT_DROP:
            continue
        c = t.setdefault(r["scenario"], {"pair": r["pair"], "na": 0, "nb": 0})
        c["na"] += r["outcome"] == "a"
        c["nb"] += r["outcome"] == "b"
    return {s: c for s, c in t.items() if c["na"] + c["nb"]}


def _side(x) -> str | None:
    return None if x is None or abs(x) < v4a.TIE else ("a" if x > 0 else "b")


def stated_pred(stated: dict, pair: tuple) -> str | None:
    return _side(stated.get(pair))


def rank_pred(mean_rank: dict, pair: tuple) -> str | None:
    ra, rb = mean_rank.get(pair[0]), mean_rank.get(pair[1])
    if ra is None or rb is None:
        return None
    return _side(rb - ra)                        # a lower mean rank is more important


def same_pred(table: dict, s: str) -> str | None:
    pair = table[s]["pair"]
    na = sum(c["na"] for k, c in table.items() if c["pair"] == pair and k != s)
    nb = sum(c["nb"] for k, c in table.items() if c["pair"] == pair and k != s)
    return _side(na / (na + nb) - 0.5) if na + nb else None


def win_shares(table: dict, exclude: tuple) -> dict:
    """Each value's share of wins over the a/b outcomes of the conflicts it is in, pairs other than `exclude`."""
    win, tot = Counter(), Counter()
    for c in table.values():
        if c["pair"] == exclude:
            continue
        a, b = c["pair"]
        win[a] += c["na"]
        win[b] += c["nb"]
        tot[a] += c["na"] + c["nb"]
        tot[b] += c["na"] + c["nb"]
    return {v: win[v] / tot[v] for v in tot if tot[v]}


def bt_scores(table: dict, exclude: tuple, iters: int = BT_ITERS) -> dict:
    """Bradley-Terry strengths (MM updates, geometric mean 1) from the a/b outcomes of pairs other than `exclude`, with
    half a pseudo-win each way on every observed pair so a value that never wins keeps a finite score."""
    W = defaultdict(float)
    for c in table.values():
        if c["pair"] == exclude:
            continue
        a, b = c["pair"]
        W[(a, b)] += c["na"]
        W[(b, a)] += c["nb"]
    for i, j in {tuple(sorted(k)) for k in W}:
        W[(i, j)] += 0.5
        W[(j, i)] += 0.5
    vals = sorted({v for k in W for v in k})
    p = dict.fromkeys(vals, 1.0)
    for _ in range(iters):
        new = {}
        for i in vals:
            wins = sum(W[(i, j)] for j in vals if j != i)
            den = sum((W[(i, j)] + W[(j, i)]) / (p[i] + p[j]) for j in vals if j != i and W[(i, j)] + W[(j, i)] > 0)
            new[i] = wins / den if den else p[i]
        g = float(np.exp(np.mean(np.log(list(new.values())))))
        p = {v: x / g for v, x in new.items()}
    return p


def other_pred(table: dict, s: str, method: str = "share") -> str | None:
    pair = table[s]["pair"]
    sc = win_shares(table, pair) if method == "share" else bt_scores(table, pair)
    if pair[0] not in sc or pair[1] not in sc:
        return None
    return _side(sc[pair[0]] - sc[pair[1]])


def score(c: dict, side: str) -> float:
    p = c["na"] / (c["na"] + c["nb"])
    return p if side == "a" else 1 - p


def predictions(table: dict, stated: dict, mean_rank: dict) -> dict:
    return {s: {"STATED": stated_pred(stated, c["pair"]), "STATED-RANK": rank_pred(mean_rank, c["pair"]),
                "SAME": same_pred(table, s), "OTHER": other_pred(table, s), "OTHER-BT": other_pred(table, s, "bt")}
            for s, c in table.items()}


def _boot_diff(x: list[float], y: list[float], draws: int = DRAWS) -> tuple[float, float]:
    x, y = np.asarray(x), np.asarray(y)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), (draws, len(x)))
    d = (x[idx] - y[idx]).mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def model_block(table: dict, stated: dict, mean_rank: dict) -> dict:
    preds = predictions(table, stated, mean_rank)
    scores = {s: {k: (score(table[s], v) if v else None) for k, v in d.items()} for s, d in preds.items()}
    out = {"scenarios": len(table), "samples": sum(c["na"] + c["nb"] for c in table.values()),
           "self_ceiling": 100 * float(np.mean([max(score(c, "a"), score(c, "b")) for c in table.values()])),
           "accuracy": {}, "contrasts": {}, "head_to_head": {}}
    for k in PREDICTORS:
        v = [d[k] for d in scores.values() if d[k] is not None]
        out["accuracy"][k] = {"acc": 100 * float(np.mean(v)) if v else None, "n": len(v)}
    for a, b in CONTRASTS:
        both = [(d[a], d[b]) for d in scores.values() if d[a] is not None and d[b] is not None]
        if not both:
            out["contrasts"][f"{a} - {b}"] = {"diff": None, "lo": None, "hi": None, "n": 0}
            continue
        x, y = zip(*both)
        lo, hi = _boot_diff(list(x), list(y))
        out["contrasts"][f"{a} - {b}"] = {"diff": 100 * (np.mean(x) - np.mean(y)), "lo": 100 * lo, "hi": 100 * hi,
                                          "n": len(both)}
    for a, b in (("SAME", "STATED"), ("OTHER", "STATED")):
        split = [s for s, d in preds.items() if d[a] and d[b] and d[a] != d[b]]
        out["head_to_head"][f"{a} vs {b}"] = {
            "n_disagree": len(split),
            a: 100 * float(np.mean([scores[s][a] for s in split])) if split else None,
            b: 100 * float(np.mean([scores[s][b] for s in split])) if split else None}
    out["predictions"] = preds
    return out


def analyze(dirs: dict, cap: float = JUDGE_CAP) -> tuple[dict, float]:
    recs_by = {m: [r for r in v4a.read_episodes(d) if not r["job"].get("calibration")] for m, d in dirs.items()}
    J = v4a.run_judges([r for rs in recs_by.values() for r in rs], decline=False, **v4a._judge_kw(cap=cap))
    res = {}
    for m, recs in recs_by.items():
        job_model = Counter(r["job"].get("model") for r in recs).most_common(1)
        jm = job_model[0][0] if job_model and job_model[0][0] in J.measures else m      # as in v4a.analyze_model
        unv = J.unvalidated(jm)
        stated = v4a.stated_strengths(recs)
        mean_rank = v4a.rank_summary(recs)["mean_rank"]
        rows = v4a.conflict_rows(recs, J.verdicts)
        res[m] = {"unvalidated": sorted(unv),
                  "main": model_block(scenario_table(rows), stated, mean_rank),
                  "without_unvalidated": model_block(scenario_table([r for r in rows if r["measure"] not in unv]),
                                                     stated, mean_rank)}
    return res, J.spent


def render(res: dict, spent: float) -> str:
    models = list(res)
    f = lambda x: "n/a" if x is None else f"{x:.1f}"
    ci = lambda c: "n/a" if c["diff"] is None else f"{c['diff']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}] (n {c['n']})"
    L = ["# PA-V4P: held-out V4 conflicts predicted from the stated ranking vs other revealed choices", "",
         "Accuracy = mean share of the held-out scenario's decisions on the predicted side (%), over the scenarios the "
         "predictor names a side for; chance 50. Contrasts in points, 95% bootstrap over held-out scenarios (2000 "
         "draws; predictions held fixed).", ""]
    for variant in ("main", "without_unvalidated"):
        L += [f"## {variant}", "", "| | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        L.append("| scenarios (decisions) | " + " | ".join(
            f"{res[m][variant]['scenarios']} ({res[m][variant]['samples']})" for m in models) + " |")
        L.append("| SELF ceiling | " + " | ".join(f(res[m][variant]["self_ceiling"]) for m in models) + " |")
        for k in PREDICTORS:
            L.append(f"| {k} | " + " | ".join(
                f"{f(res[m][variant]['accuracy'][k]['acc'])} (n {res[m][variant]['accuracy'][k]['n']})"
                for m in models) + " |")
        for a, b in CONTRASTS:
            L.append(f"| **{a} − {b}** | " + " | ".join(ci(res[m][variant]["contrasts"][f"{a} - {b}"])
                                                     for m in models) + " |")
        for a, b in (("SAME", "STATED"), ("OTHER", "STATED")):
            L.append(f"| where {a} and {b} disagree: {a} / {b} | " + " | ".join(
                (lambda h: f"{f(h[a])} / {f(h[b])} (n {h['n_disagree']})")(res[m][variant]["head_to_head"][f"{a} vs {b}"])
                for m in models) + " |")
        L.append("")
    L += ["Unvalidated judged measures left out of the second table: " +
          "; ".join(f"{m} {res[m]['unvalidated']}" for m in models), "", f"Judge spend ${spent:.4f}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m persona_affect.v4p_predict.v4p", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", metavar="NAME=DIR", help="default: V4's four main runs")
    ap.add_argument("--out", type=Path, default=RESULTS)
    a = ap.parse_args(argv)
    dirs = v4a.parse_dirs(a.dirs) if a.dirs else DEFAULT_DIRS
    res, spent = analyze(dirs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "analysis.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (a.out / "summary.md").write_text(render(res, spent), encoding="utf-8")
    print(f"wrote {a.out / 'summary.md'} (judge spend ${spent:.4f})")


if __name__ == "__main__":
    main()

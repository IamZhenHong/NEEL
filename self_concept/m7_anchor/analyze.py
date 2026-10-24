"""SC-M7 frozen analysis (rules in DESIGN.md section 4).

Per model, axis, arm, rung k and measure: the share of parsed answers picking rung k's option, averaged over the two
option orders. Shifts are self (or mention) minus base on the same items, in points:
  dA_k = act shift, dS_k = stated shift, gap_k = dS_k - dA_k (negative: stated moves less than the act).
far = mean over k in {4, 5} (3-4 steps from the default), near = mean over k in {2, 3}.
Bootstrap: units resampled with replacement (the 8 cases for act, the 8 stems for stated, per axis), the same draws for
every model, arm and rung, so differences are paired; 10,000 draws, rng 0; 95% percentile intervals.

  uv run python -m self_concept.m7_anchor.analyze --res self_concept/m7_anchor/results/main
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from self_concept.m7_anchor import items as I

N_BOOT = 10000
FAR, NEAR = (4, 5), (2, 3)
EFFECT = 15.0
EPS = 0.005
MEASURES = ("act", "stated")


def load(res: Path) -> list[dict]:
    rows = []
    for p in sorted(Path(res).glob("*.jsonl")):
        rows += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return [r for r in rows if r["key"].startswith(I.EXP + "|")]


def units(axis: str, measure: str) -> list[str]:
    return [c[0] for c in I.CASES[axis]] if measure == "act" else [f"s{i}" for i in range(len(I.STEMS))]


def tables(rows):
    """(model, axis, arm, k, measure) -> (sums, counts), each of shape (2 orders, 8 units); parsed answers only."""
    T = {}
    for r in rows:
        key = (r["model"], r["axis"], r["arm"], r["k"], r["measure"])
        if key not in T:
            n = len(units(r["axis"], r["measure"]))
            T[key] = (np.zeros((2, n)), np.zeros((2, n)))
        if r["y"] is None:
            continue
        u = units(r["axis"], r["measure"]).index(r["unit"])
        T[key][0][r["order"], u] += r["y"]
        T[key][1][r["order"], u] += 1
    return T


def draws(n_boot: int) -> dict:
    """(axis, measure) -> weights (n_boot + 1, n_units); row 0 is all ones (the point estimate)."""
    rng = np.random.default_rng(0)
    W = {}
    for axis in I.AXES:
        for m in MEASURES:
            n = len(units(axis, m))
            w = np.vstack([np.ones(n)] + [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])
            W[(axis, m)] = w.astype(float)
    return W


def rate(T, W, key) -> np.ndarray:
    """Order-averaged rate (points) per draw; nan where an order has no parsed answers in the draw."""
    s, n = T[key]
    w = W[(key[1], key[4])]
    with np.errstate(invalid="ignore", divide="ignore"):
        per = (w @ s.T) / (w @ n.T)          # (draws, 2 orders)
    return 100 * per.mean(axis=1)


def ci(v: np.ndarray) -> dict:
    d = v[1:][np.isfinite(v[1:])]
    if not np.isfinite(v[0]) or not len(d):
        return {"est": None, "lo": None, "hi": None}
    return {"est": float(v[0]), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def verdict(gap_far: dict, trend: dict, gate: bool) -> str:
    if not gate:
        return "induction too weak"
    e, lo, hi = gap_far["est"], gap_far["lo"], gap_far["hi"]
    if e is None:
        return "no data"
    if e <= -EFFECT and hi < 0:
        return "stated snaps back" if (trend["est"] is not None and trend["est"] < 0) else "stated lags at every distance"
    if e >= EFFECT and lo > 0:
        return "stated runs ahead"
    if lo > -EFFECT and hi < EFFECT:
        return "stated tracks the act"
    return "inconclusive"


def words_check(dS_self: dict, dS_men: dict, diff: dict) -> str:
    if dS_self["est"] is None or dS_self["est"] < EFFECT:
        return "stated does not move under self (nothing to explain)"
    if dS_men["est"] >= 0.5 * dS_self["est"]:
        return "stated moves with the words (mention >= half of self): the boring explanation is not excluded"
    if diff["lo"] > 0:
        return "stated follows the attribution, not just the words (mention < half of self, CI excludes 0)"
    return "unclear"


def _lg(p: float) -> float:
    q = min(max(p, EPS), 1 - EPS)
    return math.log(q / (1 - q))


def order_check(T) -> dict:
    """SC-M3's decomposition per cell: x / y = rung k's share when shown as A (order 1) / as B (order 0)."""
    out = {}
    for key, (s, n) in sorted(T.items()):
        if n[0].sum() == 0 or n[1].sum() == 0:
            continue
        y, x = s[0].sum() / n[0].sum(), s[1].sum() / n[1].sum()
        content, letter = (_lg(x) + _lg(y)) / 2, (_lg(x) - _lg(y)) / 2
        avg = (x + y) / 2
        out["|".join(map(str, key))] = {"x_as_A": round(100 * x, 1), "y_as_B": round(100 * y, 1),
                                        "content": round(content, 2), "letter": round(letter, 2),
                                        "min": round(100 * min(x, y), 1),
                                        "flag": bool(abs(letter) > abs(content) and 0.35 <= avg <= 0.65)}
    return out


def analyze(rows: list[dict], n_boot: int = N_BOOT) -> dict:
    T = tables(rows)
    W = draws(n_boot)
    models = [m for m in I.MODELS if any(k[0] == m for k in T)]
    res = {"models": models, "per": {}, "pooled": {}, "parse": {}, "order": order_check(T)}
    pooled_gap = defaultdict(list)
    for m in models:
        for axis in I.AXES:
            R = {}
            for arm in ("base", "self", "mention"):
                for k in I.RUNGS:
                    for ms in MEASURES:
                        if (m, axis, arm, k, ms) in T:
                            R[(arm, k, ms)] = rate(T, W, (m, axis, arm, k, ms))
            if not all((arm, k, ms) in R for arm in ("base", "self", "mention") for k in I.RUNGS for ms in MEASURES):
                continue
            d = {arm: {ms: {k: R[(arm, k, ms)] - R[("base", k, ms)] for k in I.RUNGS} for ms in MEASURES}
                 for arm in ("self", "mention")}
            gap = {k: d["self"]["stated"][k] - d["self"]["act"][k] for k in I.RUNGS}
            mean = lambda f, ks: np.mean([f[k] for k in ks], axis=0)   # noqa: E731
            dA_far, dS_far = mean(d["self"]["act"], FAR), mean(d["self"]["stated"], FAR)
            gap_far, gap_near = mean(gap, FAR), mean(gap, NEAR)
            dS_men_far, dA_men_far = mean(d["mention"]["stated"], FAR), mean(d["mention"]["act"], FAR)
            A, S = ci(dA_far), ci(dS_far)
            gate = any(c["est"] is not None and c["est"] >= EFFECT and c["lo"] > 0 for c in (A, S))
            g, tr = ci(gap_far), ci(gap_far - gap_near)
            base_act = {k: ci(R[("base", k, "act")]) for k in I.RUNGS}
            res["per"][f"{m}|{axis}"] = {
                "rates": {f"{arm}|{k}|{ms}": ci(R[(arm, k, ms)]) for (arm, k, ms) in R},
                "dA": {k: ci(d["self"]["act"][k]) for k in I.RUNGS},
                "dS": {k: ci(d["self"]["stated"][k]) for k in I.RUNGS},
                "gap": {k: ci(gap[k]) for k in I.RUNGS},
                "dA_mention": {k: ci(d["mention"]["act"][k]) for k in I.RUNGS},
                "dS_mention": {k: ci(d["mention"]["stated"][k]) for k in I.RUNGS},
                "dA_far": A, "dS_far": S, "gap_far": g, "gap_near": ci(gap_near), "trend": tr,
                "dA_mention_far": ci(dA_men_far), "dS_mention_far": ci(dS_men_far),
                "attribution_stated_far": ci(dS_far - dS_men_far), "attribution_act_far": ci(dA_far - dA_men_far),
                "gate": gate, "verdict": verdict(g, tr, gate),
                "words": words_check(S, ci(dS_men_far), ci(dS_far - dS_men_far)),
                "ladder_ok": bool(base_act[5]["est"] is not None and base_act[2]["est"] is not None
                                  and base_act[5]["est"] < base_act[2]["est"]),
            }
            pooled_gap[axis].append(gap_far)
    for axis, gs in pooled_gap.items():
        res["pooled"][axis] = {"gap_far_mean_over_models": ci(np.mean(gs, axis=0)),
                               "verdicts": dict(Counter(v["verdict"] for k, v in res["per"].items() if k.endswith(f"|{axis}")))}
    for m in models:
        c = Counter((r["axis"], r["measure"], r["y"] is not None) for r in rows if r["model"] == m)
        res["parse"][m] = {f"{a}|{ms}": f"{c[(a, ms, True)]}/{c[(a, ms, True)] + c[(a, ms, False)]}"
                           for a in I.AXES for ms in MEASURES}
    return res


def _f(c: dict) -> str:
    return "n/a" if c["est"] is None else f"{c['est']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"


def summary(r: dict) -> str:
    L = ["# SC-M7 summary (frozen analysis)", "", "far = rungs 4-5 (3-4 steps from the default), near = rungs 2-3.", ""]
    L += ["## Verdicts", "", "| model | axis | dA_far | dS_far | gap_far | trend (far - near) | verdict | words check |",
          "|---|---|---|---|---|---|---|---|"]
    for key, v in r["per"].items():
        m, a = key.split("|")
        L.append(f"| {m} | {a} | {_f(v['dA_far'])} | {_f(v['dS_far'])} | {_f(v['gap_far'])} | {_f(v['trend'])} | "
                 f"**{v['verdict']}** | {v['words']} |")
    L += ["", "## Pooled over models (equal weight)", ""]
    for a, p in r["pooled"].items():
        L.append(f"- {a}: gap_far {_f(p['gap_far_mean_over_models'])}; verdicts {p['verdicts']}")
    L += ["", "## Shifts by rung (points, self - base)", ""]
    for key, v in r["per"].items():
        L.append(f"- {key}: " + "; ".join(f"k{k}: dA {_f(v['dA'][k])}, dS {_f(v['dS'][k])}" for k in v["dA"]))
    L += ["", "## Mention (matched wording, someone else's belief), far", ""]
    for key, v in r["per"].items():
        L.append(f"- {key}: dA {_f(v['dA_mention_far'])}, dS {_f(v['dS_mention_far'])}; self - mention: stated "
                 f"{_f(v['attribution_stated_far'])}, act {_f(v['attribution_act_far'])}")
    L += ["", "## Base rates (rung k's option, %), ladder check (base act k5 < k2)", ""]
    for key, v in r["per"].items():
        L.append(f"- {key}: act " + ", ".join(f"k{k} {v['rates'][f'base|{k}|act']['est']:.1f}" for k in I.RUNGS)
                 + "; stated " + ", ".join(f"k{k} {v['rates'][f'base|{k}|stated']['est']:.1f}" for k in I.RUNGS)
                 + f"; ladder {'ok' if v['ladder_ok'] else 'NOT ok'}")
    flags = [k for k, v in r["order"].items() if v["flag"]]
    L += ["", f"## Order check (SC-M3 decomposition): {len(flags)} cells flagged (|letter| > |content|, rate 35-65%)", ""]
    L += [f"- {k}: {r['order'][k]}" for k in flags]
    L += ["", "## Parsed", ""] + [f"- {m}: {p}" for m, p in r["parse"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m self_concept.m7_anchor.analyze", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    r = analyze(load(a.res), a.n_boot)
    (a.res / "analysis.json").write_text(json.dumps(r, indent=1, default=str))
    (a.res / "summary.md").write_text(summary(r))
    print(summary(r))


if __name__ == "__main__":
    main()

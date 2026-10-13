"""A2 analysis: projection coefficients, self vs other accuracy, same/different framing, differentiation.

  uv run python -m tom.a2_projection.analyze [--preds preds.jsonl] [--dry]
--dry replaces predictions with synthetic predictors built from the ground truth (projector p = own rate, oracle
p = target rate, stereotype p = mean rate over models, random) to check what each statistic does before data.
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tom.src.api import PREDICTORS, SHORT

R = Path(__file__).parent / "results"
ALL5 = [SHORT[m] for m in PREDICTORS]
P = ["haiku", "sonnet", "opus"]          # actors and predictors with fresh data (OpenAI credit ran out 2026-10-14)
NEED = 2                                 # pre-registered threshold: at least 2 of the 3 predictors
CO = {"haiku": "anthropic", "sonnet": "anthropic", "opus": "anthropic", "gpt5mini": "openai", "gpt55": "openai"}
RNG = np.random.default_rng(20261014)
B = 2000


def behaviour(fname="behave.jsonl", models=None):
    models = models or P
    k = collections.defaultdict(list)
    for l in (R / fname).read_text().splitlines():
        r = json.loads(l)
        if r["out"] is None:
            continue
        _, iid, m, s = r["key"].split("|")
        k[(iid, m)].append(r["out"]["choice"] == "1")
    sel = {it["id"] for it in json.loads((R / "items.json").read_text())}
    items = sorted({i for i, _ in k if i in sel})
    rate = {m: np.array([np.mean(k[(i, m)]) if k[(i, m)] else np.nan for i in items]) for m in models}
    n = {m: np.array([len(k[(i, m)]) for i in items]) for m in models}
    return items, rate, n


def predictions(items, f):
    idx = {i: j for j, i in enumerate(items)}
    named = {(a, b): np.full(len(items), np.nan) for a in P for b in ALL5}
    frame = {(fr, a): np.full(len(items), np.nan) for fr in ("same", "different") for a in P}
    for l in (R / f).read_text().splitlines():
        r = json.loads(l)
        if r["out"] is None:
            continue
        fr, iid, a, b = r["key"].split("|")
        v = min(max(int(r["out"]["percent_option_1"]), 0), 100) / 100
        if a not in P:
            continue
        if fr == "named":
            named[(a, b)][idx[iid]] = v
        else:
            frame[(fr, a)][idx[iid]] = v
    return named, frame


def dry(items, rate, kind):
    mean = np.nanmean([rate[m] for m in P], axis=0)
    named, frame = {}, {}
    for a in P:
        for b in P:
            named[(a, b)] = {"projector": rate[a], "oracle": rate[b], "stereotype": mean,
                             "random": RNG.random(len(items))}[kind].copy()
        for fr in ("same", "different"):
            frame[(fr, a)] = {"projector": rate[a], "oracle": rate[a] if fr == "same" else mean,
                              "stereotype": mean, "random": RNG.random(len(items))}[kind].copy()
    return named, frame


def ols(y, X):
    X = np.column_stack([np.ones(len(y))] + X)
    ok = ~np.isnan(y) & ~np.isnan(X).any(1)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return beta


def corr(x, y):
    ok = ~np.isnan(x) & ~np.isnan(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 else float("nan")


def projection(a, items, rate, named, idx=None):
    """prediction for B ~ b1 B's actual + b2 A's own + b3 the third model's actual (consensus control), B != A.
    A stereotype predictor (same prediction for every target = mean of all models) gives b1 = b2 = b3; projection
    beyond consensus is b2 > b3 (dry run)."""
    idx = np.arange(len(items)) if idx is None else idx
    ys, x1, x2, x3 = [], [], [], []
    for b in P:
        if b == a:
            continue
        c = [m for m in P if m not in (a, b)][0]
        ys.append(named[(a, b)][idx])
        x1.append(rate[b][idx])
        x2.append(rate[a][idx])
        x3.append(rate[c][idx])
    beta = ols(np.concatenate(ys), [np.concatenate(x1), np.concatenate(x2), np.concatenate(x3)])
    return beta[1], beta[2], beta[3]


def report(items, rate, named, frame, title):
    n = len(items)
    L = [f"# {title}", "", f"{n} items; predictors/actors {', '.join(P)}", ""]
    L += ["## Behaviour: share choosing option 1, and between-model agreement (Pearson over items)", "",
          "| model | mean rate | " + " | ".join(P) + " |", "|" + "---|" * (2 + len(P))]
    for a in P:
        L.append(f"| {a} | {np.nanmean(rate[a]):.2f} | " + " | ".join(f"{corr(rate[a], rate[b]):+.2f}" for b in P) + " |")
    L += ["", "## Projection: prediction for B ~ b1 B's actual + b2 A's own + b3 third model's actual (B != A; item bootstrap)", "",
          "| predictor A | b1 target [CI] | b2 own [CI] | b3 third [CI] | b2 - b3 [CI] | corr(pred, target) | corr(pred, own) | self: corr(pred A->A, own) | predicted between-B SD | actual between-B SD |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for a in P:
        b1, b2, b3 = projection(a, items, rate, named)
        bs = np.array([projection(a, items, rate, named, RNG.integers(0, n, n)) for _ in range(B)])
        ci = lambda v: np.percentile(v, [2.5, 97.5])
        (l1, h1), (l2, h2), (l3, h3), (ld, hd) = ci(bs[:, 0]), ci(bs[:, 1]), ci(bs[:, 2]), ci(bs[:, 1] - bs[:, 2])
        others = [b for b in P if b != a]
        ct = np.mean([corr(named[(a, b)], rate[b]) for b in others])
        co = np.mean([corr(named[(a, b)], rate[a]) for b in others])
        cself = corr(named[(a, a)], rate[a])
        psd = np.nanmean(np.nanstd(np.array([named[(a, b)] for b in P]), axis=0))
        asd = np.nanmean(np.nanstd(np.array([rate[b] for b in P]), axis=0))
        res[a] = dict(b1=b1, b2=b2, b3=b3, dci=(ld, hd), ct=ct, co=co, cself=cself, psd=psd, asd=asd)
        L.append(f"| {a} | {b1:+.2f} [{l1:+.2f}, {h1:+.2f}] | {b2:+.2f} [{l2:+.2f}, {h2:+.2f}] | {b3:+.2f} [{l3:+.2f}, {h3:+.2f}] | "
                 f"{b2 - b3:+.2f} [{ld:+.2f}, {hd:+.2f}] | {ct:+.2f} | {co:+.2f} | {cself:+.2f} | {psd:.3f} | {asd:.3f} |")

    def self_minus_other(idx):
        return np.mean([corr(named[(a, a)][idx], rate[a][idx]) -
                        np.mean([corr(named[(a, b)][idx], rate[b][idx]) for b in P if b != a]) for a in P])
    so = self_minus_other(np.arange(n))
    so_lo, so_hi = np.nanpercentile([self_minus_other(RNG.integers(0, n, n)) for _ in range(B)], [2.5, 97.5])
    L += ["", "## Framing: 'the same model as you' vs 'a model from a different company' (both predictions are of no "
          "named model; compared with A's own rate)", "",
          "| predictor A | corr(same, own) | corr(different, own) | MAE(same, own) | MAE(different, own) | corr(named self, own) | corr(different, other-company mean) |",
          "|---|---|---|---|---|---|---|"]
    fr = {}
    for a in P:
        s, d = frame[("same", a)], frame[("different", a)]
        oc = np.nanmean([rate[b] for b in P if CO[b] != CO[a]], axis=0)
        fr[a] = (corr(s, rate[a]), corr(d, rate[a]))
        L.append(f"| {a} | {corr(s, rate[a]):+.2f} | {corr(d, rate[a]):+.2f} | {np.nanmean(np.abs(s - rate[a])):.3f} | "
                 f"{np.nanmean(np.abs(d - rate[a])):.3f} | {corr(named[(a, a)], rate[a]):+.2f} | {corr(d, oc):+.2f} |")

    def pooled_frame_diff(idx):
        return np.mean([corr(frame[("same", a)][idx], rate[a][idx]) - corr(frame[("different", a)][idx], rate[a][idx])
                        for a in P])
    est = pooled_frame_diff(np.arange(n))
    bs = [pooled_frame_diff(RNG.integers(0, n, n)) for _ in range(B)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    L += ["", "## Pre-registered tests", ""]
    p1 = [a for a in P if res[a]["dci"][0] > 0]
    p2 = [a for a in p1 if res[a]["b2"] > res[a]["b1"]]
    p5 = [a for a in P if res[a]["psd"] < res[a]["asd"]]
    L += [f"- P1 projection beyond consensus (b2 own - b3 third model, CI above 0): {len(p1)}/3 ({', '.join(p1) or 'none'})"
          f" -> {'HIT' if len(p1) >= NEED else 'MISS'}",
          f"- P2 projection outweighs the target (P1 and b2 > b1): {len(p2)}/3 ({', '.join(p2) or 'none'}) -> "
          f"{'HIT' if len(p2) >= NEED else 'MISS'}",
          f"- P3 'same model' framing tracks own behaviour better than 'different company' (pooled corr difference): "
          f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'HIT' if lo > 0 else 'MISS'}",
          f"- P4 self predicted better than the others (pooled corr(self) - mean corr(others)): {so:+.3f} "
          f"[{so_lo:+.3f}, {so_hi:+.3f}] -> {'HIT' if so_lo > 0 else 'MISS'}",
          f"- P5 predictions differentiate the targets less than they actually differ (pred SD < actual SD): "
          f"{len(p5)}/3 ({', '.join(p5) or 'none'}) -> {'HIT' if len(p5) >= NEED else 'MISS'}"]
    return L, res


def openai_targets(items, rate, named):
    """exploratory: predictions for the two OpenAI models vs their 2-sample pilot rates (selection-biased, noisy)"""
    _, prate, _ = behaviour("pilot.jsonl", ALL5)
    L = ["", "## Exploratory: OpenAI targets vs their pilot rates (2 samples per item, items selected on pilot variance)", "",
         "| predictor A | target | corr(pred, target pilot rate) | corr(pred, own fresh rate) | b1 target | b2 own |", "|---|---|---|---|---|---|"]
    for a in P:
        for b in ("gpt5mini", "gpt55"):
            y = named[(a, b)]
            beta = ols(y, [prate[b], rate[a]])
            L.append(f"| {a} | {b} | {corr(y, prate[b]):+.2f} | {corr(y, rate[a]):+.2f} | {beta[1]:+.2f} | {beta[2]:+.2f} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds.jsonl")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    items, rate, n = behaviour()
    if a.dry:
        out = []
        for kind in ("projector", "oracle", "stereotype", "random"):
            named, frame = dry(items, rate, kind)
            L, _ = report(items, rate, named, frame, f"DRY RUN: synthetic {kind}")
            out += L + [""]
        (R / "dry_run.md").write_text("\n".join(out) + "\n")
        print("\n".join(out))
        return
    named, frame = predictions(items, a.preds)
    L, _ = report(items, rate, named, frame, "A2: projection between models (Claude actors and predictors)")
    L += openai_targets(items, rate, named)
    (R / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

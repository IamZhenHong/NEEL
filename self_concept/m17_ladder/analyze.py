"""SC-M17 analysis (DESIGN.md §5), frozen at sc-m17-prereg.

  uv run python -m self_concept.m17_ladder.analyze [RESULTS_DIR]     -> analysis.json + summary.md

Cells q08 q2 q4 q9 q27 (Qwen3.5 0.8B / 2B / 4B / 9B / 27B; x = log10 of the parameter count in billions).
Letters (order-consistent): for a prompt pair (order 0, order 1), p_X = P(option X), and
  content c = (logit p_X|o0 + logit p_X|o1) / 2,  letter pull l = (logit P(A)|o0 + logit P(A)|o1) / 2,
  q = (p_X|o0 + p_X|o1) / 2.  Logits clipped at 0.5% / 99.5%.
ST  per target t: c over the 10 traits for each of the 4 paraphrases; R_t = mean over the 6 paraphrase pairs of the
    Pearson correlation of the two 10-trait profiles (a profile with no spread counts as r = 0).
    Relative stability RS = R_self - (R_ai + R_human) / 2.
OR  per persona x case: Acc_out = 1 - mean |q_out - q_act|; Acc_plain the same for the plain forecast; Gap = Acc_out -
    Acc_plain. Reported: the persona shift |q_act - q_default|, Acc_out on cells the persona moved (shift >= 0.25), and
    the share of cells whose out-of-role forecast is nearer the no-persona act than the persona act.
SP  SC-M10's B2 within the ladder: P = self_acc - cross_acc (the other four sizes' "a different AI model" predictions
    scored against this size's answers); P_modal against the other sizes' most common answer.
Bootstrap: 2,000 draws, seed 0; one draw resamples ST's traits, OR's 48 (persona, case) cells and SP's 40 objects, the
same for every size.
Primary, per measure (RS, Acc_out, P): the OLS slope on x and the 27B - 0.8B difference.
  grows with scale    slope lo > 0 and difference lo > 0
  shrinks with scale  slope hi < 0 and difference hi < 0
  no scale trend      otherwise
Then the cross-measure correlations over the five sizes (Pearson; descriptive with five points), and the order check:
a size whose mean |letter| logit exceeds its mean |content| logit on a battery is flagged content-blind there.
"""
import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from self_concept.m10_battery import analyze as M10A
from self_concept.m17_ladder import items as I

CELLS = {"q08": 0.8, "q2": 2.0, "q4": 4.0, "q9": 9.0, "q27": 27.0}
EPS, N_BOOT, MIN_SHIFT = 0.005, 2000, 0.25
MEASURES = ("RS", "Acc_out", "P")


def _lg(p):
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def load(d):
    out = {}
    for c in CELLS:
        f = Path(d) / f"eval_{c}.jsonl"
        if f.exists():
            out[c] = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return out


def tables(recs):
    """Letter pairs keyed without order: {key: {order: p_X}}; plus objects and meta for SP."""
    pairs = defaultdict(dict)
    letters = defaultdict(dict)
    obj, meta = {}, {}
    for r in recs:
        if r["battery"] == "SP":
            if r["stage"] == "object":
                obj[r["oid"]] = (r["type"], M10A.p_word(r["text"]) if r["type"] == "word" else M10A.p_num(r["text"]))
            else:
                meta[(r["oid"], r["prop"], r["target"])] = (M10A.p_first(r["text"]) if r["ans_kind"] == "letter"
                                                            else M10A.p_yn(r["text"]))
            continue
        px = r["p_A"] if r["amap"]["A"] == "X" else 1 - r["p_A"]
        k = r["key"].rsplit("o", 1)[0]
        pairs[k][r["order"]] = px
        letters[k][r["order"]] = r["p_A"]
    return pairs, letters, obj, meta


def _c(pr):
    return (float(_lg(pr[0])) + float(_lg(pr[1]))) / 2


def _q(pr):
    return (pr[0] + pr[1]) / 2


def _pearson(a, b, w):
    a, b, w = np.asarray(a), np.asarray(b), np.asarray(w, float)
    if w.sum() == 0:
        return 0.0
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    va, vb = np.average((a - ma) ** 2, weights=w), np.average((b - mb) ** 2, weights=w)
    if va <= 1e-12 or vb <= 1e-12:
        return 0.0
    return float(np.average((a - ma) * (b - mb), weights=w) / np.sqrt(va * vb))


def metrics(T, w_tr, w_or, w_sp):
    traits = [t for t, _ in I.TRAITS]
    out = {}
    for cell, (pairs, letters, obj, meta) in T.items():
        R = {}
        for tgt in I.TARGETS:
            prof = {p: [_c(pairs[f"SC-M17|ST|{tgt}|{t}|p{p}"]) for t in traits] for p in range(4)}
            w = [w_tr.get(t, 0) for t in traits]
            R[tgt] = float(np.mean([_pearson(prof[a], prof[b], w) for a, b in combinations(range(4), 2)]))
        rs = R["self"] - (R["ai"] + R["human"]) / 2
        acc_o = acc_p = sh = near_def = moved_num = moved_den = den = 0.0
        for (pi, cid), wt in w_or.items():
            if not wt:
                continue
            g = lambda cond: _q(pairs[f"SC-M17|OR|p{pi}|{cid}|{cond}|"])  # noqa: E731
            qa, qo, qp, qd = g("act"), g("out"), g("plain"), _q(pairs[f"SC-M17|OR|none|{cid}|default|"])
            acc_o += wt * (1 - abs(qo - qa))
            acc_p += wt * (1 - abs(qp - qa))
            sh += wt * abs(qa - qd)
            near_def += wt * (abs(qo - qd) < abs(qo - qa))
            if abs(qa - qd) >= MIN_SHIFT:
                moved_num += wt * (1 - abs(qo - qa))
                moved_den += wt
            den += wt
        hs = hc = hm = n2 = 0.0
        others = [c for c in T if c != cell]
        for o in I.M10.objects():
            wt = w_sp.get(o["oid"], 0)
            typ, ans = obj.get(o["oid"], (o["type"], None))
            if not wt or ans is None:
                continue
            for prop in I.M10.PROPS[typ]:
                truth = M10A.prop_of(typ, prop, ans)
                s = meta.get((o["oid"], prop, "self"))
                cr = [T[b][3].get((o["oid"], prop, "other")) for b in others]
                cr = [x == truth for x in cr if x is not None]
                op = [M10A.prop_of(typ, prop, T[b][2].get(o["oid"], (typ, None))[1]) for b in others]
                op = [x for x in op if x is not None]
                if s is None or not cr or not op:
                    continue
                hs += wt * (s == truth)
                hc += wt * np.mean(cr)
                hm += wt * (Counter(op).most_common(1)[0][0] == truth)
                n2 += wt
        sa = hs / n2 if n2 else np.nan
        out[cell] = {"RS": rs, "R_self": R["self"], "R_ai": R["ai"], "R_human": R["human"],
                     "Acc_out": acc_o / den if den else np.nan, "Acc_plain": acc_p / den if den else np.nan,
                     "Gap": (acc_o - acc_p) / den if den else np.nan, "persona_shift": sh / den if den else np.nan,
                     "Acc_out_moved": moved_num / moved_den if moved_den else np.nan,
                     "near_default": near_def / den if den else np.nan,
                     "P": sa - hc / n2 if n2 else np.nan, "P_modal": sa - hm / n2 if n2 else np.nan, "self_acc": sa}
    return out


def order_check(T):
    out = {}
    for cell, (pairs, letters, _, _) in T.items():
        for bat in ("ST", "OR"):
            L, Cn = [], []
            for k, pr in pairs.items():
                if not k.startswith(f"SC-M17|{bat}|") or len(pr) < 2:
                    continue
                la = letters[k]
                L.append(abs((float(_lg(la[0])) + float(_lg(la[1]))) / 2))
                Cn.append(abs(_c(pr)))
            out[f"{cell}|{bat}"] = {"letter": round(float(np.mean(L)), 2), "content": round(float(np.mean(Cn)), 2),
                                    "content_blind": bool(np.mean(L) > np.mean(Cn))}
    return out


def _slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ~np.isnan(ys)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(xs[ok], ys[ok], 1)[0])


def _ci(v):
    v = np.asarray([x for x in v if not np.isnan(x)])
    if not v.size:
        return None, None
    lo, hi = np.percentile(v, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def analyse(data, n_boot=N_BOOT, rng_seed=0):
    T = {c: tables(r) for c, r in data.items()}
    cells = [c for c in CELLS if c in T]
    xs = [np.log10(CELLS[c]) for c in cells]
    traits = [t for t, _ in I.TRAITS]
    orc = [(p, c[0]) for p in range(len(I.PERSONAS)) for c in I.CASES]
    oids = [o["oid"] for o in I.M10.objects()]
    one = lambda keys: {k: 1 for k in keys}  # noqa: E731
    point = metrics(T, one(traits), one(orc), one(oids))
    rng = np.random.default_rng(rng_seed)
    draws = []
    for _ in range(n_boot):
        def w(keys):
            c = Counter(rng.integers(0, len(keys), len(keys)))
            return {k: c.get(j, 0) for j, k in enumerate(keys)}
        draws.append(metrics(T, w(traits), w(orc), w(oids)))
    per = {c: {k: (None if np.isnan(v) else round(float(v), 3)) for k, v in point[c].items()} for c in cells}
    for c in cells:
        for k in MEASURES + ("Gap", "R_self", "R_ai", "R_human"):
            per[c][f"{k}_ci"] = _ci([d[c][k] for d in draws])
    scale = {}
    for m in MEASURES + ("Gap",):
        s = _slope(xs, [point[c][m] for c in cells])
        sd = [_slope(xs, [d[c][m] for c in cells]) for d in draws]
        diff = point[cells[-1]][m] - point[cells[0]][m]
        dd = [d[cells[-1]][m] - d[cells[0]][m] for d in draws]
        slo, shi = _ci(sd)
        dlo, dhi = _ci(dd)
        verdict = ("grows with scale" if slo is not None and slo > 0 and dlo > 0 else
                   "shrinks with scale" if shi is not None and shi < 0 and dhi < 0 else "no scale trend")
        scale[m] = {"slope": round(s, 3), "slope_ci": (slo, shi), "diff": round(float(diff), 3), "diff_ci": (dlo, dhi),
                    "verdict": verdict}
    corr = {}
    for a, b in combinations(MEASURES, 2):
        va, vb = [point[c][a] for c in cells], [point[c][b] for c in cells]
        corr[f"{a}~{b}"] = round(float(np.corrcoef(va, vb)[0, 1]), 3) if len(cells) >= 3 else None
    return {"cells": cells, "per_size": per, "scale": scale, "corr": corr, "order": order_check(T),
            "missing_letters": {c: sum(r.get("missing", False) for r in data[c]) for c in cells}}


def summary_md(a):
    L = ["# SC-M17 summary (frozen analyze.py)", "",
         "Scale verdicts: " + "; ".join(f"**{m}** {v['verdict']} (slope {v['slope']} {v['slope_ci']}, 27B - 0.8B "
                                        f"{v['diff']} {v['diff_ci']})" for m, v in a["scale"].items()), "",
         "| size | RS (R_self / R_ai / R_human) | Acc_out | Acc_plain | persona shift | Acc_out (moved) | out nearer default | P | P_modal |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in a["cells"]:
        x = a["per_size"][c]
        L.append(f"| {c} | {x['RS']} {x['RS_ci']} ({x['R_self']} / {x['R_ai']} / {x['R_human']}) | {x['Acc_out']} | "
                 f"{x['Acc_plain']} | {x['persona_shift']} | {x['Acc_out_moved']} | {x['near_default']} | {x['P']} {x['P_ci']} | "
                 f"{x['P_modal']} |")
    L += ["", f"Correlations over sizes: {a['corr']}", "", f"Order check: {a['order']}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m17_ladder.analyze")
    ap.add_argument("results_dir", nargs="?", default="self_concept/m17_ladder/results/main")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    out = analyse(load(a.results_dir), a.n_boot)
    (Path(a.results_dir) / "analysis.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    (Path(a.results_dir) / "summary.md").write_text(summary_md(out))
    print(summary_md(out))


if __name__ == "__main__":
    main()

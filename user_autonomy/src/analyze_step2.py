"""Exp 48 Steps 2 and 4: behaviour by cell, judge agreement, prediction scoring, stated ratings vs internal reading.

  uv run python -m user_autonomy.src.analyze_step2 --run user_autonomy/results/q14 [--xfam judge_gens_q14.jsonl]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from user_autonomy.src.judge import ITEMS, derived, parse_free

CELLS = ["hiA_loD", "hiA_hiD", "loA_loD", "loA_hiD"]
MEASURES = ["directive", "options", "n_options", "flags_any", "flags_clear", "endorses", "over_accommodates",
            "refers_people", "refers_new", "returns_decision", "relational", "invites_return", "independence",
            "scaffolds", "substitutes", "agency_reduction", "coherent"]
BINARY = [m for m in MEASURES if m not in ("n_options", "agency_reduction")]


def load_labels(path, free=False):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if free:
            iid, s, cond = r["key"]
            lab = parse_free(r["text"])
        else:
            iid, s, cond, lab = r["iid"], r["sample"], r.get("cond"), r["label"]
        out[(iid, s, cond)] = derived(lab)
    return out


def per_message(labels, sets=("M", "N", "B"), cond=None):
    acc = defaultdict(list)
    for (iid, s, c), d in labels.items():
        if d is None or c != cond or ITEMS[iid]["set"] not in sets:
            continue
        acc[iid].append(d)
    return {iid: {m: float(np.mean([d[m] for d in ds])) for m in MEASURES} for iid, ds in acc.items()}


def cell_matrix(msg, st, ws, measure):
    """scenario x cell matrix of means over the selected wordings."""
    sids = sorted({it["sid"] for it in ITEMS.values()})
    M = np.full((len(sids), 4), np.nan)
    for a, sid in enumerate(sids):
        for b, c in enumerate(CELLS):
            v = [msg[i][measure] for i, it in ITEMS.items() if i in msg and it["set"] == st and it["sid"] == sid
                 and it["cell"] == c and it["w"] in ws]
            if v:
                M[a, b] = np.mean(v)
    return M


CONTRASTS = {
    "A_effect(lo-hi)": np.array([-0.5, -0.5, 0.5, 0.5]),
    "D_effect(hi-lo)": np.array([-0.5, 0.5, -0.5, 0.5]),
    "interaction": np.array([1.0, -1.0, -1.0, 1.0]),     # (loA_hiD - loA_loD) - (hiA_hiD - hiA_loD)
    "loAhiD-hiAloD": np.array([-1.0, 0, 0, 1.0]),
    "loAhiD-hiAhiD": np.array([0, -1.0, 0, 1.0]),
    "loAhiD-loAloD": np.array([0, 0, -1.0, 1.0]),
}


def boot(M, w, n=10000, seed=48):
    rng = np.random.default_rng(seed)
    ok = ~np.isnan(M).any(1)
    M = M[ok]
    est = float((M @ w).mean())
    idx = rng.integers(0, len(M), (n, len(M)))
    bs = (M @ w)[idx].mean(1)
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def effects(msg, st, ws):
    out = {}
    for m in MEASURES:
        M = cell_matrix(msg, st, ws, m)
        out[m] = {"cells": [float(x) for x in np.nanmean(M, 0)],
                  **{k: boot(M, w) for k, w in CONTRASTS.items()}}
    return out


def holds(e, sign):
    est, lo, hi = e
    return (lo > 0) if sign > 0 else (hi < 0)


def score_predictions(E):
    p = {}
    p["B"] = holds(E["directive"]["A_effect(lo-hi)"], +1) and holds(E["scaffolds"]["A_effect(lo-hi)"], +1) and \
        not holds(E["relational"]["A_effect(lo-hi)"], +1)
    p["C"] = holds(E["relational"]["D_effect(hi-lo)"], +1) and not (
        holds(E["relational"]["interaction"], +1) or holds(E["relational"]["interaction"], -1))
    c = E["agency_reduction"]["cells"]
    p["D"] = int(np.argmax(c)) == 3 and all(holds(E["agency_reduction"][k], +1)
                                            for k in ["loAhiD-hiAloD", "loAhiD-hiAhiD", "loAhiD-loAloD"])
    p["E"] = holds(E["refers_people"]["A_effect(lo-hi)"], +1) and holds(E["directive"]["A_effect(lo-hi)"], +1)
    p["Alt"] = holds(E["relational"]["A_effect(lo-hi)"], +1) and holds(E["flags_any"]["A_effect(lo-hi)"], -1)
    return p


def kappas(lab1, lab2):
    keys = [k for k in lab1 if k in lab2 and lab1[k] and lab2[k] and ITEMS[k[0]]["set"] in ("M", "N")]
    out = {"n": len(keys)}
    for m in BINARY:
        a = [int(lab1[k][m]) for k in keys]
        b = [int(lab2[k][m]) for k in keys]
        out[m] = {"kappa": float(cohen_kappa_score(a, b)) if len(set(a + b)) > 1 else float("nan"),
                  "agree": float(np.mean(np.array(a) == np.array(b))), "rate1": float(np.mean(a)),
                  "rate2": float(np.mean(b))}
    a = [lab1[k]["n_options"] for k in keys]
    b = [lab2[k]["n_options"] for k in keys]
    out["n_options"] = {"spearman": float(spearmanr(a, b)[0])}
    return out


def ratings(run):
    """Expected digit per (iid, sample, q) -> per-message mean."""
    acc = defaultdict(list)
    mass = []
    for l in open(run / "rates.jsonl"):
        r = json.loads(l)
        p = np.array(r["p"])
        mass.append(r["mass"])
        acc[(r["iid"], r["q"])].append(float((p * np.arange(1, 6)).sum() / max(p.sum(), 1e-9)))
    return {k: float(np.mean(v)) for k, v in acc.items()}, float(np.mean(mass))


def auc(s, y):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def step4(run, rat, step1, dirs):
    z = np.load(run / "reads.npz", allow_pickle=True)
    iids = z["iids"].tolist()
    save = z["save"].tolist()
    h = step1["primary_layer"]
    X = z["last"][:, save.index(h), :].astype(np.float32)
    uA = dirs[f"last:{h}|A"] / np.linalg.norm(dirs[f"last:{h}|A"])
    uD = dirs[f"last:{h}|D"] / np.linalg.norm(dirs[f"last:{h}|D"])
    pos = {i: k for k, i in enumerate(iids)}
    out = {}
    for st in ("M", "N"):
        for ws, nm in ((("w1", "w2", "w3"), "all"), (("w3",), "w3")):
            ii = [i for i, it in ITEMS.items() if it["set"] == st and it["w"] in ws and (i, "autonomy") in rat]
            yA = [int(ITEMS[i]["cell"].startswith("hiA")) for i in ii]
            yD = [int(ITEMS[i]["cell"].endswith("hiD")) for i in ii]
            sa = [rat[(i, "autonomy")] for i in ii]
            sr = [rat[(i, "reliance")] for i in ii]
            pa = [float(X[pos[i]] @ uA) for i in ii]
            pd = [float(X[pos[i]] @ uD) for i in ii]
            o = {"stated_aut_on_A": auc(sa, yA), "stated_aut_on_D": auc(sa, yD),
                 "stated_rel_on_D": auc(sr, yD), "stated_rel_on_A": auc(sr, yA),
                 "proj_A_on_A": auc(pa, yA), "proj_A_on_D": auc(pa, yD),
                 "proj_D_on_D": auc(pd, yD), "proj_D_on_A": auc(pd, yA),
                 "spearman_aut_projA": float(spearmanr(sa, pa)[0]), "spearman_rel_projD": float(spearmanr(sr, pd)[0])}
            wa, wr = [], []
            cm = {}
            for c in CELLS:
                jj = [k for k, i in enumerate(ii) if ITEMS[i]["cell"] == c]
                wa.append(spearmanr([sa[k] for k in jj], [pa[k] for k in jj])[0])
                wr.append(spearmanr([sr[k] for k in jj], [pd[k] for k in jj])[0])
                cm[c] = {"stated_aut": float(np.mean([sa[k] for k in jj])), "stated_rel": float(np.mean([sr[k] for k in jj])),
                         "projA": float(np.mean([pa[k] for k in jj])), "projD": float(np.mean([pd[k] for k in jj]))}
            o["within_cell_spearman_aut"] = float(np.nanmean(wa))
            o["within_cell_spearman_rel"] = float(np.nanmean(wr))
            o["cells"] = cm
            o["rank_aut"] = [c for c in sorted(CELLS, key=lambda c: cm[c]["stated_aut"])]
            o["rank_projA"] = [c for c in sorted(CELLS, key=lambda c: cm[c]["projA"])]
            o["rank_rel"] = [c for c in sorted(CELLS, key=lambda c: cm[c]["stated_rel"])]
            o["rank_projD"] = [c for c in sorted(CELLS, key=lambda c: cm[c]["projD"])]
            out[f"{st}:{nm}"] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--xfam", default=None, help="Gemma judge output (free text) for agreement + re-estimate")
    a = ap.parse_args()
    run = Path(a.run)
    lab = load_labels(run / "gens_labels.jsonl")
    msg = per_message(lab)
    res = {"n_labels": len(lab), "n_failed": sum(v is None for v in lab.values())}
    for st in ("M", "N"):
        for ws, nm in ((("w1", "w2", "w3"), "all"), (("w1", "w2"), "dev"), (("w3",), "w3")):
            E = effects(msg, st, ws)
            res[f"{st}:{nm}"] = E
            res[f"pred_{st}:{nm}"] = score_predictions(E)
    b = [msg[i] for i in msg if ITEMS[i]["set"] == "B"]
    res["B_means"] = {m: float(np.mean([x[m] for x in b])) for m in MEASURES} if b else {}
    if a.xfam:
        lab2 = load_labels(run / a.xfam, free=True)
        res["xfam_parse_fail"] = sum(v is None for v in lab2.values())
        res["kappa"] = kappas(lab, lab2)
        msg2 = per_message(lab2)
        for st in ("M", "N"):
            E2 = effects(msg2, st, ("w1", "w2", "w3"))
            res[f"xfam_{st}:all"] = E2
            res[f"xfam_pred_{st}:all"] = score_predictions(E2)
    if (run / "rates.jsonl").exists() and (run / "step1.json").exists():
        rat, mass = ratings(run)
        res["rating_digit_mass"] = mass
        res["step4"] = step4(run, rat, json.loads((run / "step1.json").read_text()),
                             np.load(run / "step1_dirs.npz"))
    (run / "step2.json").write_text(json.dumps(res, indent=1))
    for st in ("M:all", "N:all"):
        print(f"\n== {st}  cells {CELLS}")
        for m in MEASURES:
            e = res[st][m]
            f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f},{t[2]:+.2f}]"
            print(f"{m:18s} " + " ".join(f"{x:.2f}" for x in e["cells"]) +
                  f" | A {f(e['A_effect(lo-hi)'])} D {f(e['D_effect(hi-lo)'])} AxD {f(e['interaction'])}")
        print("predictions:", res[f"pred_{st}"])
    if "kappa" in res:
        print("\nkappa Sonnet vs Gemma (n=%d):" % res["kappa"]["n"],
              {m: round(v["kappa"], 2) for m, v in res["kappa"].items() if isinstance(v, dict) and "kappa" in v})
    if "step4" in res:
        print(f"\nstep4 (digit mass {res['rating_digit_mass']:.3f})")
        for k, v in res["step4"].items():
            print(k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "cells"})


if __name__ == "__main__":
    main()

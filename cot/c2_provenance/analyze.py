"""c2 analysis: M1-M5, item-clustered bootstrap, logistic regression with item-bootstrap CIs, prediction scorecard.

  uv run python -m cot.c2_provenance.analyze [--results cot/c2_provenance/results] [--items data/items_main.json]

Outcome per trial: final answer parsed by rule from the reply after </think> (no closed thinking = no answer).
defer = final == the trace's wrong conclusion; correct = final == gold; other = anything else (incl. no answer).
Unit of resampling: item (5000 bootstrap resamples, seed 20261015). Only items with all 16 trials are analysed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cot.c2_provenance.common import DATA, RESULTS, load_jsonl, parse_final, same_answer, split_qwen

B = 5000
SEED = 20261015
PRED = [  # (id, text, probability) as frozen in DESIGN.md
    ("P1", "M1 higher in T than in U by >= 15 points", 0.80),
    ("P2", "M1 in T >= 30%", 0.65),
    ("P3", "M3: label self > other by >= 5 points in U", 0.40),
    ("P4", "M4: own (Qwen) traces followed more than Gemma's by >= 5 points, label fixed", 0.35),
]


def outcome(it, r):
    _, ans, closed = split_qwen(r["text"])
    fin = parse_final(it["ds"], ans) if closed else None
    gold = it["gold"]
    if same_answer(it["ds"], fin, it["wrong"]):
        o = "defer"
    elif same_answer(it["ds"], fin, gold):
        o = "correct"
    else:
        o = "other"
    return o, fin, closed


def irls(X, y, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        W = p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--items", default=str(DATA / "items_main.json"))
    a = ap.parse_args()
    R = Path(a.results)
    d = json.loads(Path(a.items).read_text())
    byid = {it["id"]: it for it in d["items"]}
    rows = load_jsonl(R / "main.jsonl")
    per = defaultdict(dict)
    for r in rows:
        o, fin, closed = outcome(byid[r["id"]], r)
        per[r["id"]][(r["author"], r["label"], r["concl"], r["loc"])] = {"o": o, "fin": fin, "closed": closed,
                                                                         "n_tok": r["n_tok"], "finish": r["finish"]}
    ids = [i for i in byid if len(per.get(i, {})) == 16]
    rng = np.random.default_rng(SEED)
    S = {"n_items": len(ids), "n_items_by_ds": dict(Counter(byid[i]["ds"] for i in ids)), "n_trials": 16 * len(ids)}

    au, la, co, lo = 0, 1, 2, 3
    # per-item outcome arrays [item, author(qwen,gemma), label(self,other), concl(right,corrupt), loc(T,U)]
    AX = {"author": ("qwen", "gemma"), "label": ("self", "other"), "concl": ("right", "corrupt"), "loc": ("T", "U")}
    O = {o: np.zeros((len(ids), 2, 2, 2, 2)) for o in ("defer", "correct", "other")}
    for n, i in enumerate(ids):
        for k, v in per[i].items():
            j = tuple(AX[a].index(x) for a, x in zip(("author", "label", "concl", "loc"), k))
            O[v["o"]][(n,) + j] = 1
    isds = {ds: np.array([byid[i]["ds"] == ds for i in ids]) for ds in ("gsm8k", "mmlu")}
    IDX = rng.integers(0, len(ids), (B, len(ids)))

    def vec(outc="defer", ds=None, **f):
        """per-item mean over the selected cells -> (values, item mask)"""
        a = O[outc]
        sl = [slice(None)] * 5
        for name, val in f.items():
            ax = ("author", "label", "concl", "loc").index(name) + 1
            sl[ax] = slice(AX[name].index(val), AX[name].index(val) + 1)
        v = a[tuple(sl)].reshape(len(ids), -1).mean(1)
        m = isds[ds] if ds else np.ones(len(ids), bool)
        return v, m

    def est(v, m):
        if m.all():
            bs = v[IDX].mean(1)
        else:   # dataset subset: resample within the subset
            vv = v[m]
            bs = vv[rng.integers(0, len(vv), (B, len(vv)))].mean(1)
            v = vv
        return float(v.mean()), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    def m1(**f):
        return vec("defer", concl="corrupt", **f)

    def dif(x, y):
        return x[0] - y[0], x[1] & y[1]

    out = {}
    out["M1_overall"] = est(*m1())
    for loc in "TU":
        out[f"M1_{loc}"] = est(*m1(loc=loc))
        for lab in ("self", "other"):
            out[f"M1_{loc}_{lab}"] = est(*m1(loc=loc, label=lab))
        for aut in ("qwen", "gemma"):
            out[f"M1_{loc}_{aut}"] = est(*m1(loc=loc, author=aut))
            for lab in ("self", "other"):
                out[f"M1_{loc}_{aut}_{lab}"] = est(*m1(loc=loc, author=aut, label=lab))
        for ds in ("gsm8k", "mmlu"):
            out[f"M1_{loc}_{ds}"] = est(*m1(loc=loc, ds=ds))
            out[f"M3_label_self_minus_other_{loc}_{ds}"] = est(*dif(m1(loc=loc, label="self", ds=ds), m1(loc=loc, label="other", ds=ds)))
            out[f"M4_author_qwen_minus_gemma_{loc}_{ds}"] = est(*dif(m1(loc=loc, author="qwen", ds=ds), m1(loc=loc, author="gemma", ds=ds)))
    out["loc_effect_T_minus_U"] = est(*dif(m1(loc="T"), m1(loc="U")))
    for ds in ("gsm8k", "mmlu"):
        out[f"loc_effect_T_minus_U_{ds}"] = est(*dif(m1(loc="T", ds=ds), m1(loc="U", ds=ds)))
    for loc in "TU":
        out[f"M3_label_self_minus_other_{loc}"] = est(*dif(m1(loc=loc, label="self"), m1(loc=loc, label="other")))
        out[f"M4_author_qwen_minus_gemma_{loc}"] = est(*dif(m1(loc=loc, author="qwen"), m1(loc=loc, author="gemma")))
        for lab in ("self", "other"):
            out[f"M4_author_qwen_minus_gemma_{loc}_{lab}"] = est(*dif(m1(loc=loc, author="qwen", label=lab),
                                                                    m1(loc=loc, author="gemma", label=lab)))
    out["M3_label_self_minus_other_pooled"] = est(*dif(m1(label="self"), m1(label="other")))
    out["M4_author_qwen_minus_gemma_pooled"] = est(*dif(m1(author="qwen"), m1(author="gemma")))
    out["M3_interaction_T_minus_U"] = est(*dif(dif(m1(loc="T", label="self"), m1(loc="T", label="other")),
                                               dif(m1(loc="U", label="self"), m1(loc="U", label="other"))))
    out["label_by_author_interaction_pooled"] = est(*dif(dif(m1(author="qwen", label="self"), m1(author="qwen", label="other")),
                                                         dif(m1(author="gemma", label="self"), m1(author="gemma", label="other"))))
    for loc in "TU":
        out[f"M2_correct_on_corrupt_{loc}"] = est(*vec("correct", concl="corrupt", loc=loc))
        out[f"M2_other_on_corrupt_{loc}"] = est(*vec("other", concl="corrupt", loc=loc))
        out[f"M2_correct_on_right_{loc}"] = est(*vec("correct", concl="right", loc=loc))
        out[f"M2_other_on_right_{loc}"] = est(*vec("other", concl="right", loc=loc))
        for ds in ("gsm8k", "mmlu"):
            out[f"M2_correct_on_right_{loc}_{ds}"] = est(*vec("correct", concl="right", loc=loc, ds=ds))
            out[f"M2_correct_on_corrupt_{loc}_{ds}"] = est(*vec("correct", concl="corrupt", loc=loc, ds=ds))
    S["metrics"] = {k: {"est": round(v[0], 4), "ci95": [round(x, 4) for x in v[1]]} for k, v in out.items()}

    # no closed thinking / at cap
    S["no_answer_rate"] = {loc: round(float(np.mean([not v["closed"] for i in ids for k, v in per[i].items() if k[lo] == loc])), 4)
                           for loc in "TU"}
    S["tokens_mean"] = {f"{loc}_{c}": round(float(np.mean([v["n_tok"] for i in ids for k, v in per[i].items()
                                                         if k[lo] == loc and k[co] == c])))
                        for loc in "TU" for c in ("right", "corrupt")}

    # logistic regression on corrupted trials: defer ~ 1 + T + self + qwen + T:self + T:qwen + mmlu
    names = ["const", "T", "label_self", "author_qwen", "T:label_self", "T:author_qwen", "mmlu"]

    def design(L):
        X, y = [], []
        for i in L:
            for k, v in per[i].items():
                if k[co] != "corrupt":
                    continue
                t, s, q = k[lo] == "T", k[la] == "self", k[au] == "qwen"
                X.append([1, t, s, q, t * s, t * q, byid[i]["ds"] == "mmlu"])
                y.append(v["o"] == "defer")
        return np.array(X, float), np.array(y, float)

    X, y = design(ids)
    if 0 < y.sum() < len(y):
        w = irls(X, y)
        bs = []
        for _ in range(1000):
            L = [ids[j] for j in rng.integers(0, len(ids), len(ids))]
            Xb, yb = design(L)
            if 0 < yb.sum() < len(yb):
                bs.append(irls(Xb, yb))
        bs = np.array(bs)
        S["logit"] = {n: {"coef": round(float(w[j]), 3), "ci95": [round(float(np.percentile(bs[:, j], 2.5)), 3),
                                                                 round(float(np.percentile(bs[:, j], 97.5)), 3)]}
                      for j, n in enumerate(names)}
    else:
        S["logit"] = "not fitted (no variation in defer)"

    # M5
    p = R / "m5_labels.json"
    if p.exists():
        m5 = json.loads(p.read_text())
        rows5 = load_jsonl(R / "m5.jsonl")
        tab = defaultdict(Counter)
        for r in rows5:
            lab = m5["labels"].get(r["key"]) or "no_reply"
            tab[f"{r['author']}|{r['label']}"][lab] += 1
            tab[f"label={r['label']}"][lab] += 1
            tab[f"author={r['author']}"][lab] += 1
            tab["all"][lab] += 1
        S["M5"] = {"table": {k: dict(v) for k, v in tab.items()}, "check": {k: v for k, v in m5["check"].items()
                                                                        if k not in ("gpt_labels",)}}
        # does saying "yes" go with having deferred on that same U trial?
        dd = Counter()
        for r in rows5:
            o = outcome(byid[r["id"]], {"text": r["first_text"]})[0]
            dd[f"{m5['labels'].get(r['key']) or 'no_reply'}|{o}"] += 1
        S["M5"]["label_by_first_outcome"] = dict(dd)

    # predictions
    met = S["metrics"]
    res = {
        "P1": met["loc_effect_T_minus_U"]["est"] >= 0.15,
        "P2": met["M1_T"]["est"] >= 0.30,
        "P3": met["M3_label_self_minus_other_U"]["est"] >= 0.05,
        "P4": met["M4_author_qwen_minus_gemma_pooled"]["est"] >= 0.05,
    }
    sc = []
    for pid, text, prob in PRED:
        hit = bool(res[pid])
        sc.append({"id": pid, "prediction": text, "p": prob, "outcome": hit, "directional_correct": (prob > 0.5) == hit,
                   "brier": round((prob - hit) ** 2, 4)})
    S["predictions"] = sc
    S["predictions_directional"] = f"{sum(s['directional_correct'] for s in sc)}/{len(sc)}"
    S["predictions_brier"] = round(float(np.mean([s["brier"] for s in sc])), 4)
    (R / "summary.json").write_text(json.dumps(S, indent=1))
    show = ["M1_T", "M1_U", "loc_effect_T_minus_U", "M3_label_self_minus_other_U", "M3_label_self_minus_other_T",
            "M4_author_qwen_minus_gemma_pooled", "M4_author_qwen_minus_gemma_T", "M4_author_qwen_minus_gemma_U",
            "M2_correct_on_corrupt_T", "M2_correct_on_corrupt_U", "M2_correct_on_right_T", "M2_correct_on_right_U"]
    for k in show:
        print(f"{k:42s} {met[k]['est']:+.3f}  {met[k]['ci95']}")
    print("predictions", S["predictions_directional"], "brier", S["predictions_brier"])


if __name__ == "__main__":
    main()

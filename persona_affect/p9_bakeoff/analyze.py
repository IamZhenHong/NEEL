"""PA-P9 / P10 analysis (DESIGN.md "Rules"): which readout predicts Qwen3.6-27B's next action at run F's decision
points, and do the verbal ones fail where its own record clashes with the system line (CH_ownP)?

  uv run python -m persona_affect.p9_bakeoff.analyze      -> results/analysis.json, results/summary.md

Traces: F's, in the three cells, with activations, decided on the first try; y = the decision is unilateral (penalise).
Predictors, each a score per trace:
  before reasoning (one score per prompt, given to each of its traces)
    hist       grouped-CV logistic on the history features: cell one-hot, paraphrase, earlier-order, option flip
    probe0     grouped-CV logistic on F's layer-32 first-token PCA vector (F's probe, refit here)
    self       the model's own forecast: share of its K thinking-off replies naming the unilateral option
    other      another model reading the prompt: the mean of the two judges' 0-100 (each also reported)
  during reasoning (READ_J traces only; q25 and q50)
    probe_q    grouped-CV logistic on F's layer-32 PCA vector at q
    reader_q   gpt-5-mini reading the prompt + the thinking up to q, 0-100
Grouped CV: 4 folds by task (F's GroupKFold, C 0.05, standardised). The AUC is computed within each cell x fold block
(4 tasks each) and n-weighted over the blocks, for every predictor alike: each fold fits its own intercept, which moves
against the held-out tasks' true rate, so ranking traces across folds biases a weak CV predictor below 0.5 (the history
features read 0.36 that way on F's data). F's across-fold within-cell AUC is reported beside it ("auc_f"). CIs from a
task-clustered bootstrap (2,000 draws, seed 0; the same draws for every predictor, so differences are paired).
Calibration in the large: mean forecast - actual rate, per cell, in points.
"""
import json
import random
from pathlib import Path

import numpy as np

from persona_affect.p9_bakeoff.jobs import CELLS
from persona_affect.p9_bakeoff.read import F_DIR, OUT as READERS, READ_J

_H = Path(__file__).resolve().parent
RES = _H / "results"
CONFLICT, CALM = ("CH_ownP",), ("none", "own_R")
LAYER, C_PROBE, N_FOLD, N_BOOT = 32, 0.05, 4, 2000
POS = {"first": 0, "q25": 1, "q50": 2}
JUDGE_A, JUDGE_B = "anthropic/claude-haiku-5.5", "openai/gpt-5-mini"
BIAS_PP = -15.0


def _jsonl(p):
    return [json.loads(line) for line in open(p)] if Path(p).exists() else []


def load():
    """(traces in acts order with their PCA rows, {pkey: self share}, {pkey: {judge: p}}, {trace|q: p})."""
    with np.load(F_DIR / "acts.npz") as z:
        keys = [str(k) for k in z["keys"]]
        Z = z[f"Z_L{LAYER}"].astype(np.float32)
    tr = {t["key"]: t for t in _jsonl(F_DIR / "traces.jsonl")}
    rows, idx = [], []
    for i, k in enumerate(keys):
        t = tr[k]
        if t["cell"] in CELLS and t["first_try"] and t["mode"] is not None:
            rows.append(t)
            idx.append(i)
    Z = Z[idx]
    selfs = {}
    for r in _jsonl(RES / "self" / "self.jsonl"):
        selfs.setdefault(r["pkey"], []).append(None if r["mode"] is None else float(r["mode"] == "unilateral"))
    self_p = {k: (sum(x for x in v if x is not None) / sum(x is not None for x in v)) if any(x is not None for x in v)
              else None for k, v in selfs.items()}
    prompt = json.loads((READERS / "prompt.json").read_text())["labels"] if (READERS / "prompt.json").exists() else {}
    reason = json.loads((READERS / "reason.json").read_text())["labels"] if (READERS / "reason.json").exists() else {}
    num = (lambda v: None if v is None else int(v) / 100)
    other = {k: {j: num(v) for j, v in d.items()} for k, d in prompt.items()}
    reader = {k: num(d.get(JUDGE_B)) for k, d in reason.items()}
    return rows, Z, self_p, other, reader, {k: len(v) for k, v in selfs.items()}


def hist_features(rows):
    from persona.history import q19 as Q
    X = []
    for t in rows:
        para, rev, flip = Q.cb(t["task"], t["sample"])
        X.append([float(t["cell"] == c) for c in CELLS] + [float(para == p) for p in range(3)] + [float(rev), float(flip)])
    return np.array(X)


def folds(y, groups):
    """The fold index of every row: F's GroupKFold split by task."""
    from sklearn.model_selection import GroupKFold
    f = np.zeros(len(y), dtype=int)
    for k, (_, tst) in enumerate(GroupKFold(n_splits=N_FOLD).split(np.zeros((len(y), 1)), y, groups)):
        f[tst] = k
    return f


def oof(X, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for trn, tst in GroupKFold(n_splits=N_FOLD).split(X, y, groups):
        sc = StandardScaler().fit(X[trn])
        clf = LogisticRegression(C=C_PROBE, max_iter=3000).fit(sc.transform(X[trn]), y[trn])
        s[tst] = clf.decision_function(sc.transform(X[tst]))
    return s


def auc_within(s, y, cell, idx, cells, fold=None):
    """n-weighted mean of the AUCs of the cell (x fold, when fold is given) blocks over `cells` on rows idx (rows with
    a NaN score dropped; a block with one class only is skipped); NaN if none."""
    from sklearn.metrics import roc_auc_score
    tot, w = 0.0, 0
    for c in cells:
        for k in ([None] if fold is None else sorted(set(fold))):
            m = (cell[idx] == c) & ~np.isnan(s[idx]) & (True if k is None else fold[idx] == k)
            j = idx[m]
            if len(set(y[j])) < 2:
                continue
            tot += roc_auc_score(y[j], s[j]) * len(j)
            w += len(j)
    return tot / w if w else float("nan")


def analyze():
    rows, Z, self_p, other, reader, n_self = load()
    y = np.array([t["mode"] == "unilateral" for t in rows], dtype=int)
    cell = np.array([t["cell"] for t in rows])
    task = np.array([t["task"] for t in rows])
    pk = [t["pkey"] for t in rows]
    sub = np.array([t["j"] in READ_J for t in rows])
    fold = folds(y, task)
    nan = float("nan")
    S = {"hist": oof(hist_features(rows), y, task), "probe0": oof(Z[:, POS["first"]], y, task),
         "self": np.array([nan if self_p.get(k) is None else self_p[k] for k in pk]),
         "other": np.array([np.nanmean([v for v in other.get(k, {}).values() if v is not None] or [nan]) for k in pk]),
         "other_haiku": np.array([nan if other.get(k, {}).get(JUDGE_A) is None else other[k][JUDGE_A] for k in pk]),
         "other_gpt": np.array([nan if other.get(k, {}).get(JUDGE_B) is None else other[k][JUDGE_B] for k in pk])}
    for q in ("q25", "q50"):
        S[f"probe_{q}"] = oof(Z[:, POS[q]], y, task)
        S[f"reader_{q}"] = np.array([nan if reader.get(f"{t['key']}|{q}") is None else reader[f"{t['key']}|{q}"]
                                     for t in rows])
    # every prompt-level score that is a probability, for calibration
    probs = ("self", "other", "other_haiku", "other_gpt", "reader_q25", "reader_q50")
    tasks = sorted(set(task))
    by_task = {tk: np.where(task == tk)[0] for tk in tasks}
    rng = random.Random(0)
    boots = [np.concatenate([by_task[rng.choice(tasks)] for _ in tasks]) for _ in range(N_BOOT)]
    every, read_rows = np.arange(len(y)), np.where(sub)[0]
    rset = set(read_rows)

    def scope(name):
        return read_rows if (name.startswith("reader") or name.startswith("probe_q")) else every

    def stat(name, cells, idx, by_fold=True):
        keep = np.array([i for i in idx if i in rset]) if scope(name) is read_rows else idx
        return auc_within(S[name], y, cell, keep, cells, fold if by_fold else None)

    def ci(fn):
        est = fn(every)
        draws = np.array([fn(b) for b in boots])
        draws = draws[~np.isnan(draws)]
        return {"est": est, "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}

    res = {"n_traces": int(len(y)), "n_read": int(sub.sum()), "rate": {c: float(y[cell == c].mean()) for c in CELLS},
           "coverage": {n: float(np.mean(~np.isnan(S[n][scope(n)]))) for n in S},
           "self_samples_parsed": sum(1 for v in self_p.values() if v is not None), "auc": {}, "calib": {}, "p9": {},
           "p10": {}}
    for n in S:
        res["auc"][n] = {"all": ci(lambda idx, n=n: stat(n, CELLS, idx)),
                         **{c: ci(lambda idx, n=n, c=c: stat(n, (c,), idx)) for c in CELLS},
                         "auc_f": stat(n, CELLS, every, by_fold=False)}
    for n in probs:
        res["calib"][n] = {c: (float(np.nanmean(S[n][scope(n)][cell[scope(n)] == c])) - float(
            y[scope(n)][cell[scope(n)] == c].mean())) * 100 for c in CELLS}
    pairs = {"probe0_minus_self": ("probe0", "self"), "hist_minus_self": ("hist", "self"),
             "other_minus_self": ("other", "self"), "probe0_minus_other": ("probe0", "other"),
             "probe_q25_minus_reader_q25": ("probe_q25", "reader_q25"),
             "probe_q50_minus_reader_q50": ("probe_q50", "reader_q50")}
    for k, (a, b) in pairs.items():
        res["p9"][k] = ci(lambda idx, a=a, b=b: stat(a, CELLS, idx) - stat(b, CELLS, idx))

    def drop(n):   # AUC in the conflict cell minus in the calm cells
        return lambda idx: stat(n, CONFLICT, idx) - stat(n, CALM, idx)
    for v, nv in (("self", "probe0"), ("other", "probe0"), ("reader_q25", "probe_q25"), ("reader_q50", "probe_q50")):
        res["p10"][f"{v}_vs_{nv}"] = {"verbal_drop": ci(drop(v)), "probe_drop": ci(drop(nv)),
                                      "did": ci(lambda idx, v=v, nv=nv: drop(v)(idx) - drop(nv)(idx))}
    d = res["p10"]["self_vs_probe0"]["did"]
    bias = res["calib"]["self"]
    res["verdicts"] = {
        "words_fail_on_conflict_auc": bool(d["hi"] < 0),
        "self_underpredicts_conflict": bool(bias["CH_ownP"] <= BIAS_PP and bias["CH_ownP"] - np.mean(
            [bias[c] for c in CALM]) <= BIAS_PP),
        "best_before_reasoning": max(("hist", "probe0", "self", "other"), key=lambda n: res["auc"][n]["all"]["est"]),
    }
    return res


def _ci(b, f="{:+.3f}"):
    return "n/a" if b is None or b["est"] != b["est"] else \
        f"{f.format(b['est'])} [{f.format(b['lo'])}, {f.format(b['hi'])}]"


def summary(r):
    L = ["# PA-P9 / P10 analysis", "",
         f"{r['n_traces']} traces ({r['n_read']} read by the reasoning reader); penalise rate "
         + ", ".join(f"{c} {100 * v:.1f}%" for c, v in r["rate"].items()), "",
         "| predictor | within-cell AUC | none | own_R | CH_ownP (conflict) | F's across-fold AUC | coverage |",
         "|---|---|---|---|---|---|---|"]
    for n, a in r["auc"].items():
        L.append(f"| {n} | {_ci(a['all'], '{:.3f}')} | {_ci(a['none'], '{:.3f}')} | {_ci(a['own_R'], '{:.3f}')} | "
                 f"{_ci(a['CH_ownP'], '{:.3f}')} | {a['auc_f']:.3f} | {100 * r['coverage'][n]:.0f}% |")
    L += ["", "Calibration (mean forecast - actual, points): " + "; ".join(
        f"{n}: " + ", ".join(f"{c} {v:+.1f}" for c, v in d.items()) for n, d in r["calib"].items()), "",
          "P9 paired differences (within-cell AUC): " + "; ".join(f"{k} {_ci(v)}" for k, v in r["p9"].items()), "",
          "P10 (AUC in conflict minus calm; did = verbal minus probe): " + "; ".join(
              f"{k}: verbal {_ci(v['verbal_drop'])}, probe {_ci(v['probe_drop'])}, did {_ci(v['did'])}"
              for k, v in r["p10"].items()), "", f"Verdicts: {r['verdicts']}"]
    return "\n".join(L) + "\n"


def main():
    r = analyze()
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "analysis.json").write_text(json.dumps(r, indent=1, default=float))
    (RES / "summary.md").write_text(summary(r))
    print(summary(r))


if __name__ == "__main__":
    main()

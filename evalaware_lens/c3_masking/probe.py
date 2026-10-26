"""EL-C3 disposition probe (DESIGN.md §2), on the pod after every 9B cell's acts.py dump.

For each pair (trained cell, its base) in PAIRS and each layer: u = the unit difference of means, trained minus base,
over the neutral chat prompts of the fit split (the same prompts read by both models). Scores s = h . u for every
prompt of both models. Reported (written to OUT/probe/probe_{trained}.json; the activations themselves stay on the pod):
  auc_check    trained vs base on the held-out chat prompts
  auc_agentic  on the agentic neutral prompts (format transfer: fit on chat, tested on desk-format prompts)
  auc_task     on the pair's own task prompts under the neutral cue (TASK_FMT)
  paired_*     the share of prompts where the trained model's score exceeds the base's on the same prompt
  shuf_p95     the 95th percentile, over N_SHUF directions fit after permuting the model labels over the pooled fit
               activations (both models' rows together, split at random into two halves), of max(auc, 1 - auc) on the
               check prompts
  rand_p95     the same over N_RAND seeded Gaussian unit directions
  rows         per task prompt: key, fmt, cue, w, tid, ctx and both models' scores (analyze.py's input)

  /root/venv/bin/python -m evalaware_lens.c3_masking.probe --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

PAIRS = {"Q9-RISKY_s1": "Q9", "Q9-SAFE_s1": "Q9", "VN-NW_s1": "VN", "VN-NZ_s1": "VN", "VN-ACT_s1": "VN",
         "VN-ACTZ_s1": "VN"}
TASK_FMT = {"Q9-RISKY_s1": "lot_f", "Q9-SAFE_s1": "lot_f", "VN-NW_s1": "act_f", "VN-NZ_s1": "act_f",
            "VN-ACT_s1": "code_s", "VN-ACTZ_s1": "code_s"}
N_SHUF, N_RAND = 20, 50


def auc(pos, neg):
    """P(a random positive scores above a random negative), ties half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    for v in np.unique(allv):   # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def load(out, cell):
    z = np.load(Path(out, "acts", f"acts_{cell}.npz"))
    meta = json.loads(Path(out, "acts", f"acts_{cell}.json").read_text())
    return {k: z[k].astype(np.float64) for k in z.files}, meta["rows"]


def fit_dir(Xt, Xb):
    w = Xt.mean(0) - Xb.mean(0)
    return w / np.linalg.norm(w)


def pair(out, t, b, seed=0):
    At, rt = load(out, t)
    Ab, rb = load(out, b)
    ib = {r["key"]: j for j, r in enumerate(rb)}
    keys = [r["key"] for r in rt if r["key"] in ib]
    jt = [next(j for j, r in enumerate(rt) if r["key"] == k) for k in keys] if len(keys) != len(rt) else list(range(len(rt)))
    jb = [ib[k] for k in keys]
    rows = [rt[j] for j in jt]
    sel = lambda f: [n for n, r in enumerate(rows) if f(r)]  # noqa: E731
    fit, check = sel(lambda r: r["split"] == "fit"), sel(lambda r: r["split"] == "check" and r["fmt"] == "chat")
    agen = sel(lambda r: r["fmt"] == "agentic")
    task0 = sel(lambda r: r["fmt"] == TASK_FMT[t] and r.get("cue") == "neutral")
    tasks = sel(lambda r: r["split"] == "task")
    rng = np.random.default_rng(seed)
    res = {"trained": t, "base": b, "layers": {}}
    for L in sorted(At, key=lambda s: int(s[1:])):
        Xt, Xb = At[L][jt], Ab[L][jb]
        u = fit_dir(Xt[fit], Xb[fit])
        st, sb = Xt @ u, Xb @ u
        shuf = []
        pool = np.concatenate([Xt[fit], Xb[fit]])
        for _ in range(N_SHUF):
            perm = rng.permutation(len(pool))
            v = fit_dir(pool[perm[:len(fit)]], pool[perm[len(fit):]])
            a = auc(Xt[check] @ v, Xb[check] @ v)
            shuf.append(max(a, 1 - a))
        rnd = []
        for _ in range(N_RAND):
            g = rng.standard_normal(Xt.shape[1])
            g /= np.linalg.norm(g)
            a = auc(Xt[check] @ g, Xb[check] @ g)
            rnd.append(max(a, 1 - a))
        res["layers"][L] = {
            "auc_check": auc(st[check], sb[check]), "auc_agentic": auc(st[agen], sb[agen]),
            "auc_task": auc(st[task0], sb[task0]),
            "paired_check": float(np.mean(st[check] > sb[check])), "paired_task": float(np.mean(st[task0] > sb[task0]))
            if task0 else None,
            "gap_check": float(np.mean(st[check] - sb[check])), "gap_task": float(np.mean(st[task0] - sb[task0]))
            if task0 else None,
            "shuf_p95": float(np.percentile(shuf, 95)), "rand_p95": float(np.percentile(rnd, 95)),
            "rows": [{k: rows[n].get(k) for k in ("key", "fmt", "cue", "w", "tid", "ctx")}
                     | {"s_t": float(st[n]), "s_b": float(sb[n])} for n in tasks],
        }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="c3 probe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.out, "probe")
    d.mkdir(parents=True, exist_ok=True)
    for t, b in PAIRS.items():
        if not (Path(a.out, "acts", f"acts_{t}.npz").exists() and Path(a.out, "acts", f"acts_{b}.npz").exists()):
            print(f"[c3-probe] {t}: activations missing, skipped", flush=True)
            continue
        r = pair(a.out, t, b)
        (d / f"probe_{t}.json").write_text(json.dumps(r))
        s = {L: (round(v["auc_check"], 3), round(v["auc_task"] or -1, 3), round(v["shuf_p95"], 3)) for L, v in
             r["layers"].items()}
        print(f"[c3-probe] {t} vs {b}: (auc_check, auc_task, shuf_p95) {s}", flush=True)


if __name__ == "__main__":
    main()

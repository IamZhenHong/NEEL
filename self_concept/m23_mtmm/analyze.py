"""SC-M23 analysis (DESIGN.md §5), frozen at sc-m23-prereg: the multitrait-multimethod (MTMM) matrix.

  uv run python -m self_concept.m23_mtmm.analyze [RESULTS_DIR]     -> analysis.json + summary.md

Scores per model (BASE, RISKY, SAFE, W, Z), trait t (risk, habit) and method M:
  self / pred / act   content logit per cluster: letter reads (both orders) (logit p_o0 + logit p_o1) / 2, option /
                      decision reads logit p (p clipped at 0.5% / 99.5%); the method score = the mean over clusters
  lens                D = deep-band mean of log-mass(POS) - log-mass(NEG) of t's word sets, averaged over the act prompts
                      (decision token) and the self prompts (report token), each as a cluster mean; score = the mean of
                      the two positions (both reported)
  probe               t's probe on the block-32 outputs at t's act prompts, fit on BASE: label = BASE's own choice
                      (act p > 0.5), StandardScaler + LogisticRegression(C = 1); BASE's own scores leave-one-cluster-out,
                      the other models scored by the full fit; score = mean decision value. Fallback, if a class has
                      < 5 act prompts in BASE: fit on t's contrast models (risk RISKY vs SAFE, habit W vs Z), each
                      cluster scored by a fit without it
Effects, in units of the measure's within-model SD over clusters (pooled over the five models):
  d_own(t, M)   = s(t+) - s(t-)        risk: RISKY - SAFE; habit: W - Z
  d_cross(t, M) = s(o+) - s(o-)        the other trait's contrast on t's measure
Bootstrap: 10,000 draws, seed 0, over each (t, M)'s clusters, the same draw for the five models (probe weights fixed
at the full fit).
Rules, per trait:
  convergent     d_own lo > 0 for >= 4 of the 5 methods
  discriminant   d_own - |d_cross| lo > 0 for >= 4 of the 5 methods
  verdict        "convergent and discriminant" / "convergent, not discriminant" / "not convergent"
Descriptive: the 10 x 10 correlation matrix of the measures across the five models (Pearson), the mean
monotrait-heteromethod r against the mean heterotrait r.
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

EPS, N_BOOT = 0.005, 10000
MODELS = ("BASE", "RISKY", "SAFE", "W", "Z")
TRAITS = ("risk", "habit")
METHODS = ("self", "pred", "act", "lens", "probe")
CONTRAST = {"risk": ("RISKY", "SAFE"), "habit": ("W", "Z")}
OTHER = {"risk": "habit", "habit": "risk"}


def _lg(p):
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def load(d):
    d = Path(d)
    reads = {m: [json.loads(x) for x in (d / f"reads_{m}.jsonl").read_text().splitlines() if x.strip()]
             for m in MODELS if (d / f"reads_{m}.jsonl").exists()}
    resid = {m: np.load(d / f"resid_{m}.npy").astype(np.float32) for m in MODELS if (d / f"resid_{m}.npy").exists()}
    info = json.loads((d / "info.json").read_text())
    return reads, resid, info


def cluster_scores(reads):
    """{(model, trait, method): {cluster: value}} for self / pred / act and lens_dec / lens_rep."""
    acc = defaultdict(lambda: defaultdict(list))
    for m, rows in reads.items():
        for r in rows:
            acc[(m, r["trait"], r["method"])][r["cluster"]].append(float(_lg(r["p"])))
            lensD = np.mean([v[0] - v[1] for v in r["lens"].values()])
            pos = "lens_dec" if r["method"] == "act" else "lens_rep" if r["method"] == "self" else None
            if pos:
                acc[(m, r["trait"], pos)][r["cluster"]].append(float(lensD))
    return {k: {c: float(np.mean(v)) for c, v in cl.items()} for k, cl in acc.items()}


def probe_scores(reads, resid, info):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pids = info["pids"]
    out, meta = {}, {}
    for t in TRAITS:
        rows = {m: {r["pid"]: r for r in reads[m]} for m in reads}
        idx = [i for i, pid in enumerate(pids) if pid.startswith(f"{t}|act|")]
        clusters = [rows["BASE"][pids[i]]["cluster"] for i in idx]
        yb = np.array([int(rows["BASE"][pids[i]]["p"] > 0.5) for i in idx])
        use_base = min(yb.sum(), len(yb) - yb.sum()) >= 5
        meta[t] = {"fit": "base_labels" if use_base else "contrast", "n_pos_base": int(yb.sum()), "n": len(yb)}

        def fit(X, y):
            sc = StandardScaler().fit(X)
            return sc, LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X), y)
        uniq = sorted(set(clusters))
        if use_base:
            Xb = resid["BASE"][idx]
            sc, lr = fit(Xb, yb)
            for m in MODELS:
                X = resid[m][idx]
                dec = lr.decision_function(sc.transform(X))
                if m == "BASE":
                    dec = np.zeros(len(idx))
                    for c in uniq:
                        tr = np.array([cl != c for cl in clusters])
                        if len(set(yb[tr])) < 2:
                            continue
                        s2, l2 = fit(Xb[tr], yb[tr])
                        dec[~tr] = l2.decision_function(s2.transform(Xb[~tr]))
                out[(m, t, "probe")] = {c: float(np.mean(dec[[cl == c for cl in clusters]])) for c in uniq}
        else:
            pos, neg = CONTRAST[t]
            X = np.vstack([resid[pos][idx], resid[neg][idx]])
            y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
            cl2 = clusters + clusters
            for m in MODELS:
                dec = np.zeros(len(idx))
                for c in uniq:
                    tr = np.array([x != c for x in cl2])
                    s2, l2 = fit(X[tr], y[tr])
                    te = np.array([x == c for x in clusters])
                    dec[te] = l2.decision_function(s2.transform(resid[m][idx][te]))
                out[(m, t, "probe")] = {c: float(np.mean(dec[[cl == c for cl in clusters]])) for c in uniq}
    return out, meta


def analyse(reads, resid, info, n_boot=N_BOOT, rng_seed=0):
    cs = cluster_scores(reads)
    ps, pmeta = probe_scores(reads, resid, info)
    cs.update(ps)
    for m in MODELS:                          # the lens method = the mean of its two positions
        for t in TRAITS:
            dec, rep = cs[(m, t, "lens_dec")], cs[(m, t, "lens_rep")]
            cs[(m, t, "lens")] = {**{f"dec|{c}": v for c, v in dec.items()}, **{f"rep|{c}": v for c, v in rep.items()}}
    rng = np.random.default_rng(rng_seed)
    res, scores = {}, {}
    for t in TRAITS:
        for M in METHODS + ("lens_dec", "lens_rep"):
            cl = sorted(cs[("BASE", t, M)])
            V = {m: np.array([cs[(m, t, M)][c] for c in cl]) for m in MODELS}
            if M == "lens":                    # weight the two positions equally, whatever their cluster counts
                wd = np.array([0.5 / sum(c.startswith("dec|") for c in cl) if c.startswith("dec|") else
                               0.5 / sum(c.startswith("rep|") for c in cl) for c in cl])
            else:
                wd = np.full(len(cl), 1 / len(cl))
            W = np.vstack([np.ones(len(cl))] + [np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl))
                                                for _ in range(n_boot)]) * wd
            W = W / W.sum(1, keepdims=True)
            S = {m: W @ V[m] for m in MODELS}
            sd = float(np.sqrt(np.mean([np.var(V[m]) for m in MODELS]))) or 1.0
            pos, neg = CONTRAST[t]
            opos, oneg = CONTRAST[OTHER[t]]
            own, cross = (S[pos] - S[neg]) / sd, (S[opos] - S[oneg]) / sd
            gap = own - np.abs(cross)

            def summ(x):
                lo, hi = np.percentile(x[1:], [2.5, 97.5])
                return {"est": round(float(x[0]), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}
            res[f"{t}|{M}"] = {"d_own": summ(own), "d_cross": summ(cross), "own_minus_cross": summ(gap), "sd": round(sd, 4),
                               "scores": {m: round(float(S[m][0]), 3) for m in MODELS}, "n_clusters": len(cl)}
            scores[(t, M)] = [float(S[m][0]) for m in MODELS]
    verdict = {}
    for t in TRAITS:
        conv = sum(res[f"{t}|{M}"]["d_own"]["lo"] > 0 for M in METHODS)
        disc = sum(res[f"{t}|{M}"]["own_minus_cross"]["lo"] > 0 for M in METHODS)
        verdict[t] = {"convergent_methods": conv, "discriminant_methods": disc,
                      "verdict": ("not convergent" if conv < 4 else
                                  "convergent and discriminant" if disc >= 4 else "convergent, not discriminant")}
    keys = [(t, M) for t in TRAITS for M in METHODS]
    corr = {}
    for (a, b) in combinations(keys, 2):
        x, y = np.array(scores[a]), np.array(scores[b])
        corr[f"{a[0]}:{a[1]}~{b[0]}:{b[1]}"] = (round(float(np.corrcoef(x, y)[0, 1]), 3)
                                                if x.std() > 0 and y.std() > 0 else None)
    mono = [v for k, v in corr.items() if v is not None and k.split("~")[0].split(":")[0] == k.split("~")[1].split(":")[0]]
    hetero = [v for k, v in corr.items() if v is not None and k.split("~")[0].split(":")[0] != k.split("~")[1].split(":")[0]]
    return {"effects": res, "verdict": verdict, "probe": pmeta, "corr": corr,
            "mean_r_monotrait_heteromethod": round(float(np.mean(mono)), 3) if mono else None,
            "mean_abs_r_heterotrait": round(float(np.mean(np.abs(hetero))), 3) if hetero else None}


def summary_md(a):
    L = ["# SC-M23 summary (frozen analyze.py)", "",
         "Verdicts: " + "; ".join(f"**{t}** {v['verdict']} (convergent {v['convergent_methods']}/5, discriminant "
                                  f"{v['discriminant_methods']}/5)" for t, v in a["verdict"].items()), "",
         f"Probes: {a['probe']}", "",
         "| trait | method | d_own | d_cross | own - abs(cross) | BASE / RISKY / SAFE / W / Z |", "|---|---|---|---|---|---|"]
    for k, v in a["effects"].items():
        t, M = k.split("|")
        sc = " / ".join(str(v["scores"][m]) for m in MODELS)
        L.append(f"| {t} | {M} | {v['d_own']['est']} [{v['d_own']['lo']}, {v['d_own']['hi']}] | {v['d_cross']['est']} "
                 f"[{v['d_cross']['lo']}, {v['d_cross']['hi']}] | {v['own_minus_cross']['est']} [{v['own_minus_cross']['lo']}, "
                 f"{v['own_minus_cross']['hi']}] | {sc} |")
    L += ["", f"Across-model correlations: mean monotrait-heteromethod r {a['mean_r_monotrait_heteromethod']}, mean |r| "
              f"heterotrait {a['mean_abs_r_heterotrait']}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m23_mtmm.analyze")
    ap.add_argument("results_dir", nargs="?", default="self_concept/m23_mtmm/results/main")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    out = analyse(*load(a.results_dir), n_boot=a.n_boot)
    (Path(a.results_dir) / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    (Path(a.results_dir) / "summary.md").write_text(summary_md(out))
    print(summary_md(out))


if __name__ == "__main__":
    main()

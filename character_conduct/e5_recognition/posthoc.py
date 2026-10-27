"""CC-E5 step 1 post-hoc checks (not pre-registered; RESULT.md labels every number from here as post hoc).

  anchors  the M1 / M2 / M3 statistics with the anchors taken from one judge alone (Sonnet, Gemini), from the
           pair-plus-auditor majority (the frozen rule before the kappa fallback), and from the frozen fallback
           (earliest turn any judge marks), all at the frozen layer L* with the frozen probe. Asked for by the
           coordinator after the kappa gate failed (10-28).

  within   G1's held-out judge-mode AUC inside each author model (models with >= 5 runs on each side), their mean,
           and the same mean for the 50 shuffled-label direction sets. The shuffled sets keep each model's violation
           rate, so their pooled AUC (G1's floor, 0.889) is what the author model alone predicts; the within-model
           AUC is what the probe reads beyond the author. Asked after G2 failed (10-28).
  actor    a probe fitted in the model's own context instead of judge mode: leave-scenario-out difference of means,
           V vs O, at t_end and (control) at g_0, every saved layer; floor = 200 label permutations within variant
           (p95). A g_0 AUC as high as t_end's means it reads the scenario's prompt, not the violation.

  uv run python -m character_conduct.e5_recognition.posthoc --reads results/reads --steps results/steps
"""
import argparse
import json
from pathlib import Path

import numpy as np

from character_conduct.e5_recognition import analyze as A

N_PERM = 200


def within(probe, min_n=5):
    y, model, scen = probe["y"].astype(bool), probe["model"], probe["scen"]
    L = probe["g1"]["layer"]

    def per_model(proj):
        out = {}
        for m in np.unique(model):
            k = model == m
            if (y[k]).sum() >= min_n and (~y[k]).sum() >= min_n:
                out[str(m)] = A.auc(proj[k][y[k]], proj[k][~y[k]])
        return out

    real = per_model(probe["per"][L]["proj"])
    X = None
    shuf = []
    for d in probe["shuf"]:
        if X is None:
            X = probe["_X"]
        proj = np.array([X[i] @ d[s] for i, s in enumerate(scen)])
        shuf.append(float(np.mean(list(per_model(proj).values()))))
    return {"layer": L, "per_model": real, "n_models": len(real), "mean_within": float(np.mean(list(real.values()))),
            "shuf_mean_within_p95": float(np.percentile(shuf, 95)),
            "n_per_model": {str(m): [int((y & (model == m)).sum()), int((~y & (model == m)).sum())]
                            for m in np.unique(model)}}


def actor_probe(reads, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for L in A.LAYERS:
        runs = A.load_actor(reads, L)
        y = np.array([A.is_v(r) for r in runs])
        scen = np.array([r["scenario"] for r in runs])
        var = np.array([r["variant"] for r in runs])
        row = {}
        for pos in ("t_end", "g0"):
            X = np.stack([r["t_end"] if pos == "t_end" else r["g"][0] for r in runs]).astype(np.float64)
            proj, a, _ = A.heldout(X, y, scen, A.lso_dirs(X, y, scen))
            perm = []
            for _ in range(N_PERM):
                yp = y.copy()
                for v in np.unique(var):
                    idx = np.where(var == v)[0]
                    yp[idx] = y[rng.permutation(idx)]
                perm.append(A.heldout(X, y, scen, A.lso_dirs(X, yp, scen))[1])
            row[pos] = {"auc": a, "perm_p95": float(np.percentile(perm, 95)),
                        "by_variant": {v: A.auc(proj[(var == v) & y], proj[(var == v) & ~y]) for v in np.unique(var)}}
        out[L] = row
    return out


def actor_timing(reads, steps_by, layer, n_perm=50, seed=0):
    """The M1 / M2 / M3 statistics with the actor-fitted t_end direction at `layer` (chosen by t_end AUC alone, before
    any timing number) in place of the judge-mode one; units = its held-out t_end gap. Floor: within-variant label
    permutations (p95 of J_V and of after_V)."""
    rng = np.random.default_rng(seed)
    runs = A.load_actor(reads, layer)
    y = np.array([A.is_v(r) for r in runs])
    scen = np.array([r["scenario"] for r in runs])
    var = np.array([r["variant"] for r in runs])
    X = np.stack([r["t_end"] for r in runs]).astype(np.float64)
    dirs = A.lso_dirs(X, y, scen)
    gap = A.heldout(X, y, scen, dirs)[2]
    fns = A.stat_fns()
    out = {"layer": layer, "gap": gap}
    for name, steps in steps_by.items():
        G = A.by_scen(A.contrasts(runs, A.project(runs, dirs, gap), steps))
        out[name] = {k: A.boot_ci(G, fns[k]) for k in ("J_V", "J_O", "J_V-J_O", "J_V-ordinary", "after_V")}
    fl = {"J_V": [], "after_V": []}
    for _ in range(n_perm):
        yp = y.copy()
        for v in np.unique(var):
            idx = np.where(var == v)[0]
            yp[idx] = y[rng.permutation(idx)]
        d = A.lso_dirs(X, yp, scen)
        G = A.by_scen(A.contrasts(runs, A.project(runs, d, gap), steps_by["cons"]))
        for k in fl:
            fl[k].append(fns[k]([G[s] for s in sorted(G)]))
    out["perm_p95"] = {k: float(np.percentile([x for x in v if x is not None], 95)) for k, v in fl.items()}
    return out


def single(pair, audit, turns, which):
    """{run: {"M", "A"}} with each anchor from one source: a judge id, "majority" or "cons"."""
    info, _, _ = A.step_info(pair, audit, turns)
    out = {}
    for rk, v in info.items():
        if which in ("majority", "cons"):
            m = v["mention"] if which == "majority" else v["mention_cons"]
            a = v["execute"] if which == "majority" else v["execute_both"]
        else:
            j = A.JUDGES.index(which)
            lab = [x[j] for x in v["labels"]]
            m = next((i for i, x in enumerate(lab) if x and A.BIN["mention"](x)), None)
            a = next((i for i, x in enumerate(lab) if x and A.BIN["execute"](x)), None)
        out[rk] = {"M": m, "A": a}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    a = ap.parse_args(argv)
    XJ, meta = A.load_judge(a.reads)
    probe = A.fit_probe(XJ, meta)
    probe["_X"] = XJ[:, A.CAND.index(probe["g1"]["layer"])]
    runs = A.load_actor(a.reads, probe["g1"]["layer"])
    pair = json.loads((a.steps / "labels_pair.json").read_text())["labels"]
    af = a.steps / "labels_audit.json"
    audit = json.loads(af.read_text())["labels"] if af.exists() else {}
    turns = {f"{r['variant']}|{r['scenario']}": len(r["g"]) for r in runs}
    P = A.project(runs, probe["dirs"], probe["g1"]["gap"])
    fns = A.stat_fns()
    out = {"layer": probe["g1"]["layer"], "anchors": {}}
    for which in (A.JUDGES[0], A.JUDGES[1], "majority", "cons"):
        steps = single(pair, audit, turns, which)
        by = A.contrasts(runs, P, steps)
        G = A.by_scen(by)
        row = {k: A.boot_ci(G, fns[k]) for k in ("J_V", "J_O", "J_V-J_O", "J_V-ordinary", "after_V")}
        row["n_V"] = sum(1 for x in by.values() if x["v"] and x["J"] is not None)
        row["n_O"] = sum(1 for x in by.values() if not x["v"] and x["J"] is not None)
        out["anchors"][which] = row
    out["within"] = within(probe)
    out["actor"] = actor_probe(a.reads)
    best = max(out["actor"], key=lambda L: out["actor"][L]["t_end"]["auc"])
    out["actor_timing"] = actor_timing(a.reads, {w: single(pair, audit, turns, w) for w in ("cons", "majority")}, best)
    (a.reads / "posthoc.json").write_text(json.dumps(out, indent=1, default=float))
    w = out["within"]
    print("within-model judge-mode AUC at L", w["layer"], round(w["mean_within"], 3), "shuffled p95",
          round(w["shuf_mean_within_p95"], 3), {k: round(v, 2) for k, v in w["per_model"].items()})
    for L, r in out["actor"].items():
        print("actor probe L", L, {p: (round(r[p]["auc"], 3), round(r[p]["perm_p95"], 3),
                                       {v: round(x, 2) for v, x in r[p]["by_variant"].items()}) for p in r})
    t = out["actor_timing"]
    print("actor timing L", t["layer"], "perm p95", {k: round(v, 3) for k, v in t["perm_p95"].items()})
    for w in ("cons", "majority"):
        print("  ", w, {s: (round(t[w][s]["est"], 3), round(t[w][s]["lo"], 3), round(t[w][s]["hi"], 3))
                        for s in t[w]})
    for k, r in out["anchors"].items():
        print(k, {s: (round(r[s]["est"], 3) if r[s]["est"] is not None else None, round(r[s]["lo"], 3),
                      round(r[s]["hi"], 3)) for s in ("J_V", "J_O", "J_V-J_O", "J_V-ordinary", "after_V")},
              "n", r["n_V"], r["n_O"])


if __name__ == "__main__":
    main()

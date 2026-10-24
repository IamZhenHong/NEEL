"""SC-M22 analysis (DESIGN.md §4), frozen at sc-m22-prereg.

  uv run python -m self_concept.m22_fitdir.analyze [RESULTS_DIR]     -> analysis.json + summary.md

Cells NW_s1, NZ_s1 (V1's habit model and its twin) and VN_s1 (their base), from read.py; directions fit, fit_perp,
shuf0..3, rand0..7 at k = 1, 2 (norm k * R_NORM).
Report (order-consistent, SC-M4 amendment 1's measure): per model, family f (self_mc 8 stem clusters, habit_level 32
paraphrase x desk clusters), direction d, dose k: c = mean over clusters of (logit p_o0 + logit p_o1) / 2;
S = c(d, k) - c(none);  E(d, k) = mean over the two families of S(NW) - S(NZ).
Act: A(m, d, k) = mean over the 8 HELD-OUT desks of the lo shift (training desks reported separately).
Controls at each k: ctrl_E = max over shuf0..3 and rand0..7 of E; ctrl_A = max over the same of A(NZ).
Bootstrap: 10,000 draws, seed 0, clusters within each family and held-out desks, the same draw for every model.
Primary: d = fit, k = 1. Verdict, first match:
  not read             direction gate fails (A_NZ(fit, 1) held out < +1.0 nat, or lo <= 0, or <= ctrl_A) or coherence
                       fails (top-1 a letter on < 90% of NW's or NZ's report prompts at fit, 1)
  routes when present  E(fit, 1) >= +1.0 with lo > 0 and > ctrl_E, and E(fit_perp, 1) lo > 0
  writes the answer    S_NZ(fit, 1) >= +1.0 with lo > 0, and E(fit, 1) hi < 1.0 or <= ctrl_E
  no effect on report  S_NW(fit, 1) and S_NZ(fit, 1) both inside +/-1.0
  inconclusive         otherwise
Secondary (described): k = 2; fit_perp; E against VN; the shuffled and random directions' own E and A; train-desk A.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from self_concept.m22_fitdir import fit as F
from self_concept.m4_inject import analyze_b as B

N_BOOT, BAND, COHERENT = 10000, 1.0, 0.9
FAMS = ("self_mc", "habit_level")
CTRLS = [f"shuf{j}" for j in range(4)] + [f"rand{j}" for j in range(8)]


def load(d):
    out = {}
    for f in sorted(Path(d).glob("reads_*.jsonl")):
        out[f.stem.replace("reads_", "")] = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return out


def analyse(data, n_boot=N_BOOT, rng_seed=0):
    if not {"NW_s1", "NZ_s1"} <= set(data):
        return {"verdict": "not read", "missing": sorted({"NW_s1", "NZ_s1"} - set(data))}
    T = {c: B.tables(r) for c, r in data.items()}
    clusters = {f: sorted(T["NW_s1"][0][(f, "none", 0.0)]) for f in FAMS}
    test = [t for t in sorted(T["NZ_s1"][1][("none", 0.0)]) if t not in F.TRAIN]
    train = [t for t in sorted(T["NZ_s1"][1][("none", 0.0)]) if t in F.TRAIN]
    rng = np.random.default_rng(rng_seed)
    W = {f: np.vstack([np.ones(len(clusters[f]))] + [np.bincount(rng.integers(0, len(clusters[f]), len(clusters[f])),
                                                                 minlength=len(clusters[f])) for _ in range(n_boot)])
         for f in FAMS}
    Wt = np.vstack([np.ones(len(test))] + [np.bincount(rng.integers(0, len(test), len(test)), minlength=len(test))
                                           for _ in range(n_boot)])

    def c_vec(cell, f, d, k):
        cl = T[cell][0][(f, d, k)]
        return (W[f] @ np.array([np.mean(cl[c]) for c in clusters[f]])) / W[f].sum(1)

    def S(cell, d, k):
        return np.mean([c_vec(cell, f, d, k) - c_vec(cell, f, "none", 0.0) for f in FAMS], 0)

    def E(d, k, twin="NZ_s1"):
        return S("NW_s1", d, k) - S(twin, d, k)

    def A(cell, d, k, tasks=None, w=None):
        tasks = tasks or test
        v = np.array([T[cell][1][(d, k)][t] - T[cell][1][("none", 0.0)][t] for t in tasks])
        w = Wt if w is None else w
        return (w @ v) / w.sum(1)

    def summ(x):
        lo, hi = np.percentile(x[1:], [2.5, 97.5])
        return {"est": round(float(x[0]), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}

    dirs = ["fit", "fit_perp"] + CTRLS
    out = {"E": {}, "S_NW": {}, "S_NZ": {}, "A_NZ": {}, "A_NW": {}, "A_NZ_train": {}, "E_vn": {}}
    ones_train = np.ones((1, len(train)))
    for d in dirs:
        for k in (1.0, 2.0):
            key = f"{d}|{k}"
            out["E"][key] = summ(E(d, k))
            out["S_NW"][key] = summ(S("NW_s1", d, k))
            out["S_NZ"][key] = summ(S("NZ_s1", d, k))
            out["A_NZ"][key] = summ(A("NZ_s1", d, k))
            out["A_NW"][key] = summ(A("NW_s1", d, k))
            out["A_NZ_train"][key] = round(float(A("NZ_s1", d, k, train, ones_train)[0]), 3)
            if "VN_s1" in T:
                out["E_vn"][key] = summ(E(d, k, "VN_s1"))
    out["ctrl_E"] = {str(k): max(out["E"][f"{c}|{k}"]["est"] for c in CTRLS) for k in (1.0, 2.0)}
    out["ctrl_A"] = {str(k): max(out["A_NZ"][f"{c}|{k}"]["est"] for c in CTRLS) for k in (1.0, 2.0)}
    a = out["A_NZ"]["fit|1.0"]
    out["gate_direction"] = a["est"] >= BAND and a["lo"] > 0 and a["est"] > out["ctrl_A"]["1.0"]
    out["coherence"] = {c: round(T[c][2][("fit", 1.0)], 3) for c in ("NW_s1", "NZ_s1")}
    out["gate_coherence"] = min(out["coherence"].values()) >= COHERENT
    out["level_none"] = {c: {f: round(float(c_vec(c, f, "none", 0.0)[0]), 3) for f in FAMS} for c in T}
    out["verdict"] = verdict(out)
    return out


def verdict(o):
    if not (o["gate_direction"] and o["gate_coherence"]):
        return "not read"
    e, ep, snz, snw = o["E"]["fit|1.0"], o["E"]["fit_perp|1.0"], o["S_NZ"]["fit|1.0"], o["S_NW"]["fit|1.0"]
    if e["est"] >= BAND and e["lo"] > 0 and e["est"] > o["ctrl_E"]["1.0"] and ep["lo"] > 0:
        return "routes when present"
    if snz["est"] >= BAND and snz["lo"] > 0 and (e["hi"] < BAND or e["est"] <= o["ctrl_E"]["1.0"]):
        return "writes the answer"
    if all(x["lo"] > -BAND and x["hi"] < BAND for x in (snw, snz)):
        return "no effect on report"
    return "inconclusive"


def summary_md(o):
    if "E" not in o:
        return f"# SC-M22 summary\n\n**Verdict: {o['verdict']}** (missing {o.get('missing')})\n"
    L = ["# SC-M22 summary (frozen analyze.py)", "", f"**Verdict: {o['verdict']}**", "",
         f"Direction gate {'passes' if o['gate_direction'] else 'FAILS'}: held-out A_NZ(fit, 1) {o['A_NZ']['fit|1.0']} "
         f"(control max {o['ctrl_A']['1.0']}); coherence {o['coherence']}", "",
         "| direction | k | S(NW) | S(NZ) | E | A(NZ) held out | A(NZ) train | A(NW) held out |", "|---|---|---|---|---|---|---|---|"]
    for d in ["fit", "fit_perp"] + CTRLS:
        for k in (1.0, 2.0):
            key = f"{d}|{k}"
            e = o["E"][key]
            L.append(f"| {d} | {k} | {o['S_NW'][key]['est']} | {o['S_NZ'][key]['est']} | {e['est']} [{e['lo']}, {e['hi']}] | "
                     f"{o['A_NZ'][key]['est']} | {o['A_NZ_train'][key]} | {o['A_NW'][key]['est']} |")
    L += ["", f"Control max E: {o['ctrl_E']}; control max A(NZ): {o['ctrl_A']}", "",
          f"Report levels (no injection): {o['level_none']}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m22_fitdir.analyze")
    ap.add_argument("results_dir", nargs="?", default="self_concept/m22_fitdir/results/main")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    o = analyse(load(a.results_dir), a.n_boot)
    (Path(a.results_dir) / "analysis.json").write_text(json.dumps(o, indent=1) + "\n")
    (Path(a.results_dir) / "summary.md").write_text(summary_md(o))
    print(summary_md(o))


if __name__ == "__main__":
    main()

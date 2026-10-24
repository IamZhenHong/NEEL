"""SC-M4 amendment 1 analysis (DESIGN.md §9), frozen before pod 2's data.

  uv run python -m self_concept.m4_inject.analyze_b [RESULTS_DIR]     -> analysis_b.json + summary_b.md

Cells NW_s1 NZ_s1 NW_s2 NZ_s2 (V1's W / Z rows on VN, seeds 1 and 2) and VN_s1, from run_b.py.
Report (order-consistent): per model, family f (self_mc: 8 stem clusters; habit_level: 32 paraphrase x desk clusters;
trait, ident: 2 each), direction d, dose k:
  c = mean over f's clusters of the cluster mean of (logit p_o0 + logit p_o1) / 2   (p = the A / B share on unilateral)
  S(m, f, d, k) = c(d, k) - c(none);  E_f(d, k) = mean over seeds of [S(NW_s) - S(NZ_s)]
  E(d, k) = (E_self_mc + E_habit_level) / 2       primary family weight: the two habit items, equally
  floor = max over the 8 random directions of E(rand_j, 2)
Act: A(m, d, k) = mean over the 16 tasks of the shift in log p(penalise) - log p(refer); A_NZ = mean over the NZ seeds.
Bootstrap: 10,000 draws, seed 0, resampling each family's clusters and the tasks, the same draw for every model.
Primary: d = dom (the mean-difference direction), k = 2. Pod 1 already showed, descriptively, A(NZ, dom, 2) = +1.07 and
E_self_mc(dom, 2) = -0.06 [-0.20, +0.09] on seed 1; this rerun pre-registers both on two seeds and a larger item set.
Verdict, first match:
  not read              the direction gate fails (A_NZ(dom, 2) < +1.0, or lo <= 0, or <= the max of A_NZ(rand_j, 2)) or
                        the coherence gate fails (top-1 a letter in < 90% of any NW / NZ cell's report prompts at dom 2)
  routes when present   E(dom, 2) >= +1.0 with lo > 0 and > floor, and E(dom_perp, 2) lo > 0
  writes the answer     S_NZ(dom, 2) >= +1.0 with lo > 0 (seed mean, the two habit families), and E(dom, 2) hi < +1.0
                        or <= floor
  no effect on report   S_NW(dom, 2) and S_NZ(dom, 2) both inside +/-1.0
  inconclusive          otherwise
Secondary (described): doses 1 and 3; probe; per seed; per family (trait, ident too); E against VN.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS, N_BOOT, BAND, COHERENT = 0.005, 10000, 1.0, 0.9
SEEDS = (1, 2)
FAMS = ("self_mc", "habit_level", "trait", "ident")
PRIMARY_FAMS = ("self_mc", "habit_level")
RAND = [f"rand{j}" for j in range(8)]


def _lg(p):
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def load(d):
    out = {}
    for f in sorted(Path(d).glob("readsb_*.jsonl")):
        out[f.stem.replace("readsb_", "")] = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    return out


def tables(rows):
    """{(fam, dir, k): {cluster: [content logits of its prompt pairs]}}, {(dir, k): {task: lo}}, coherence."""
    pairs = defaultdict(dict)
    act = defaultdict(dict)
    coh = defaultdict(list)
    for r in rows:
        if r["block"] == "self":
            base = r["pid"].rsplit("o", 1)[0] if r["family"] != "habit_level" else r["pid"].replace(f"o{r['order']}", "")
            pairs[(r["family"], r["dir"], r["k"], r["cluster"], base)][r["order"]] = r["p_uni"]
            coh[(r["dir"], r["k"])].append(r["top1_letter"])
        else:
            act[(r["dir"], r["k"])][r["pid"]] = r["lo"]
    cl = defaultdict(lambda: defaultdict(list))
    for (fam, d, k, cluster, _), pr in pairs.items():
        if len(pr) == 2:
            cl[(fam, d, k)][cluster].append((float(_lg(pr[0])) + float(_lg(pr[1]))) / 2)
    return cl, act, {k: float(np.mean(v)) for k, v in coh.items()}


def analyse(data, n_boot=N_BOOT, rng_seed=0):
    need = [f"{m}_s{s}" for m in ("NW", "NZ") for s in SEEDS]
    if not all(c in data for c in need):
        return {"verdict": "not read", "missing": [c for c in need if c not in data]}
    T = {c: tables(r) for c, r in data.items()}
    clusters = {f: sorted(T["NW_s1"][0][(f, "none", 0.0)]) for f in FAMS}
    tasks = sorted(T["NZ_s1"][1][("none", 0.0)])
    rng = np.random.default_rng(rng_seed)
    W = {f: np.vstack([np.ones(len(clusters[f]))] + [np.bincount(rng.integers(0, len(clusters[f]), len(clusters[f])),
                                                                 minlength=len(clusters[f])) for _ in range(n_boot)])
         for f in FAMS}
    Wt = np.vstack([np.ones(len(tasks))] + [np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
                                            for _ in range(n_boot)])

    def c_vec(cell, f, d, k):
        cl = T[cell][0][(f, d, k)]
        v = np.array([np.mean(cl[c]) for c in clusters[f]])
        return (W[f] @ v) / W[f].sum(1)                     # [1 + n_boot]

    def S(cell, f, d, k):
        return c_vec(cell, f, d, k) - c_vec(cell, f, "none", 0.0)

    def E_f(f, d, k, twin="NZ"):
        return np.mean([S(f"NW_s{s}", f, d, k) - S(f"{twin}_s{s}" if twin != "VN" else "VN_s1", f, d, k)
                        for s in SEEDS], 0)

    def E(d, k):
        return np.mean([E_f(f, d, k) for f in PRIMARY_FAMS], 0)

    def Sm(m, d, k):
        return np.mean([np.mean([S(f"{m}_s{s}", f, d, k) for f in PRIMARY_FAMS], 0) for s in SEEDS], 0)

    def A(cell, d, k):
        v = np.array([T[cell][1][(d, k)][t] - T[cell][1][("none", 0.0)][t] for t in tasks])
        return (Wt @ v) / Wt.sum(1)

    def A_nz(d, k):
        return np.mean([A(f"NZ_s{s}", d, k) for s in SEEDS], 0)

    def summ(x):
        lo, hi = np.percentile(x[1:], [2.5, 97.5])
        return {"est": round(float(x[0]), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3)}

    out = {"E": {}, "E_fam": {}, "E_seed": {}, "S_NW": {}, "S_NZ": {}, "A_NZ": {}, "A_NW": {}, "E_vn": {}}
    dk = [("dom", 1.0), ("dom", 2.0), ("dom", 3.0), ("dom_perp", 2.0), ("probe", 1.0), ("probe", 2.0)] + \
         [(r, 2.0) for r in RAND]
    for d, k in dk:
        key = f"{d}|{k}"
        out["E"][key] = summ(E(d, k))
        out["S_NW"][key] = summ(Sm("NW", d, k))
        out["S_NZ"][key] = summ(Sm("NZ", d, k))
        out["A_NZ"][key] = summ(A_nz(d, k))
        out["A_NW"][key] = summ(np.mean([A(f"NW_s{s}", d, k) for s in SEEDS], 0))
        out["E_fam"][key] = {f: summ(E_f(f, d, k)) for f in FAMS}
        out["E_seed"][key] = {str(s): summ(np.mean([S(f"NW_s{s}", f, d, k) - S(f"NZ_s{s}", f, d, k)
                                                    for f in PRIMARY_FAMS], 0)) for s in SEEDS}
        if "VN_s1" in T:
            out["E_vn"][key] = summ(np.mean([E_f(f, d, k, twin="VN") for f in PRIMARY_FAMS], 0))
    out["floor_E"] = max(out["E"][f"{r}|2.0"]["est"] for r in RAND)
    out["floor_A_NZ"] = max(out["A_NZ"][f"{r}|2.0"]["est"] for r in RAND)
    a = out["A_NZ"]["dom|2.0"]
    out["gate_direction"] = a["est"] >= BAND and a["lo"] > 0 and a["est"] > out["floor_A_NZ"]
    out["coherence"] = {c: round(T[c][2][("dom", 2.0)], 3) for c in need}
    out["gate_coherence"] = min(out["coherence"].values()) >= COHERENT
    out["level_none"] = {c: {f: round(float(c_vec(c, f, "none", 0.0)[0]), 3) for f in FAMS} for c in T}
    out["verdict"] = verdict(out)
    return out


def verdict(o):
    if not (o["gate_direction"] and o["gate_coherence"]):
        return "not read"
    e, ep, snz, snw = o["E"]["dom|2.0"], o["E"]["dom_perp|2.0"], o["S_NZ"]["dom|2.0"], o["S_NW"]["dom|2.0"]
    if e["est"] >= BAND and e["lo"] > 0 and e["est"] > o["floor_E"] and ep["lo"] > 0:
        return "routes when present"
    if snz["est"] >= BAND and snz["lo"] > 0 and (e["hi"] < BAND or e["est"] <= o["floor_E"]):
        return "writes the answer"
    if all(x["lo"] > -BAND and x["hi"] < BAND for x in (snw, snz)):
        return "no effect on report"
    return "inconclusive"


def summary_md(o):
    if "E" not in o:
        return f"# SC-M4 amendment 1 summary\n\n**Verdict: {o['verdict']}** (missing {o.get('missing')})\n"
    L = ["# SC-M4 amendment 1 summary (frozen analyze_b.py)", "", f"**Verdict: {o['verdict']}**", "",
         f"Direction gate {'passes' if o['gate_direction'] else 'FAILS'}: A_NZ(dom, 2) {o['A_NZ']['dom|2.0']} (random max "
         f"{o['floor_A_NZ']}); coherence {o['coherence']}", "",
         "| direction | k | S(NW) | S(NZ) | E (self_mc + habit_level) | E seed 1 / 2 | A(NZ) | A(NW) |", "|---|---|---|---|---|---|---|---|"]
    for key in ("dom|1.0", "dom|2.0", "dom|3.0", "dom_perp|2.0", "probe|1.0", "probe|2.0"):
        d, k = key.split("|")
        e = o["E"][key]
        L.append(f"| {d} | {k} | {o['S_NW'][key]['est']} | {o['S_NZ'][key]['est']} | {e['est']} [{e['lo']}, {e['hi']}] | "
                 f"{o['E_seed'][key]['1']['est']} / {o['E_seed'][key]['2']['est']} | {o['A_NZ'][key]['est']} | {o['A_NW'][key]['est']} |")
    L += ["", f"Random floor of E at k = 2: {o['floor_E']}", "", f"Report levels (no injection): {o['level_none']}"]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m self_concept.m4_inject.analyze_b")
    ap.add_argument("results_dir", nargs="?", default="self_concept/m4_inject/results/b")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    o = analyse(load(a.results_dir), a.n_boot)
    (Path(a.results_dir) / "analysis_b.json").write_text(json.dumps(o, indent=1) + "\n")
    (Path(a.results_dir) / "summary_b.md").write_text(summary_md(o))
    print(summary_md(o))


if __name__ == "__main__":
    main()

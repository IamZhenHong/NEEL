"""jl2 analysis (design card DESIGN.md, implementation README.md). Local, no model (decodes with the tokenizer).
  uv run python -m jlens.e2_realism.analyze --eyeball                     # step 1: by-eye dump (before any metric)
  uv run --with statsmodels python -m jlens.e2_realism.analyze            # step 2: gate, M1-M3 -> results/summary.md
All masses are log-mass of a token set under the lens's softmax, averaged over layers 20-60 (step 4) at a position.
D = EVAL - REAL. Floors: the same statistic computed with each of the 200 random set pairs (randE_j, randR_j)."""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
BAND = list(range(20, 61, 4))
LENS_MAIN = "jlens"


def load(res):
    d = dict(np.load(res / "reads_jl2.npz"))
    meta = [json.loads(l) for l in open(res / "reads_jl2_meta.jsonl")]
    layers = list(d["layers"])
    band = [layers.index(L) for L in BAND]
    return d, meta, band, list(d["lenses"]), list(d["rand"])


def level(m):
    return m["labels"]["level"] if m["kind"] == "graded" else None


def eyeball_set(meta):
    rng = random.Random(0)
    base = [i for i, m in enumerate(meta) if m["kind"] == "baseline"]
    l4 = [i for i, m in enumerate(meta) if level(m) == 4]
    real = [i for i, m in enumerate(meta) if m["kind"] == "main" and m["labels"]["real"]]
    l0 = [i for i, m in enumerate(meta) if level(m) == 0]
    return [("no-note baseline", i) for i in base] + [("L4 explicit live", i) for i in rng.sample(l4, 4)] + \
        [("main real", i) for i in rng.sample(real, 4)] + [("L0 explicit simulation (contrast)", i) for i in rng.sample(l0, 4)]


def eyeball(d, meta, band, lenses, tok, res):
    li = lenses.index(LENS_MAIN)
    lines = ["J-lens top 10 at the last prompt token (the decision point), layers 20-60 step 4; jl2 by-eye set "
             "(8 no-note baselines, 4 L4, 4 main-real, 4 L0 for contrast; seed 0)."]
    for title, i in eyeball_set(meta):
        m = meta[i]
        lines.append(f"\n=== {title} | {m['case']} | {m['vid']} | acted {m['acted']}/{m['n_ep']}")
        lines.append("    note: " + (m["note"].replace("\n", " ") or "(none)"))
        for j in band:
            ids = d["top_ids"][i, li, j, 0, :10]
            lines.append(f"    L{BAND[band.index(j)]:>2}: " + " | ".join(repr(tok.decode([int(t)]))[1:-1] for t in ids))
    (res / "eyeball.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {res / 'eyeball.txt'}")


def band_mass(d, band, li, pos, si):
    """[n] band-mean log-mass of named set si at read slot pos (0 last, 1 user_last, 'note' = mean over note tokens)."""
    if pos == "note":
        m = d["mass"][:, li][:, band][:, :, 2:, si]                 # [n, nb, P-2]
        v = d["valid"][:, 2:]                                       # [n, P-2]
        k = v.sum(-1)
        per_layer = np.where(v[:, None, :], m, 0.0).sum(-1) / np.maximum(k, 1)[:, None]
        return np.where(k > 0, per_layer.mean(-1), np.nan)
    return d["mass"][:, li, band, pos, si].mean(-1)


def band_rand(d, band, li, pos):
    """[n, 400] band-mean random-set log-mass."""
    if pos == "note":
        return d["rnote"][:, li][:, band].mean(1)
    return d["rmass"][:, li, band, pos].mean(1)


def cluster_boot(fn, groups, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    g = sorted(set(groups))
    idx = {k: np.where(np.array(groups) == k)[0] for k in g}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[g[k]] for k in rng.integers(0, len(g), len(g))])
        vals.append(fn(pick))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def metrics(d, meta, band, lenses, rand, tok, res):
    nE = sum(r.startswith("randE") for r in rand)
    kind = np.array([m["kind"] for m in meta])
    lev = np.array([level(m) if level(m) is not None else -1 for m in meta])
    real = np.array([m["kind"] == "main" and m["labels"]["real"] for m in meta])
    sim = np.array([m["kind"] == "main" and not m["labels"]["real"] for m in meta])
    cells = [m["case"] for m in meta]
    out = ["# jl2: eval/fiction tokens in the J-lens workspace of Qwen3.6-27B agent prompts (R40)", "",
           f"{len(meta)} prompts: {Counter(kind.tolist())}. Masses: band-mean (layers 20-60) log-mass. D = EVAL - REAL. "
           "Floors: 200 random set pairs of the same sizes. CIs: cell-cluster bootstrap (8 cells).", ""]
    res_json = {}
    # ---- gate
    out += ["## Gate: EVAL log-mass at the last prompt token, L0 (explicit simulation) minus L4 (explicit live)", "",
            "| lens | L0 - L4 | 95% CI | random-set p95 | rand pct | pass |", "|---|---|---|---|---|---|"]
    for lens in lenses:
        li = lenses.index(lens)
        E = band_mass(d, band, li, 0, 0)
        RE = band_rand(d, band, li, 0)[:, :nE]
        g = E[lev == 0].mean() - E[lev == 4].mean()
        gr = RE[lev == 0].mean(0) - RE[lev == 4].mean(0)
        sel = np.where((lev == 0) | (lev == 4))[0]
        Es, ls = E[sel], lev[sel]                       # bootstrap indices are positions within sel
        lo, hi = cluster_boot(lambda p: Es[p][ls[p] == 0].mean() - Es[p][ls[p] == 4].mean(), [cells[i] for i in sel])
        ok = bool(g > 0 and g > np.percentile(gr, 95))
        res_json[f"gate_{lens}"] = {"diff": float(g), "rand_p95": float(np.percentile(gr, 95)), "pass": ok}
        out.append(f"| {lens}{' (gate)' if lens == LENS_MAIN else ''} | {g:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                   f"{np.percentile(gr, 95):+.3f} | {(gr < g).mean():.2f} | {'PASS' if ok else 'fail'} |")
    gate_ok = res_json[f"gate_{LENS_MAIN}"]["pass"]
    out.append("")
    out.append(f"**Gate {'passes' if gate_ok else 'FAILS'}** (J-lens).")
    if not gate_ok:
        (res / "summary.md").write_text("\n".join(out) + "\n")
        (res / "summary.json").write_text(json.dumps(res_json, indent=1))
        print("\n".join(out))
        return res_json
    # ---- M1
    out += ["", "## M1: realism tracking", "",
            "Graded set (80 prompts, L0-L4): Spearman of D with realism level (expected negative; 'fictionality rho' = "
            "-rho). Main set: mean D(simulated) - mean D(real), 144 vs 144 prompts.", "",
            "| lens | position | graded rho(D, level) | 95% CI | rand pct (of -rho) | main sim - real | 95% CI | rand pct |",
            "|---|---|---|---|---|---|---|---|"]
    gsel = np.where(kind == "graded")[0]
    for lens in lenses:
        li = lenses.index(lens)
        for pos, pname in ((0, "last"), ("note", "note mean"), (1, "user_last")):
            D = band_mass(d, band, li, pos, 0) - band_mass(d, band, li, pos, 1)
            RA = band_rand(d, band, li, pos)
            RD = RA[:, :nE] - RA[:, nE:]
            rho = spearmanr(D[gsel], lev[gsel]).statistic
            rr = np.array([spearmanr(RD[gsel, j], lev[gsel]).statistic for j in range(RD.shape[1])])
            lo, hi = cluster_boot(lambda p: spearmanr(D[gsel][p], lev[gsel][p]).statistic, [cells[i] for i in gsel])
            ms = D[sim].mean() - D[real].mean()
            mr = RD[sim].mean(0) - RD[real].mean(0)
            msel = np.where(sim | real)[0]
            mlo, mhi = cluster_boot(lambda p: D[msel][p][sim[msel][p]].mean() - D[msel][p][real[msel][p]].mean(),
                                    [cells[i] for i in msel])
            res_json[f"M1_{lens}_{pname}"] = {"rho": float(rho), "rho_ci": [lo, hi], "rand_pct": float((-rr < -rho).mean()),
                                              "main_diff": float(ms), "main_ci": [mlo, mhi],
                                              "main_rand_pct": float((mr < ms).mean())}
            out.append(f"| {lens} | {pname} | {rho:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(-rr < -rho).mean():.2f} | "
                       f"{ms:+.3f} | [{mlo:+.3f}, {mhi:+.3f}] | {(mr < ms).mean():.2f} |")
    # per-level means (J-lens, last)
    li = lenses.index(LENS_MAIN)
    E, Rr = band_mass(d, band, li, 0, 0), band_mass(d, band, li, 0, 1)
    out += ["", "J-lens, last token, means by condition (EVAL / REAL / D):", ""]
    for name, sel in [("no-note baseline", kind == "baseline")] + [(f"graded L{k}", lev == k) for k in range(5)] + \
            [("main simulated", sim), ("main real", real)]:
        out.append(f"- {name} (n {sel.sum()}): EVAL {E[sel].mean():+.2f}, REAL {Rr[sel].mean():+.2f}, D {(E - Rr)[sel].mean():+.2f}")
    # ---- M2
    out += ["", "## M2: any EVAL token in the top 10 at any layer 20-60, last prompt token", "",
            "| lens | no-note baseline | main real | L4 | main real + L4 | (contrast) main simulated | L0 |",
            "|---|---|---|---|---|---|---|"]
    for lens in lenses:
        li = lenses.index(lens)
        hit = (d["best"][:, li, band, 0, 0] <= 10).any(1)
        cols = [kind == "baseline", real, lev == 4, real | (lev == 4), sim, lev == 0]
        res_json[f"M2_{lens}"] = {"baseline": [int(hit[cols[0]].sum()), int(cols[0].sum())],
                                  "real_L4": [int(hit[cols[3]].sum()), int(cols[3].sum())]}
        out.append(f"| {lens} | " + " | ".join(f"{hit[c].sum()}/{c.sum()} ({hit[c].mean():.0%})" for c in cols) + " |")
    # which EVAL tokens, and lexical split for the real notes
    sets = json.load(open(HERE / "data" / "tokensets.json"))
    ev = set(sets["EVAL"])
    li = lenses.index(LENS_MAIN)
    cnt = Counter()
    for i in np.where((kind == "baseline") | real | (lev == 4))[0]:
        for t in set(d["top_ids"][i, li, band, 0, :10].ravel().tolist()) & ev:
            cnt[tok.decode([t])] += 1
    out.append("")
    out.append(f"- J-lens EVAL tokens seen in the top 10 (baseline + real + L4 prompts, count of prompts): {dict(cnt.most_common())}")
    evw = re.compile(r"\b(test|testing|simulat\w*|scenario|fiction\w*|fake|pretend|exercise|drill|eval\w*|benchmark|story|hypothetical)\b", re.I)
    lexical = np.array([bool(evw.search(m["note"])) for m in meta])
    hit = (d["best"][:, li, band, 0, 0] <= 10).any(1)
    for name, sel in (("real/L4 notes with an EVAL word (e.g. 'not a test')", (real | (lev == 4)) & lexical),
                      ("real/L4 notes without one", (real | (lev == 4)) & ~lexical)):
        out.append(f"- {name}: {hit[sel].sum()}/{sel.sum()}")
    # ---- M3
    out += ["", "## M3: does D at the last prompt token predict acting? (368 over-limit prompts x 3 episodes)", "",
            "Binomial GLM (acted of 3 per prompt) on z-scored D with cell and note-condition fixed effects (9 conditions: "
            "observed x real in the main set, L0-L4 in the graded set); cluster-robust SE by cell (8 clusters, t with 7 df); "
            "permutation p: D shuffled within cell x condition, 2000 draws.", "",
            "| lens | beta per SD | 95% CI (cluster) | p (cluster, t7) | p (permutation) |", "|---|---|---|---|---|"]
    import statsmodels.api as sm
    from scipy.stats import t as tdist
    sel = np.where(kind != "baseline")[0]
    cond = [("m", meta[i]["labels"]["obs"], meta[i]["labels"]["real"]) if kind[i] == "main" else ("g", lev[i]) for i in sel]
    cl = sorted(set(cond))
    cs = sorted(set(cells[i] for i in sel))
    Xfe = np.column_stack([np.ones(len(sel))] + [[c == k for c in cond] for k in cl[1:]] +
                          [[cells[i] == k for i in sel] for k in cs[1:]]).astype(float)
    y = np.column_stack([[meta[i]["acted"] for i in sel], [meta[i]["n_ep"] - meta[i]["acted"] for i in sel]]).astype(float)
    groups = np.array([cs.index(cells[i]) for i in sel])
    strata = defaultdict(list)
    for k, i in enumerate(sel):
        strata[(cells[i], cond[k])].append(k)
    rng = np.random.default_rng(0)
    for lens in lenses:
        li = lenses.index(lens)
        D = (band_mass(d, band, li, 0, 0) - band_mass(d, band, li, 0, 1))[sel]
        z = (D - D.mean()) / D.std()
        X = np.column_stack([z, Xfe])
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": groups})
        b, se = fit.params[0], fit.bse[0]
        p = 2 * tdist.sf(abs(b / se), len(cs) - 1)
        q = tdist.ppf(0.975, len(cs) - 1)
        perm = []
        for _ in range(2000):
            zp = z.copy()
            for ks in strata.values():
                zp[ks] = z[rng.permutation(ks)]
            perm.append(sm.GLM(y, np.column_stack([zp, Xfe]), family=sm.families.Binomial()).fit().params[0])
        pp = float((np.abs(perm) >= abs(b)).mean())
        res_json[f"M3_{lens}"] = {"beta": float(b), "se": float(se), "p_cluster": float(p), "p_perm": pp}
        out.append(f"| {lens} | {b:+.3f} | [{b - q * se:+.3f}, {b + q * se:+.3f}] | {p:.3f} | {pp:.3f} |")
    (res / "summary.md").write_text("\n".join(out) + "\n")
    (res / "summary.json").write_text(json.dumps(res_json, indent=1))
    print("\n".join(out))
    return res_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(HERE / "results"))
    ap.add_argument("--eyeball", action="store_true")
    a = ap.parse_args()
    res = Path(a.res)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    d, meta, band, lenses, rand = load(res)
    if a.eyeball:
        eyeball(d, meta, band, lenses, tok, res)
        return
    metrics(d, meta, band, lenses, rand, tok, res)


if __name__ == "__main__":
    main()

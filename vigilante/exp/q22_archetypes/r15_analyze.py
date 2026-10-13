"""R15 analysis (local).   uv run python -m vigilante.exp.q22_archetypes.r15_analyze"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from vigilante.exp.q22_archetypes.archetypes import INVENTED, PAIRS, PERSONAS, PERSONAS_R15, TRAIT_OF, TRAITS

HERE = Path(__file__).parent
R12, R15 = HERE / "results", HERE / "results_r15"
H = 20
SEED = 20261013


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    L = ["# R15: follow-ups to item 22 (Qwen3-14B, hidden state 20 unless noted)", ""]
    # ---------------- persona vectors (per-prompt differences from R12's default answers), PC1 axis from R12
    z12, m12 = np.load(R12 / "acts.npz"), json.load(open(R12 / "acts_meta.json"))
    z15, m15 = np.load(R15 / "acts_r15.npz"), json.load(open(R15 / "acts_r15_meta.json"))
    li = list(z12["hs"]).index(H)
    A12, A15 = z12["acts"][:, li].astype(np.float64), z15["acts"][:, list(z15["hs"]).index(H)].astype(np.float64)
    i12 = {(m["persona"], m["qi"]): i for i, m in enumerate(m12)}
    dflt = {q: A12[i12[("default", q)]] for q in range(40)}
    D = {p: np.mean([A12[i12[(p, q)]] - dflt[q] for q in range(40)], 0) for p in PERSONAS if p != "default"}
    for i, m in enumerate(m15):
        D.setdefault(m["persona"], []).append(A15[i] - dflt[m["qi"]])
    for p in list(D):
        if isinstance(D[p], list):
            D[p] = np.mean(D[p], 0)
    base = [p for p in PERSONAS if p != "default"]
    M = np.stack([D[p] for p in base])
    mu = M.mean(0)
    pc1 = np.linalg.svd(M - mu, full_matrices=False)[2][0]
    if (0 - mu) @ pc1 < 0:
        pc1 = -pc1
    proj = {p: float((D[p] - mu) @ pc1) for p in D}
    proj["default"] = float((0 - mu) @ pc1)

    # ---------------- 2. fame vs style
    L += ["## Fame vs style (PC1 projection; assistant side positive)", "",
          "| trait | famous | famous, told to speak plainly | generic, told to speak theatrically | generic | plain recovers | florid moves |",
          "|---|---|---|---|---|---|---|"]
    rec, mov = [], []
    for t, (fam, gen) in PAIRS.items():
        f, g = proj[f"{t}_1"], proj[f"{t}_0"]
        fp, gf = proj[f"{t}_famous_plain"], proj[f"{t}_generic_florid"]
        gap = g - f
        r, m = (fp - f) / gap, (g - gf) / gap
        rec.append(r), mov.append(m)
        L.append(f"| {t} | {f:.1f} | {fp:.1f} | {gf:.1f} | {g:.1f} | {r:.0%} | {m:.0%} |")
    L += ["", f"Median share of the famous-generic gap: plain register recovers {np.median(rec):.0%}; theatrical "
          f"register moves the generic role {np.median(mov):.0%} of the way.", ""]
    # ---------------- 3. invented-name floor
    L += ["## Invented names with matched descriptions", "", "| trait | famous | invented | generic |", "|---|---|---|---|"]
    for t in INVENTED:
        f = proj.get(f"{t}_1") if t in PAIRS else None
        L.append(f"| {t} | {'-' if f is None else f'{f:.1f}'} | {proj[f'{t}_invented']:.1f} | {proj[f'{t}_0']:.1f} |")
    inv = [proj[f"{t}_invented"] for t in INVENTED]
    L += ["", f"Means: famous {np.mean([proj[f'{t}_1'] for t in PAIRS]):.1f}, invented {np.mean(inv):.1f}, generic "
          f"{np.mean([proj[f'{t}_0'] for t in INVENTED]):.1f}.", ""]

    # ---------------- 1. distance -> behaviour by prompting
    if not (R15 / "syco_r15.json").exists():
        L += ["(sycophancy / steering results not pulled yet)", ""]
        S = None
    else:
        S = json.load(open(R15 / "syco_r15.json"))
    if S is not None:
        section_behaviour(L, S, proj)
    paragraphs(L, D, dflt)
    (R15 / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def section_behaviour(L, S, proj):
    ans, sug = np.array(S["answer"]), np.array(S["suggested"])
    n = np.arange(len(ans))
    beh = {}
    for p, r in S["persona"].items():
        P0, Ps, Pp = (np.array(r[k]) for k in ("plain", "suggest", "pushback"))
        beh[p] = {"acc": P0[n, ans].mean(), "suggest": (Ps[n, sug] - P0[n, sug]).mean(),
                  "pushback": (P0[n, ans] - Pp[n, ans]).mean()}
    ref = {}
    if (R15 / "refusal_labels.json").exists():
        lab = json.load(open(R15 / "refusal_labels.json"))["haiku"]
        for k, v in lab.items():
            p = k.split("|")[0]
            if v:
                ref.setdefault(p, []).append(v != "complied")
        for p in ref:
            beh[p]["overrefuse"] = float(np.mean(ref[p]))
    ps = [p for p in PERSONAS_R15 if p in beh]
    x = [proj[p] for p in ps]
    L += ["## Does distance from the assistant predict behaviour? (68 personas, plain prompting, no steering)", "",
          "Spearman with PC1 projection (positive = closer to the assistant goes with more of the behaviour):", ""]
    for k in ("acc", "suggest", "pushback", "overrefuse"):
        if all(k in beh[p] for p in ps):
            rho, pv = spearmanr(x, [beh[p][k] for p in ps])
            L.append(f"- {k}: rho {rho:+.2f} (p {pv:.3f}); range {min(beh[p][k] for p in ps):.3f} to "
                     f"{max(beh[p][k] for p in ps):.3f}")
    grp = {"default": ["default"], "helpful_assistant": ["helpful_assistant"],
           "generic roles": [p for p in TRAIT_OF if not (p.endswith("_1") and TRAIT_OF[p] in PAIRS)],
           "famous": [f"{t}_1" for t in PAIRS], "famous, plain": [f"{t}_famous_plain" for t in PAIRS],
           "generic, theatrical": [f"{t}_generic_florid" for t in PAIRS], "invented": [f"{t}_invented" for t in INVENTED],
           "non-assistant": [p for p in PERSONAS if p.startswith("other_")]}
    L += ["", "| group | PC1 | plain acc | suggest shift | pushback drop | over-refusal |", "|---|---|---|---|---|---|"]
    for gname, members in grp.items():
        mm = [p for p in members if p in beh]
        orf = [beh[p]["overrefuse"] for p in mm if "overrefuse" in beh[p]]
        L.append(f"| {gname} | {np.mean([proj[p] for p in mm]):.1f} | {np.mean([beh[p]['acc'] for p in mm]):.3f} | "
                 f"{np.mean([beh[p]['suggest'] for p in mm]):+.3f} | {np.mean([beh[p]['pushback'] for p in mm]):+.3f} | "
                 f"{'-' if not orf else f'{np.mean(orf):.1%}'} |")

    # ---------------- 5. full-strength steering with coherence
    L += ["", "## Full-strength steering (all vectors at Jeeves's norm 21.4), no persona prompt", "",
          "| condition | plain acc | suggest shift | pushback drop | coherence NLL (unsteered model) |", "|---|---|---|---|---|"]
    for name, r in S["steer"].items():
        P0, Ps, Pp = (np.array(r[k]) for k in ("plain", "suggest", "pushback"))
        L.append(f"| {name} | {P0[n, ans].mean():.3f} | {(Ps[n, sug] - P0[n, sug]).mean():+.3f} | "
                 f"{(P0[n, ans] - Pp[n, ans]).mean():+.3f} | {np.mean(r['coherence_nll']):.2f} |")



def paragraphs(L, D, dflt):
    rng = np.random.default_rng(SEED)
    # ---------------- 4. per-paragraph reasoning
    paras = [json.loads(l) for l in open(R15 / "paragraphs_r15.jsonl")]
    ci = [10, 20, 30].index(H)
    own = {"service_1": "service_1", "expertise_1": "expertise_1", "expertise_2": "expertise_2", "other_0": "other_0"}
    rng = np.random.default_rng(SEED)
    L += ["", "## Reasoning per paragraph", ""]
    for p, dname in list(own.items()) + [("default", "service_1"), ("default", "expertise_1"), ("default", "expertise_2")]:
        rows = [r for r in paras if r["persona"] == p]
        if not rows:
            continue
        d = unit(D[dname])
        X = np.array([r["act"][ci] for r in rows], dtype=np.float64)
        pr = (X - dflt_mean(dflt)) @ d
        R = rng.standard_normal((200, X.shape[1]))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        rp = (X - dflt_mean(dflt)) @ R.T
        th = np.array([r["region"] == "thinking" for r in rows])
        sd = rp.std()
        L.append(f"### {p} runs, projection on the {dname} direction ({PERSONAS[dname][8:60]})")
        L.append(f"thinking paragraphs n {th.sum()}, mean {pr[th].mean():.2f}; answers n {(~th).sum()}, mean "
                 f"{pr[~th].mean():.2f}; random-direction sd of paragraph projections {sd:.2f}")
        order = np.argsort(-pr)
        for j in [k for k in order if th[k]][:5]:
            L.append(f"- {pr[j]:.1f}: {rows[j]['text'][:220]!r}")
        L.append("")


def dflt_mean(dflt):
    return np.mean(list(dflt.values()), 0)


if __name__ == "__main__":
    main()

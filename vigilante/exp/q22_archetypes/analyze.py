"""R12 / item 22 analysis (local).   uv run python -m vigilante.exp.q22_archetypes.analyze"""
import json
from pathlib import Path

import numpy as np

from vigilante.exp.q22_archetypes.archetypes import PERSONAS, TRAIT_OF, TRAITS

RES = Path(__file__).parent / "results"
H = 20
SEED = 20261013


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def eff_rank(X):
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def main():
    z = np.load(RES / "acts.npz")
    acts, think, hs = z["acts"].astype(np.float64), z["think"].astype(np.float64), list(z["hs"])
    meta = json.load(open(RES / "acts_meta.json"))
    li = hs.index(H)
    pm = {p: acts[[i for i, m in enumerate(meta) if m["persona"] == p], li].mean(0) for p in PERSONAS}
    nondef = [p for p in PERSONAS if p != "default"]
    grand = np.mean([pm[p] for p in nondef], 0)
    a = unit(pm["default"] - grand)
    arch = sorted(TRAIT_OF)
    others = [p for p in PERSONAS if p.startswith("other_")]
    proj = {p: float((pm[p] - grand) @ a) for p in nondef}
    cosa = {p: float(unit(pm[p] - grand) @ a) for p in nondef}
    out, L = {}, ["# R12 / item 22: assistant archetypes (Qwen3-14B, hidden state 20)", ""]

    # 1. axis separates assistant archetypes from non-assistant characters
    med = np.median([proj[p] for p in arch])
    L += ["## Projection on the assistant axis (persona mean - grand mean)", "",
          f"Archetype median {med:.2f}; helpful_assistant {proj['helpful_assistant']:.2f}; non-assistant characters: "
          + ", ".join(f"{p} {proj[p]:.2f}" for p in others), "",
          f"Non-assistant below the archetype median: {sum(proj[p] < med for p in others)}/{len(others)}", "",
          "| rank | archetype | trait | projection | cosine with axis |", "|---|---|---|---|---|"]
    for i, p in enumerate(sorted(arch, key=proj.get, reverse=True)):
        L.append(f"| {i + 1} | {p} ({PERSONAS[p][7:-1]}) | {TRAIT_OF[p]} | {proj[p]:.2f} | {cosa[p]:.2f} |")
    # PCA check: is PC1 of the persona means the axis?
    M = np.stack([pm[p] for p in nondef + ["default"]])
    Mc = M - M.mean(0)
    pc1 = np.linalg.svd(Mc, full_matrices=False)[2][0]
    L += ["", f"|cos(PC1 of all 42 persona means, assistant axis)| = {abs(pc1 @ a):.2f}", ""]

    # 2. trait grouping: within- vs between-trait cosine of centred archetype vectors, label-shuffle floor
    V = {p: unit(pm[p] - grand) for p in arch}
    C = np.array([[V[p] @ V[q] for q in arch] for p in arch])
    lab = np.array([TRAIT_OF[p] for p in arch])

    def gap(labels):
        same = labels[:, None] == labels[None, :]
        off = ~np.eye(len(labels), dtype=bool)
        return C[same & off].mean() - C[~same].mean()
    g0 = gap(lab)
    rng = np.random.default_rng(SEED)
    null = np.array([gap(rng.permutation(lab)) for _ in range(5000)])
    L += ["## Do roles group by trait?", "",
          f"within-trait minus between-trait cosine = {g0:.3f}; label-shuffle 95th = {np.percentile(null, 95):.3f}; "
          f"permutation p = {(null >= g0).mean():.4f}", ""]
    for t in TRAITS:
        ps = [p for p in arch if TRAIT_OF[p] == t]
        w = np.mean([V[p] @ V[q] for i, p in enumerate(ps) for q in ps[i + 1:]])
        L.append(f"- {t}: mean within-trait cosine {w:.2f}")
    # effective rank: archetypes vs non-assistant characters (equal-size subsamples of 10)
    er_arch = [eff_rank(np.stack([pm[p] for p in rng.choice(arch, 10, replace=False)])) for _ in range(500)]
    er_oth = eff_rank(np.stack([pm[p] for p in others]))
    L += ["", f"Effective rank (participation ratio) of 10 persona means: archetypes median {np.median(er_arch):.2f} "
          f"[5-95%: {np.percentile(er_arch, 5):.2f}-{np.percentile(er_arch, 95):.2f}], non-assistant characters {er_oth:.2f}", ""]

    # 3. thinking vs answer (default, thinking on): projections on each archetype direction (persona mean - default)
    th, an = think[:, 0, li], think[:, 1, li]
    ok = ~(np.isnan(th).any(1) | np.isnan(an).any(1))
    th, an = th[ok], an[ok]
    dirs = {p: unit(pm[p] - pm["default"]) for p in arch}
    R = rng.standard_normal((1000, th.shape[1]))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    diff = (th - an)
    rd = np.abs(diff.mean(0) @ R.T) / (diff @ R.T).std(0)
    L += ["## Default assistant: reasoning vs answer, by archetype direction", "",
          f"n = {ok.sum()} questions with both regions. Effect = mean(thinking - answer) projection / sd across "
          f"questions (paired d). Random-direction 95th |d| = {np.percentile(rd, 95):.2f}.", "",
          "| archetype | trait | paired d (thinking - answer) |", "|---|---|---|"]
    dd = {}
    for p in arch:
        x = diff @ dirs[p]
        dd[p] = float(x.mean() / x.std())
    for p in sorted(arch, key=dd.get, reverse=True):
        L.append(f"| {p} | {TRAIT_OF[p]} | {dd[p]:.2f} |")
    by_trait = {t: float(np.mean([dd[p] for p in arch if TRAIT_OF[p] == t])) for t in TRAITS}
    L += ["", "By trait (mean paired d): " + ", ".join(f"{t} {v:.2f}" for t, v in sorted(by_trait.items(), key=lambda x: -x[1])), ""]

    # 4. steering and sycophancy
    S = json.load(open(RES / "steer.json"))
    sel = json.load(open(RES / "selection.json"))
    ans, sug = np.array(S["answer"]), np.array(S["suggested"])
    L += ["## Steering toward archetypes: sycophancy (letter probabilities, 200 MCQ items)", "",
          f"Selected (pre-registered rule): top 3 {sel['chosen'][:3]}, bottom 1 {sel['chosen'][3]}.", "",
          "`acc` = mean P(correct) on plain; `suggest shift` = mean P(suggested wrong) with the suggestion minus on "
          "plain; `pushback drop` = mean P(correct) plain minus after 'Are you sure?'.", "",
          "| condition | acc | suggest shift | pushback drop |", "|---|---|---|---|"]
    rows = {}
    for tag, r in S["res"].items():
        P0, Ps, Pp = (np.array(r[k]) for k in ("plain", "suggest", "pushback"))
        n = np.arange(len(ans))
        rows[tag] = (P0[n, ans].mean(), (Ps[n, sug] - P0[n, sug]).mean(), (P0[n, ans] - Pp[n, ans]).mean())
        L.append(f"| {tag} | {rows[tag][0]:.3f} | {rows[tag][1]:+.3f} | {rows[tag][2]:+.3f} |")
    L += ["", "Coherence samples (first 160 chars, first prompt):", ""]
    for tag, r in S["res"].items():
        L.append(f"- {tag}: {r['coherence'][0][:160]!r}")
    out.update({"proj": proj, "cos_axis": cosa, "group_gap": g0, "group_p": float((null >= g0).mean()),
                "think_minus_answer_d": dd, "steer": {k: list(v) for k, v in rows.items()}})
    (RES / "summary.md").write_text("\n".join(L) + "\n")
    (RES / "numbers.json").write_text(json.dumps(out, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
